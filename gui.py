import sys
import os
import numpy as np
import pickle
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QLineEdit, QSlider, QLabel, QSpinBox, 
                             QFileDialog, QTextEdit, QTabWidget, QFrame, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from tube_detection import locate_pcr_tubes, calculate_rotation_angle, rotate_point, infer_missing_tubes, detect_inner_circles
from freezing_detection import load_brightness_timeseries, load_temperature_timeseries, get_freezing_temperature
import cv2
import traceback
import io

class StreamToTextEdit(io.StringIO):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.append(text)

class BrightnessWorker(QObject):
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, image_directory, tube_location_file, temperature_recording_file):
        super().__init__()
        self.image_directory = image_directory
        self.tube_location_file = tube_location_file
        self.temperature_recording_file = temperature_recording_file

    def run(self):
        temperature_recordings = load_temperature_timeseries(self.temperature_recording_file)
        self.progress.emit(5)  # Emit progress update
        
        brightness_timeseries = load_brightness_timeseries(
            self.image_directory, 
            self.tube_location_file, 
            temperature_recordings,
            log_callback=self.log.emit  # Pass the log signal as a callback
        )
        
        self.progress.emit(100)  # Emit progress update
        self.finished.emit(temperature_recordings, brightness_timeseries)
        
class InteractivePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Droplet Freezing Assay Offline Analysis')
        self.setGeometry(100, 100, 1400, 800)

        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 创建第一个标签页
        self.tab1 = QWidget()
        self.tab_widget.addTab(self.tab1, "Tube Locating")

        # 创建第二个标签页
        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab2, "Freezing Detection")

        # 设置第一个标签页的布局
        self.setup_tube_locating_tab()

        # 设置第二个标签页的布局
        self.setup_freezing_detection_tab()

        # 初始化更新定时器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.plot_tube_detection_results)

        # 存储 PCR tubes 和 inner circles
        self.pcr_tubes = []
        self.inner_circles = []
        self.img = None

        # 绘制初始图表
        self.plot_tube_detection_results()

    def create_selection_group(self, title, button_text, selection_method):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        
        button = QPushButton(button_text)
        button.clicked.connect(selection_method)
        
        label = QLabel("Current: Not selected")
        label.setWordWrap(True)
        
        layout.addWidget(button)
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group, label

    def setup_tube_locating_tab(self):
        tab1_layout = QHBoxLayout(self.tab1)

        # 创建左侧的图表部件和布局
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        
        # 连接鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        # 创建右侧的控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Image Path
        # self.image_path_input = QLineEdit()
        # self.image_path_input.setPlaceholderText("Enter image path")
        # self.image_path_input.setText('200/img/IMG00000000000001975884.png')
        # self.image_path_input.textChanged.connect(self.schedule_update)
        # right_layout.addWidget(QLabel("Image Path:"))
        # right_layout.addWidget(self.image_path_input)
        self.sample_image_path_button = QPushButton("Select an Image")
        self.sample_image_path_button.clicked.connect(self.select_sample_image_path)
        self.sample_image_path = '200/img/IMG00000000000001975884.png'  # a default
        self.sample_image_path_label = QLabel(f"Current image: {os.path.basename(self.sample_image_path)}")
        right_layout.addWidget(self.sample_image_path_label)
        right_layout.addWidget(self.sample_image_path_button)
        

        # Rotate Input
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Enter rotation value")
        self.rotation_input.setText('auto')
        self.rotation_input.textChanged.connect(self.schedule_update)
        right_layout.addWidget(QLabel("Rotation:"))
        right_layout.addWidget(self.rotation_input)

        # Min Area
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setMinimum(10)
        self.min_area_slider.setMaximum(500)
        self.min_area_slider.setSingleStep(10)
        self.min_area_slider.setValue(100)
        self.min_area_label = QLabel("Min Area: 100")
        self.min_area_slider.valueChanged.connect(self.update_min_area)
        right_layout.addWidget(self.min_area_label)
        right_layout.addWidget(self.min_area_slider)

        # Circularity
        self.circularity_slider = QSlider(Qt.Horizontal)
        self.circularity_slider.setMinimum(10)
        self.circularity_slider.setMaximum(100)
        self.circularity_slider.setSingleStep(5)
        self.circularity_slider.setValue(20)
        self.circularity_label = QLabel("Circularity: 0.20")
        self.circularity_slider.valueChanged.connect(self.update_circularity)
        right_layout.addWidget(self.circularity_label)
        right_layout.addWidget(self.circularity_slider)

        # 添加保存按钮
        self.save_button = QPushButton("Save Inner Circles")
        self.save_button.clicked.connect(self.save_inner_circles)
        right_layout.addWidget(self.save_button)

        # 添加文本框
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self.log_text_edit, 1)  # 添加拉伸因子
        
        # 重定向标准输出到文本框
        sys.stdout = StreamToTextEdit(self.log_text_edit)
        
        # 拉伸一下
        right_layout.setSpacing(10)

        # 将部件添加到主布局
        tab1_layout.addWidget(left_widget, 2)
        tab1_layout.addWidget(right_widget, 1)

    def setup_freezing_detection_tab(self):
        tab2_layout = QHBoxLayout(self.tab2)

        # 创建左侧的图表部件
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.figure2 = Figure(figsize=(5, 4), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        left_layout.addWidget(self.toolbar2)
        left_layout.addWidget(self.canvas2)

        self.ax2 = self.figure2.add_subplot(111)

        # 创建右侧的控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 文件夹选择
        image_dir_group, self.image_directory_label = self.create_selection_group(
            "Image Directory", "Select", self.select_image_directory)
        right_layout.addWidget(image_dir_group)

        # 文件选择1
        temp_rec_group, self.temperature_recording_label = self.create_selection_group(
            "Temperature Recording", "Select", self.select_temperature_recording)
        right_layout.addWidget(temp_rec_group)

        # 文件选择2
        tube_loc_group, self.tube_locations_label = self.create_selection_group(
            "Tube Locations", "Select", self.select_tube_locations)
        right_layout.addWidget(tube_loc_group)
        
        # Add a start button
        self.start_load_timeseries_button = QPushButton("Start Analysis")
        self.start_load_timeseries_button.clicked.connect(self.load_brightness_series)
        right_layout.addWidget(self.start_load_timeseries_button)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(separator)

        # 上一个和下一个按钮
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.previous_tube)
        self.next_button.clicked.connect(self.next_tube)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        right_layout.addLayout(button_layout)

        # 输入框
        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("Enter tube number")
        self.value_input.returnPressed.connect(self.go_to_tube)
        right_layout.addWidget(QLabel("Go to tube:"))
        right_layout.addWidget(self.value_input)

        # 保存按钮
        self.save_button_freezing_temperatures = QPushButton("Save Freezing Temperatures")
        self.save_button_freezing_temperatures.clicked.connect(self.save_freezing_events_data)
        self.load_button_freezing_temperatures = QPushButton("Load Freezing Temperatures")
        self.load_button_freezing_temperatures.clicked.connect(self.load_freezing_events_data)
        right_layout.addWidget(self.save_button_freezing_temperatures)
        right_layout.addWidget(self.load_button_freezing_temperatures)

        # 日志窗口
        self.log_text_edit2 = QTextEdit()
        self.log_text_edit2.setReadOnly(True)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self.log_text_edit2, 1)

        # 将部件添加到主布局
        tab2_layout.addWidget(left_widget, 2)
        tab2_layout.addWidget(right_widget, 1)
        
        # Disable controls initially
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.value_input.setEnabled(False)
        self.save_button_freezing_temperatures.setEnabled(False)
        self.load_button_freezing_temperatures.setEnabled(False)
        
        # default setting
        self.image_directory = '200/img'
        self.tube_location_file = '200/inner_circles_20240823_112026.pkl'
        self.temperature_recording_file = '200/t.xlsx'
    
    def select_sample_image_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select an Image")
        if file:
            self.sample_image_path = file
            self.sample_image_path_label.setText(f"Current image: {os.path.basename(self.sample_image_path)}")
            self.log_text_edit.append(f"Temperature Recording: {file}")

    def select_image_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if folder:
            self.image_directory = folder
            self.image_directory_label.setText(f"Current: {os.path.basename(folder)}")
            self.log_text_edit2.append(f"Image Dir: {folder}")

    def select_temperature_recording(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Temperature Recording")
        if file:
            self.temperature_recording_file = file
            self.temperature_recording_label.setText(f"Current: {os.path.basename(file)}")
            self.log_text_edit2.append(f"Temperature Recording: {file}")

    def select_tube_locations(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Tube Locations")
        if file:
            self.tube_location_file = file
            self.tube_locations_label.setText(f"Current: {os.path.basename(file)}")
            self.log_text_edit2.append(f"Tube Locations: {file}")

    def plot_tube_detection_results(self):
        self.ax.clear()
        try:
            min_area = self.min_area_slider.value()
            circularity_threshold = self.circularity_slider.value() / 100
            image_path = self.sample_image_path
            rotation = self.rotation_input.text()

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            self.pcr_tubes, self.img = locate_pcr_tubes(image_path, min_area, circularity_threshold)
            self.all_tubes = infer_missing_tubes(self.pcr_tubes, self.img.shape, rotate=rotation)
            self.inner_circles = detect_inner_circles(self.img, self.all_tubes)
            
            img_with_tubes = self.img.copy()
            for tube, inner_circle in zip(self.all_tubes, self.inner_circles):
                color = (0, 255, 0) if 'inferred' not in tube else (0, 0, 255)
                cv2.circle(img_with_tubes, (tube['x'], tube['y']), tube['radius'], color, 2)
                
                if inner_circle:
                    cv2.circle(img_with_tubes, (inner_circle['x'], inner_circle['y']), inner_circle['radius'], (0, 0, 0), 1)

            self.ax.imshow(cv2.cvtColor(img_with_tubes, cv2.COLOR_BGR2RGB))
            self.ax.set_title(f"Detected PCR Tubes: {len(self.pcr_tubes)}, Inferred: {len(self.all_tubes) - len(self.pcr_tubes)}")
            self.ax.axis('off')

            print(f"Detected {len(self.pcr_tubes)} PCR tubes")
            print(f"Inferred {len(self.all_tubes) - len(self.pcr_tubes)} additional tubes")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            self.ax.text(0.5, 0.5, error_msg, 
                         ha='center', va='center', wrap=True,
                         bbox=dict(facecolor='red', alpha=0.2))
            self.ax.set_title("An error occurred")
            self.ax.axis('off')
            print(error_msg)

        finally:
            self.canvas.draw()

    def on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # 左键点击
            # 找到最近的 PCR tube 和 inner circle
            if self.pcr_tubes:
                closest_tube = min(self.pcr_tubes, key=lambda t: ((t['x'] - x)**2 + (t['y'] - y)**2)**0.5)
                tube_distance = ((closest_tube['x'] - x)**2 + (closest_tube['y'] - y)**2)**0.5
            else:
                closest_tube = None
                tube_distance = float('inf')

            if self.inner_circles:
                closest_circle = min(self.inner_circles, key=lambda c: ((c['x'] - x)**2 + (c['y'] - y)**2)**0.5)
                circle_distance = ((closest_circle['x'] - x)**2 + (closest_circle['y'] - y)**2)**0.5
            else:
                closest_circle = None
                circle_distance = float('inf')

            # 删除距离最近的对象
            if tube_distance < circle_distance and closest_tube:
                self.pcr_tubes.remove(closest_tube)
                print(f"Removed tube at ({closest_tube['x']}, {closest_tube['y']})")
            elif circle_distance < tube_distance and closest_circle:
                self.inner_circles.remove(closest_circle)
                print(f"Removed inner circle at ({closest_circle['x']}, {closest_circle['y']})")
            else:
                print("No tube or circle close enough to remove")
        elif event.button == 3:  # 右键点击
            # 添加新的 inner circle
            new_circle = {'x': int(x), 'y': int(y), 'radius': 5}  # 使用固定半径，你可以根据需要调整
            self.inner_circles.append(new_circle)
            print(f"Added new inner circle at ({new_circle['x']}, {new_circle['y']})")

        # 重新绘制图表
        self.redraw_tube_detection_results()

    def redraw_tube_detection_results(self):
        self.ax.clear()
        img_with_tubes = self.img.copy()

        for tube in self.pcr_tubes:
            cv2.circle(img_with_tubes, (tube['x'], tube['y']), tube['radius'], (0, 255, 0), 2)

        for circle in self.inner_circles:
            cv2.circle(img_with_tubes, (circle['x'], circle['y']), circle['radius'], (0, 0, 0), 1)

        self.ax.imshow(cv2.cvtColor(img_with_tubes, cv2.COLOR_BGR2RGB))
        self.ax.set_title(f"PCR Tubes: {len(self.pcr_tubes)}, Inner Circles: {len(self.inner_circles)}")
        self.ax.axis('off')
        self.canvas.draw()
        print(f"Redrawn: PCR Tubes: {len(self.pcr_tubes)}, Inner Circles: {len(self.inner_circles)}")

    def schedule_update(self):
        self.update_timer.start(300)  # 300ms 延迟

    def update_min_area(self, value):
        self.min_area_label.setText(f"Min Area: {value}")
        self.schedule_update()
        print(f"Min Area updated to {value}")

    def update_circularity(self, value):
        self.circularity_label.setText(f"Circularity: {value/100:.2f}")
        self.schedule_update()
        print(f"Circularity updated to {value/100:.2f}")

    def save_inner_circles(self):
        # 生成默认文件名
        default_filename = f"inner_circles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # 组合默认的完整文件路径
        default_filepath = os.path.join(".", default_filename)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Inner Circles", default_filepath, "Pickle Files (*.pkl)")
        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(self.inner_circles, f)
            print(f"Inner circles saved to {file_path}")

    def save_freezing_events_data(self):
        # Generate default filename
        default_filename = f"freezing_temperatures_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Combine default full file path
        default_filepath = os.path.join(".", default_filename)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Freezing Temperatures", default_filepath, "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write("Tube,Temperature,Timestamp\n")  # Header
                for tube, data in self.freezing_temperatures.items():
                    temperature = data['temperature']
                    timestamp = data['timestamp']
                    # Convert timestamp to a readable date-time format
                    datetime_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{tube},{temperature:.4f},{datetime_str}\n")
            
            self.log_text_edit2.append(f"Freezing Temperatures saved to {file_path}")

    def load_freezing_events_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Freezing Temperatures", ".", "Text Files (*.txt)")
        if file_path:
            try:
                self.freezing_temperatures = {}
                with open(file_path, 'r') as f:
                    next(f)  # Skip the header line
                    for line in f:
                        tube, temperature, datetime_str = line.strip().split(',')
                        tube = int(tube)
                        temperature = float(temperature)
                        timestamp = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').timestamp()
                        
                        self.freezing_temperatures[tube] = {
                            'temperature': temperature,
                            'timestamp': timestamp
                        }
                
                self.log_text_edit2.append(f"Freezing Temperatures loaded from {file_path}")
                self.log_text_edit2.append(f"Loaded data for {len(self.freezing_temperatures)} tubes")
                
                # Update the plot if data is currently displayed
                if hasattr(self, 'current_tube'):
                    self.update_brightness_timeseries_plot()
            
            except Exception as e:
                self.log_text_edit2.append(f"Error loading file: {str(e)}")
        else:
            self.log_text_edit2.append("No file selected")

    def load_brightness_series(self):
        # Disable the start button to prevent multiple threads
        self.start_load_timeseries_button.setEnabled(False)
        
        # Create a QThread object
        self.thread = QThread()
        # Create a worker object
        self.worker = BrightnessWorker(self.image_directory, self.tube_location_file, self.temperature_recording_file)
        # Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_results)
        self.worker.log.connect(self.update_subprocess_log)
        # Start the thread
        self.thread.start()

        # Final resets
        self.thread.finished.connect(
            lambda: self.start_load_timeseries_button.setEnabled(True)
        )
        self.thread.finished.connect(
            lambda: self.log_text_edit2.append("Analysis completed!")
        )

    def update_progress(self, value):
        self.log_text_edit2.append(f"Progress: {value}%")

    def process_results(self, temperature_recordings, brightness_timeseries):
        self.temperature_recordings = temperature_recordings
        self.brightness_timeseries = brightness_timeseries
        self.freezing_temperatures = get_freezing_temperature(
            self.temperature_recordings,
            self.brightness_timeseries
        )
        self.num_tubes = len(self.inner_circles)
        self.current_tube = 0
        self.log_text_edit2.append("Data loaded successfully!")
        self.update_brightness_timeseries_plot()
        self.enable_controls()

    def update_subprocess_log(self, message):
        self.log_text_edit2.append(message)
        
    def enable_controls(self):
        self.next_button.setEnabled(True)
        self.prev_button.setEnabled(True)
        self.value_input.setEnabled(True)
        self.save_button_freezing_temperatures.setEnabled(True)
        self.load_button_freezing_temperatures.setEnabled(True)

    def next_tube(self):
        if self.current_tube < self.num_tubes - 1:
            self.current_tube += 1
            self.update_brightness_timeseries_plot()

    def previous_tube(self):
        if self.current_tube > 0:
            self.current_tube -= 1
            self.update_brightness_timeseries_plot()

    def go_to_tube(self):
        try:
            tube_number = int(self.value_input.text())
            if 0 <= tube_number < self.num_tubes:
                self.current_tube = tube_number
                self.update_brightness_timeseries_plot()
            else:
                self.log_text_edit2.append(f"Invalid tube number. Please enter a number between 0 and {self.num_tubes - 1}")
        except ValueError:
            self.log_text_edit2.append("Please enter a valid integer")
            
    def update_brightness_timeseries_plot(self):
        try:
            if not hasattr(self, 'brightness_timeseries') or not self.brightness_timeseries:
                self.log_text_edit2.append("No data loaded. Please run the analysis first.")
                return

            if self.current_tube < len(self.inner_circles):
                self.ax2.clear()

                # Align temperature and brightness data
                common_timestamps = np.intersect1d(self.temperature_recordings['timestamp'], 
                                                   self.brightness_timeseries['timestamp'])
                temp_indices = np.searchsorted(self.temperature_recordings['timestamp'], common_timestamps)
                bright_indices = np.searchsorted(self.brightness_timeseries['timestamp'], common_timestamps)

                self.current_tube_temperature = self.temperature_recordings['temperature'][temp_indices]
                self.current_tube_brightness = self.brightness_timeseries[self.current_tube][bright_indices]
                self.current_tube_timestamps = common_timestamps

                # Plot brightness vs temperature
                self.line, = self.ax2.plot(self.current_tube_temperature, self.current_tube_brightness, 'b-')
                self.ax2.invert_xaxis()
                self.ax2.set_xlabel("Temperature (°C)")
                self.ax2.set_ylabel("Brightness")
                self.ax2.set_title(f"Brightness vs Temperature for Tube {self.current_tube}")
                self.ax2.set_xlim((0, self.temperature_recordings['temperature'].min()))

                # Mark freezing event with a red dot
                self.update_freezing_point()

                # Create SpanSelector
                self.span = SpanSelector(
                    self.ax2, self.on_brightness_span_select, 'horizontal', useblit=True,
                    props=dict(alpha=0.5, facecolor='red'),
                    interactive=True, drag_from_anywhere=True
                )

                self.canvas2.draw()
                self.log_text_edit2.append(f"Displaying data for tube {self.current_tube}")
            else:
                self.log_text_edit2.append(f"Invalid index: {self.current_tube}")
        
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            self.ax2.text(0.5, 0.5, error_msg, 
                         ha='center', va='center', wrap=True,
                         bbox=dict(facecolor='red', alpha=0.2))
            self.ax2.set_title("An error occurred")
            self.ax2.axis('off')
            self.log_text_edit2.append(error_msg)

        finally:
            self.canvas2.draw()

    def update_freezing_point(self, xmin=None, xmax=None):
        if xmin is None and xmax is None:
            freezing_timestamp = self.freezing_temperatures[self.current_tube]['timestamp']
            freezing_temp_index = np.argmin(np.abs(self.current_tube_timestamps - freezing_timestamp))
            freezing_temp = self.current_tube_temperature[freezing_temp_index]
            freezing_brightness = self.current_tube_brightness[freezing_temp_index]
        else:
            # Recalculate freezing point within the selected range
            mask = (self.current_tube_temperature >= xmin) & (self.current_tube_temperature <= xmax)
            if np.sum(mask) >= 3:
                temp_range = self.current_tube_temperature[mask]
                bright_range = self.current_tube_brightness[mask]
                time_range = self.current_tube_timestamps[mask]
                
                # Calculate the derivative (rate of change) of brightness
                brightness_derivative = np.diff(bright_range)
                
                # Find the index of the largest decrease
                freezing_index = np.argmin(brightness_derivative)
                freezing_temp = temp_range[freezing_index]
                freezing_brightness = bright_range[freezing_index]
                freezing_timestamp = time_range[freezing_index]

                # Update the freezing temperature for this tube
                self.freezing_temperatures[self.current_tube]['temperature'] = freezing_temp
                self.freezing_temperatures[self.current_tube]['timestamp'] = freezing_timestamp
            else:
                self.log_text_edit2.append("Selected range is too small. Please select a larger range.")
                return

        if freezing_temp is not None:            
            # Remove old freezing point if it exists
            if hasattr(self, 'freezing_point'):
                self.freezing_point.remove()
            
            # Plot new freezing point
            self.freezing_point, = self.ax2.plot(freezing_temp, freezing_brightness, 'ro', markersize=10, 
                                                 label=f"Freezing Point: {freezing_temp:.2f}°C")
            self.ax2.legend()
            self.canvas2.draw()
            self.log_text_edit2.append(f"Updated freezing point for tube {self.current_tube}: {freezing_temp:.2f}°C at timestamp {freezing_timestamp}")
        else:
            self.log_text_edit2.append(f"No freezing point detected for tube {self.current_tube}")

    def on_brightness_span_select(self, xmin, xmax):
        self.update_freezing_point(xmin, xmax)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = InteractivePlot()
    main_window.show()
    sys.exit(app.exec_())