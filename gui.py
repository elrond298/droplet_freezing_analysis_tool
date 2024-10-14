import sys
import os
import numpy as np
import pandas as pd
import pickle
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QLineEdit, QSlider, QLabel, QSpinBox, 
                             QFileDialog, QTextEdit, QTabWidget, QFrame, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, RectangleSelector

from tube_detection import locate_pcr_tubes, calculate_rotation_angle, rotate_point, infer_missing_tubes, detect_inner_circles
from freezing_detection import load_brightness_timeseries, load_temperature_timeseries, get_freezing_temperature
import cv2
import traceback
import io

class StreamToTextEdit(io.StringIO):
    """
    A custom stream class that redirects output to a QTextEdit widget in a specific tab.

    This class inherits from io.StringIO and overrides the write method to emit a signal
    with the text to be written and the tab number. This allows the text to be displayed
    in the appropriate log text edit widget within the specified tab.
    """
    def __init__(self, signal, tab_number):
        super().__init__()
        self.signal = signal
        self.tab_number = tab_number

    def write(self, text):
        """
        Writes the given text to the log text edit widget.

        This method emits a signal with the provided text and the tab number to update
        the log text edit widget in the corresponding tab.

        Args:
            text (str): The text to be written to the log text edit widget.
        """
        self.signal.emit(text, self.tab_number)

class BrightnessWorker(QObject):
    """
    A worker class for processing brightness timeseries data in a separate thread.

    This class is responsible for loading temperature recordings and brightness timeseries
    data from the specified directories and files. It emits progress updates and logs
    messages during the process.

    Attributes:
        finished (pyqtSignal): Signal emitted when the processing is finished, carrying
                               the temperature recordings and brightness timeseries.
        progress (pyqtSignal): Signal emitted to update the progress of the processing.
        log (pyqtSignal): Signal emitted to log messages during the processing.
    """
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, image_directory, tube_location_file, temperature_recording_file):
        """
        Initializes the BrightnessWorker with the specified directories and files.

        Args:
            image_directory (str): The directory containing the image files.
            tube_location_file (str): The file containing the tube locations.
            temperature_recording_file (str): The file containing the temperature recordings.
        """
        super().__init__()
        self.image_directory = image_directory
        self.tube_location_file = tube_location_file
        self.temperature_recording_file = temperature_recording_file

    def run(self):
        """
        Executes the main processing logic for loading temperature and brightness timeseries data.

        This method loads the temperature recordings and brightness timeseries data,
        emits progress updates, and logs messages during the process. Once the processing
        is complete, it emits the finished signal with the loaded data.
        """
        temperature_recordings = load_temperature_timeseries(self.temperature_recording_file)
        self.progress.emit(5)  # Emit progress update
        
        brightness_timeseries = load_brightness_timeseries(
            self.image_directory, 
            self.tube_location_file, 
            temperature_recordings,
            log_callback=lambda msg: self.log.emit(msg)  # Pass the log signal as a callback
        )
        
        self.progress.emit(100)  # Emit progress update
        self.finished.emit(temperature_recordings, brightness_timeseries)
        
class InteractivePlot(QMainWindow):
    """
    A main window class for the Droplet Freezing Assay Offline Analysis application.

    This class sets up the main window with multiple tabs for different functionalities
    such as tube locating, freezing detection, and image cropping. It handles user
    interactions, updates the log, and manages the display of various plots and images.

    Attributes:
        update_log_signal (pyqtSignal): Signal emitted to update the log with a message and tab number.
    """
    update_log_signal = pyqtSignal(str, int)  # str for message, int for tab number
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

        # Create tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab_widget.addTab(self.tab1, "Tube Locating")
        self.tab_widget.addTab(self.tab2, "Freezing Detection")
        self.tab_widget.addTab(self.tab3, "Image Cropping")

        # Set up tab layouts
        self.setup_tube_locating_tab()
        self.setup_freezing_detection_tab()
        self.setup_image_cropping_tab()

        # 初始化更新定时器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.plot_tube_detection_results)

        # 存储 PCR tubes 和 inner circles
        self.pcr_tubes = []
        self.inner_circles = []
        self.img = None

        # 绘制初始图表
        self.crop_region = None
        self.crop_selector = None
        self.plot_tube_detection_results()
        
        self.update_log_signal.connect(self.update_log)

        # Initialize tubes size
        self.tubes_size = (10, 8)
        
    def update_log(self, message, tab_number):
        if tab_number == 1:
            self.log_text_edit.append(message)
        elif tab_number == 2:
            self.log_text_edit2.append(message)

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
        # self.sample_image_path = '1/data/images/cropped.png'  # a default
        self.sample_image_path = '1/data/images/2023-04-03_16-05-57.png'  # a default
        self.sample_image_path_label = QLabel(f"Current image: {os.path.basename(self.sample_image_path)}")
        right_layout.addWidget(self.sample_image_path_label)
        right_layout.addWidget(self.sample_image_path_button)
        
        # Add refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.plot_tube_detection_results)
        right_layout.addWidget(self.refresh_button)

        # Tubes Size Input
        self.tubes_size_input = QLineEdit()
        self.tubes_size_input.setPlaceholderText("Enter tubes size (width, height)")
        self.tubes_size_input.setText('16, 10')
        self.tubes_size_input.textChanged.connect(self.update_tubes_size)
        right_layout.addWidget(QLabel("Tubes Array Size:"))
        right_layout.addWidget(self.tubes_size_input)

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
        self.min_area_slider.setMaximum(1500)
        self.min_area_slider.setSingleStep(10)
        self.min_area_slider.setValue(800)
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
        
        # 拉伸一下
        right_layout.setSpacing(10)

        # 将部件添加到主布局
        tab1_layout.addWidget(left_widget, 2)
        tab1_layout.addWidget(right_widget, 1)
        
        # 重定向输出
        sys.stdout = StreamToTextEdit(self.update_log_signal, 1)

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
        self.image_directory = '1/data/images/'
        self.tube_location_file = '1/inner_circles_20241007_134313.pkl'
        self.temperature_recording_file = '1/data/temperature/CR1000_Sec_1.dat'
    
    def setup_image_cropping_tab(self):
        tab3_layout = QHBoxLayout(self.tab3)

        # Create matplotlib figure and canvas
        self.figure_crop = Figure(figsize=(5, 4), dpi=100)
        self.canvas_crop = FigureCanvas(self.figure_crop)
        self.ax_crop = self.figure_crop.add_subplot(111)

        # Create control panel
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Add load image button
        self.load_crop_image_button = QPushButton("Load Image")
        self.load_crop_image_button.clicked.connect(self.load_crop_image)
        control_layout.addWidget(self.load_crop_image_button)

        # Add apply crop button
        self.apply_crop_button = QPushButton("Apply Crop")
        self.apply_crop_button.clicked.connect(self.apply_crop)
        control_layout.addWidget(self.apply_crop_button)

        # Add widgets to main layout
        tab3_layout.addWidget(self.canvas_crop, 2)
        tab3_layout.addWidget(control_widget, 1)
    
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

            # Load the image
            self.img = cv2.imread(image_path)

            # Apply cropping if a region is set
            if self.crop_region:
                x, y, w, h = self.crop_region
                self.img = self.img[y:y+h, x:x+w]
                self.log_text_edit.append(f"Using cropped region: {self.crop_region}")

            self.pcr_tubes, _ = locate_pcr_tubes(self.img, min_area, circularity_threshold)
            self.inferred_tubes = infer_missing_tubes(self.pcr_tubes, self.img.shape, tubes_size=self.tubes_size, rotate=rotation)
            self.all_tubes = self.pcr_tubes + self.inferred_tubes
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

    def update_tubes_size(self, text):
        try:
            width, height = map(int, text.split(','))
            self.tubes_size = (width, height)
            self.schedule_update()
            print(f"Tubes Size updated to {self.tubes_size}")
        except ValueError:
            self.log_text_edit.append("Invalid input for tubes size. Please enter two integers separated by a comma.")

    def save_inner_circles(self):
        default_filename = f"inner_circles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        default_filepath = os.path.join(".", default_filename)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Inner Circles", default_filepath, "Pickle Files (*.pkl)")
        if file_path:
            # Restore locations if a crop region was used
            if self.crop_region:
                x, y, _, _ = self.crop_region
                restored_circles = []
                for circle in self.inner_circles:
                    restored_circle = circle.copy()
                    restored_circle['x'] += x
                    restored_circle['y'] += y
                    restored_circles.append(restored_circle)
            else:
                restored_circles = self.inner_circles

            with open(file_path, 'wb') as f:
                pickle.dump(restored_circles, f)
            self.log_text_edit.append(f"Inner circles saved to {file_path}")

    def save_freezing_events_data(self):
        default_filename = f"freezing_temperatures_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        default_filepath = os.path.join(".", default_filename)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Freezing Temperatures", default_filepath, "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write("Tube,Temperature,Timestamp\n")  # Header
                for tube, data in self.freezing_temperatures.items():
                    temperature = data['temperature']
                    timestamp = data['timestamp']
                    if temperature is not None and timestamp is not None:
                        datetime_str = pd.Timestamp(timestamp).isoformat()  # numpy datetime, str looks like: 2023-04-03T16:30:55.000000000
                        f.write(f"{tube},{temperature:.4f},{datetime_str}\n")
                    else:
                        f.write(f"{tube},N/A,N/A\n")
            
            self.log_text_edit2.append(f"Freezing Temperatures saved to {file_path}")
            
    def load_freezing_events_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Freezing Temperatures", ".", "Text Files (*.txt)")
        if file_path:
            try:
                self.freezing_temperatures = {}
                with open(file_path, 'r') as f:
                    next(f)  # Skip the header line
                    for line in f:
                        try:
                            tube, temperature, datetime_str = line.strip().split(',')
                            tube = int(tube)
                            if temperature != 'N/A':
                                temperature = float(temperature)
                                timestamp = pd.to_datetime(datetime_str).to_numpy()
                                
                                self.freezing_temperatures[tube] = {
                                    'temperature': temperature,
                                    'timestamp': timestamp
                                }
                            else:
                                self.freezing_temperatures[tube] = {
                                    'temperature': None,
                                    'timestamp': None
                                }
                        except ValueError as e:
                            self.log_text_edit2.append(f"Error parsing line: {line}. Error: {str(e)}")
                            continue
                
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
        # Load inner circles first
        self.load_inner_circles()

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
        self.num_tubes = len(self.inner_circles)  # Use the loaded inner circles
        self.current_tube = 0
        self.log_text_edit2.append("Data loaded successfully!")
        
        # Count valid freezing points
        valid_freezing_points = sum(1 for data in self.freezing_temperatures.values() 
                                    if data['temperature'] is not None and data['timestamp'] is not None)
        self.log_text_edit2.append(f"Detected freezing points for {valid_freezing_points} out of {self.num_tubes} tubes")
        
        self.update_brightness_timeseries_plot()
        self.enable_controls()

    def update_subprocess_log(self, message):
        self.update_log_signal.emit(message, 2)
        
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
            if self.current_tube in self.freezing_temperatures:
                freezing_data = self.freezing_temperatures[self.current_tube]
                freezing_temp = freezing_data['temperature']
                freezing_timestamp = freezing_data['timestamp']
                if freezing_temp is not None and freezing_timestamp is not None:
                    freezing_temp_index = np.argmin(np.abs(self.current_tube_timestamps - freezing_timestamp))
                    freezing_brightness = self.current_tube_brightness[freezing_temp_index]
                else:
                    self.log_text_edit2.append(f"No freezing point detected for tube {self.current_tube}")
                    return
            else:
                self.log_text_edit2.append(f"No freezing data available for tube {self.current_tube}")
                return
        else:
            # Recalculate freezing point within the selected range
            mask = (self.current_tube_temperature >= xmin) & (self.current_tube_temperature <= xmax)
            if np.sum(mask) >= 3:
                temp_range = self.current_tube_temperature[mask].to_numpy()
                bright_range = self.current_tube_brightness[mask]  # already a numpy array
                time_range = self.current_tube_timestamps[mask]  # already a numpy array
                
                # Calculate the derivative (rate of change) of brightness
                brightness_derivative = np.diff(bright_range)
                
                # Find the index of the largest decrease
                freezing_index = np.argmin(brightness_derivative)
                freezing_temp = temp_range[freezing_index]
                freezing_brightness = bright_range[freezing_index]
                freezing_timestamp = time_range[freezing_index]

                # Update the freezing temperature for this tube
                self.freezing_temperatures[self.current_tube] = {
                    'temperature': freezing_temp,
                    'timestamp': freezing_timestamp
                }
            else:
                self.log_text_edit2.append("Selected range is too small. Please select a larger range.")
                return

        # Remove old freezing point if it exists
        if hasattr(self, 'freezing_point'):
            self.freezing_point.remove()
        
        # Plot new freezing point
        self.freezing_point, = self.ax2.plot(freezing_temp, freezing_brightness, 'ro', markersize=10, 
                                            label=f"Freezing Point: {freezing_temp:.2f}°C")
        self.ax2.legend()
        self.canvas2.draw()
        self.log_text_edit2.append(f"Updated freezing point for tube {self.current_tube}: {freezing_temp:.2f}°C at timestamp {freezing_timestamp}")

    def on_brightness_span_select(self, xmin, xmax):
        self.update_freezing_point(xmin, xmax)

    def load_crop_image(self):
        self.ax_crop.clear()
        img = cv2.imread(self.sample_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax_crop.imshow(img_rgb)
        self.ax_crop.axis('off')
        
        if self.crop_selector is not None:
            self.crop_selector.set_active(False)
        
        self.crop_selector = RectangleSelector(
            self.ax_crop, self.on_crop_select,
            useblit=True,
            button=[1, 3],  # Left and right mouse buttons
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        self.canvas_crop.draw()

    def on_crop_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.log_text_edit.append(f"Crop region set to: {self.crop_region}")

    def apply_crop(self):
        if self.crop_region:
            self.plot_tube_detection_results()
            self.log_text_edit.append("Crop applied to tube detection")
        else:
            self.log_text_edit.append("No crop region selected")
            
    def load_inner_circles(self):
        try:
            with open(self.tube_location_file, 'rb') as f:
                self.inner_circles = pickle.load(f)
            self.log_text_edit2.append(f"Loaded {len(self.inner_circles)} inner circles from {self.tube_location_file}")
        except Exception as e:
            self.log_text_edit2.append(f"Error loading inner circles: {str(e)}")
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = InteractivePlot()
    main_window.show()
    sys.exit(app.exec_())
