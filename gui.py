import sys
import os
import numpy as np
import pandas as pd
import pickle
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QLineEdit, QSlider, QLabel, QSpinBox, 
                             QFileDialog, QTextEdit, QTabWidget, QFrame, QGroupBox, QFormLayout,
                             QSizePolicy, QScrollArea)
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
        self.configure_window_for_screen()
        self.apply_styles()

        self.sample_image_path = '1/data/images/2023-04-03_16-05-57.png'

        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(True)
        self.tab_widget.setElideMode(Qt.ElideNone)
        self.tab_widget.tabBar().setExpanding(False)
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab_widget.addTab(self.tab3, "1. Prepare Image")
        self.tab_widget.addTab(self.tab1, "2. Locate Tubes")
        self.tab_widget.addTab(self.tab2, "3. Analyze Freezing")

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

    def configure_window_for_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.setGeometry(100, 100, 1400, 800)
            self.is_compact_screen = False
            return

        available_geometry = screen.availableGeometry()
        available_width = available_geometry.width()
        available_height = available_geometry.height()

        self.is_compact_screen = available_width <= 1920 or available_height <= 1080

        target_width = max(1200, int(available_width * 0.9))
        target_height = max(780, int(available_height * 0.88))
        target_width = min(target_width, available_width)
        target_height = min(target_height, available_height)

        x_pos = available_geometry.x() + max(0, (available_width - target_width) // 2)
        y_pos = available_geometry.y() + max(0, (available_height - target_height) // 2)

        self.setGeometry(x_pos, y_pos, target_width, target_height)
        self.setMinimumSize(1100, 720)

    def create_scrollable_panel(self, content_widget):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(content_widget)
        return scroll_area
        
    def update_log(self, message, tab_number):
        if tab_number == 1:
            self.log_text_edit.append(message)
        elif tab_number == 2:
            self.log_text_edit2.append(message)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #f4f7fb;
            }
            QTabWidget::pane {
                border: 1px solid #d6dde8;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #e8eef6;
                border: 1px solid #d6dde8;
                min-width: 180px;
                padding: 10px 18px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #17324d;
            }
            QGroupBox {
                border: 1px solid #d6dde8;
                border-radius: 8px;
                margin-top: 12px;
                padding: 16px 12px 12px 12px;
                font-weight: 600;
                color: #17324d;
                background: #fbfdff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #1f6fb2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                min-height: 22px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #195d95;
            }
            QPushButton:disabled {
                background: #b8c5d3;
                color: #eef3f8;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #c9d5e2;
                border-radius: 6px;
                padding: 6px;
                background: #ffffff;
            }
            QLabel#tabHeaderTitle {
                font-size: 18px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#tabHeaderDescription {
                color: #516579;
                line-height: 1.4;
            }
            QLabel#statusLabel {
                color: #2c4f6f;
                background: #eef5fb;
                border: 1px solid #d6e4f2;
                border-radius: 6px;
                padding: 8px 10px;
            }
            QLabel#hintLabel {
                color: #516579;
            }
        """)

    def create_tab_header(self, title, description):
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("tabHeaderTitle")

        description_label = QLabel(description)
        description_label.setObjectName("tabHeaderDescription")
        description_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return header

    def create_status_label(self, text):
        label = QLabel(text)
        label.setObjectName("statusLabel")
        label.setWordWrap(True)
        return label

    def refresh_image_path_labels(self):
        current_image = os.path.basename(self.sample_image_path)
        if hasattr(self, 'sample_image_path_label'):
            self.sample_image_path_label.setText(f"Current image: {current_image}")
        if hasattr(self, 'tube_image_summary_label'):
            self.tube_image_summary_label.setText(f"Tube detection source: {current_image}")

    def create_selection_group(self, title, button_text, selection_method):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        button = QPushButton(button_text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.clicked.connect(selection_method)
        
        label = QLabel("Current: Not selected")
        label.setWordWrap(True)
        
        layout.addWidget(button)
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group, label

    def setup_tube_locating_tab(self):
        tab1_layout = QHBoxLayout(self.tab1)
        tab1_layout.setContentsMargins(12, 12, 12, 12)
        tab1_layout.setSpacing(12)

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
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)

        right_layout.addWidget(self.create_tab_header(
            "Locate tubes on the prepared image",
            "Review the cropped image, tune the detection settings, then save the inner-circle positions for the analysis step."
        ))

        self.tube_image_summary_label = self.create_status_label(
            f"Tube detection source: {os.path.basename(self.sample_image_path)}"
        )
        right_layout.addWidget(self.tube_image_summary_label)

        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setSpacing(10)

        self.refresh_button = QPushButton("Refresh Detection")
        self.refresh_button.clicked.connect(self.plot_tube_detection_results)
        detection_layout.addWidget(self.refresh_button)

        form_layout = QFormLayout()
        self.tubes_size_input = QLineEdit()
        self.tubes_size_input.setPlaceholderText("Example: 10, 8 (rows, columns)")
        self.tubes_size_input.setToolTip("Enter the tube grid as rows, columns. Example: 10, 8 means 10 rows and 8 columns.")
        self.tubes_size_input.setText('10, 8')
        self.tubes_size_input.textChanged.connect(self.update_tubes_size)
        form_layout.addRow("Tubes array size (rows, columns):", self.tubes_size_input)

        tubes_size_hint = QLabel("Example: 10, 8 means 10 rows and 8 columns.")
        tubes_size_hint.setObjectName("hintLabel")
        tubes_size_hint.setWordWrap(True)
        form_layout.addRow("", tubes_size_hint)

        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Enter rotation value")
        self.rotation_input.setText('auto')
        self.rotation_input.textChanged.connect(self.schedule_update)
        form_layout.addRow("Grid rotation:", self.rotation_input)
        detection_layout.addLayout(form_layout)

        min_area_group = QWidget()
        min_area_layout = QVBoxLayout(min_area_group)
        min_area_layout.setContentsMargins(0, 0, 0, 0)
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setMinimum(10)
        self.min_area_slider.setMaximum(1500)
        self.min_area_slider.setSingleStep(10)
        self.min_area_slider.setValue(800)
        self.min_area_label = QLabel("Min Area: 800")
        self.min_area_slider.valueChanged.connect(self.update_min_area)
        min_area_layout.addWidget(self.min_area_label)
        min_area_layout.addWidget(self.min_area_slider)
        detection_layout.addWidget(min_area_group)

        circularity_group = QWidget()
        circularity_layout = QVBoxLayout(circularity_group)
        circularity_layout.setContentsMargins(0, 0, 0, 0)
        self.circularity_slider = QSlider(Qt.Horizontal)
        self.circularity_slider.setMinimum(10)
        self.circularity_slider.setMaximum(100)
        self.circularity_slider.setSingleStep(5)
        self.circularity_slider.setValue(20)
        self.circularity_label = QLabel("Circularity: 0.20")
        self.circularity_slider.valueChanged.connect(self.update_circularity)
        circularity_layout.addWidget(self.circularity_label)
        circularity_layout.addWidget(self.circularity_slider)
        detection_layout.addWidget(circularity_group)
        right_layout.addWidget(detection_group)

        review_group = QGroupBox("Manual Review")
        review_layout = QVBoxLayout(review_group)
        review_hint = QLabel("Left click removes an inner circle. Right click adds a new one.")
        review_hint.setObjectName("hintLabel")
        review_hint.setWordWrap(True)
        review_layout.addWidget(review_hint)

        self.save_button = QPushButton("Save Inner Circles")
        self.save_button.clicked.connect(self.save_inner_circles)
        review_layout.addWidget(self.save_button)
        right_layout.addWidget(review_group)

        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        right_layout.addWidget(log_group, 1)
        
        right_layout.addStretch(1)

        right_scroll_area = self.create_scrollable_panel(right_widget)

        # 将部件添加到主布局
        tab1_layout.addWidget(left_widget, 5)
        tab1_layout.addWidget(right_scroll_area, 3)
        
        # 重定向输出
        sys.stdout = StreamToTextEdit(self.update_log_signal, 1)

    def setup_freezing_detection_tab(self):
        tab2_layout = QHBoxLayout(self.tab2)
        tab2_layout.setContentsMargins(12, 12, 12, 12)
        tab2_layout.setSpacing(12)

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
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)

        right_layout.addWidget(self.create_tab_header(
            "Review freezing events tube by tube",
            "Choose the required input files, run the timeseries analysis, then inspect or correct the freezing point for each tube."
        ))

        input_group = QGroupBox("Analysis Inputs")
        input_layout = QVBoxLayout(input_group)

        image_dir_group, self.image_directory_label = self.create_selection_group(
            "Image Directory", "Select Images Folder", self.select_image_directory)
        input_layout.addWidget(image_dir_group)

        temp_rec_group, self.temperature_recording_label = self.create_selection_group(
            "Temperature Recording", "Select Temperature File", self.select_temperature_recording)
        input_layout.addWidget(temp_rec_group)

        tube_loc_group, self.tube_locations_label = self.create_selection_group(
            "Tube Locations", "Select Tube Locations", self.select_tube_locations)
        input_layout.addWidget(tube_loc_group)
        
        self.start_load_timeseries_button = QPushButton("Run Analysis")
        self.start_load_timeseries_button.clicked.connect(self.load_brightness_series)
        input_layout.addWidget(self.start_load_timeseries_button)
        right_layout.addWidget(input_group)

        review_group = QGroupBox("Tube Review")
        review_layout = QVBoxLayout(review_group)

        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.discard_button = QPushButton("Discard")
        self.prev_button.clicked.connect(self.previous_tube)
        self.next_button.clicked.connect(self.next_tube)
        self.discard_button.clicked.connect(self.discard_tube)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.discard_button)
        review_layout.addLayout(button_layout)

        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("Enter tube number")
        self.value_input.returnPressed.connect(self.go_to_tube)
        review_layout.addWidget(QLabel("Go to tube:"))
        review_layout.addWidget(self.value_input)
        right_layout.addWidget(review_group)

        export_group = QGroupBox("Import / Export")
        export_layout = QVBoxLayout(export_group)
        self.save_button_freezing_temperatures = QPushButton("Save Freezing Temperatures")
        self.save_button_freezing_temperatures.clicked.connect(self.save_freezing_events_data)
        self.load_button_freezing_temperatures = QPushButton("Load Freezing Temperatures")
        self.load_button_freezing_temperatures.clicked.connect(self.load_freezing_events_data)
        export_layout.addWidget(self.save_button_freezing_temperatures)
        export_layout.addWidget(self.load_button_freezing_temperatures)
        right_layout.addWidget(export_group)

        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit2 = QTextEdit()
        self.log_text_edit2.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit2)
        right_layout.addWidget(log_group, 1)

        right_layout.addStretch(1)

        right_scroll_area = self.create_scrollable_panel(right_widget)

        # 将部件添加到主布局
        tab2_layout.addWidget(left_widget, 5)
        tab2_layout.addWidget(right_scroll_area, 3)
        
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
        tab3_layout.setContentsMargins(12, 12, 12, 12)
        tab3_layout.setSpacing(12)

        # Create matplotlib figure and canvas
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.figure_crop = Figure(figsize=(5, 4), dpi=100)
        self.canvas_crop = FigureCanvas(self.figure_crop)
        self.toolbar_crop = NavigationToolbar(self.canvas_crop, self)
        self.ax_crop = self.figure_crop.add_subplot(111)
        left_layout.addWidget(self.toolbar_crop)
        left_layout.addWidget(self.canvas_crop)

        # Create control panel
        control_widget = QWidget()
        control_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(12)

        control_layout.addWidget(self.create_tab_header(
            "Prepare the image before tube detection",
            "Select a source image, adjust its angle if needed, then crop the useful region before moving to tube detection."
        ))

        image_group = QGroupBox("Step 1: Choose Image")
        image_layout = QVBoxLayout(image_group)

        self.sample_image_path_label = QLabel(f"Current image: {os.path.basename(self.sample_image_path)}")
        self.sample_image_path_label.setWordWrap(True)
        image_layout.addWidget(self.sample_image_path_label)

        self.sample_image_path_button = QPushButton("Select an Image")
        self.sample_image_path_button.clicked.connect(self.select_sample_image_path)
        image_layout.addWidget(self.sample_image_path_button)

        self.load_crop_image_button = QPushButton("Load Image")
        self.load_crop_image_button.clicked.connect(self.load_crop_image)
        image_layout.addWidget(self.load_crop_image_button)
        control_layout.addWidget(image_group)

        rotation_group = QGroupBox("Step 2: Adjust Rotation")
        rotation_group_layout = QVBoxLayout(rotation_group)
        rotation_layout = QHBoxLayout()
        self.rotation_input_crop = QLineEdit()
        self.rotation_input_crop.setPlaceholderText("Example: 2.5 or -2.5 degrees")
        self.rotation_input_crop.setToolTip("Positive values rotate counter-clockwise. Negative values rotate clockwise.")
        self.rotation_input_crop.setText('0')
        rotation_layout.addWidget(QLabel("Rotation angle (degrees):"))
        rotation_layout.addWidget(self.rotation_input_crop)
        rotation_group_layout.addLayout(rotation_layout)

        rotation_hint = QLabel("Use positive values for counter-clockwise rotation and negative values for clockwise rotation. Example: 2.5 or -2.5.")
        rotation_hint.setObjectName("hintLabel")
        rotation_hint.setWordWrap(True)
        rotation_group_layout.addWidget(rotation_hint)

        self.apply_rotation_button = QPushButton("Rotate Image")
        self.apply_rotation_button.clicked.connect(self.apply_rotation)
        rotation_group_layout.addWidget(self.apply_rotation_button)
        control_layout.addWidget(rotation_group)

        crop_group = QGroupBox("Step 3: Crop And Continue")
        crop_layout = QVBoxLayout(crop_group)
        crop_hint = QLabel("Drag a rectangle on the image. Apply Crop will switch to the tube detection tab.")
        crop_hint.setObjectName("hintLabel")
        crop_hint.setWordWrap(True)
        crop_layout.addWidget(crop_hint)

        self.apply_crop_button = QPushButton("Apply Crop")
        self.apply_crop_button.clicked.connect(self.apply_crop)
        crop_layout.addWidget(self.apply_crop_button)

        self.restore_image_button = QPushButton("Restore Original Image")
        self.restore_image_button.clicked.connect(self.restore_original_image)
        crop_layout.addWidget(self.restore_image_button)
        control_layout.addWidget(crop_group)

        # Store original image
        self.original_image = None
        self.rotated_image = None

        control_layout.addStretch(1)

        control_scroll_area = self.create_scrollable_panel(control_widget)

        # Add widgets to main layout
        tab3_layout.addWidget(left_widget, 5)
        tab3_layout.addWidget(control_scroll_area, 3)

    
    def select_sample_image_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select an Image")
        if file:
            self.sample_image_path = file
            self.refresh_image_path_labels()
            self.log_text_edit.append(f"Selected image: {file}")
            self.load_crop_image()

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
            rotation = self.rotation_input.text()

            # Use the processed image if available, otherwise use the original sample image
            if hasattr(self, 'processed_image'):
                self.img = self.processed_image
            else:
                if not os.path.exists(self.sample_image_path):
                    raise FileNotFoundError(f"Image file not found: {self.sample_image_path}")
                self.img = cv2.imread(self.sample_image_path)

            # Tube detection
            self.pcr_tubes, _ = locate_pcr_tubes(self.img, min_area, circularity_threshold)

            # Prepare for further processing
            self.inferred_tubes = infer_missing_tubes(
                self.pcr_tubes,
                self.original_image.shape,
                tubes_size=self.tubes_size,
                rotate=rotation
            )

            self.all_tubes = self.pcr_tubes + self.inferred_tubes
            self.inner_circles = detect_inner_circles(self.img, self.all_tubes)

            # Visualize on the canvas
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

        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # Left click - remove point
            # Find closest inner circle
            if self.inner_circles:
                closest_circle = min(self.inner_circles,
                                    key=lambda c: ((c['x'] - x)**2 + (c['y'] - y)**2)**0.5)
                circle_distance = ((closest_circle['x'] - x)**2 + (closest_circle['y'] - y)**2)**0.5

                # Remove if close enough
                if circle_distance < 20:  # Adjust threshold as needed
                    self.inner_circles.remove(closest_circle)
                    print(f"Removed inner circle at ({closest_circle['x']}, {closest_circle['y']})")
                    self.redraw_tube_detection_results()

        elif event.button == 3:  # Right click - add point
            # Add new inner circle
            new_circle = {'x': x, 'y': y, 'radius': 10}
            self.inner_circles.append(new_circle)
            print(f"Added new inner circle at ({x}, {y})")
            self.redraw_tube_detection_results()

    def redraw_tube_detection_results(self):
        self.ax.clear()
        img_with_tubes = self.img.copy()

        for tube in self.all_tubes:
            color = (0, 255, 0) if 'inferred' not in tube else (0, 0, 255)
            cv2.circle(img_with_tubes, (tube['x'], tube['y']), tube['radius'], color, 2)

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
            rows, columns = map(int, text.split(','))
            self.tubes_size = (rows, columns)
            self.schedule_update()
            print(f"Tubes Size updated to {self.tubes_size}")
        except ValueError:
            self.log_text_edit.append("Invalid input for tubes size. Please enter rows and columns as two integers separated by a comma.")

    def save_inner_circles(self):
        default_filename = f"inner_circles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        default_filepath = os.path.join(".", default_filename)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Inner Circles", default_filepath, "Pickle Files (*.pkl)")
        if file_path:
            # Restore locations relative to original image
            restored_circles = []
            for circle in self.inner_circles:
                restored_circle = circle.copy()

                # Step 1: Restore to rotated image coordinates
                # If cropping was applied, add back the crop offsets
                if hasattr(self, 'crop_region'):
                    x_offset, y_offset, _, _ = self.crop_region
                    restored_circle['x'] += x_offset
                    restored_circle['y'] += y_offset

                # Step 2: Rotate back to original image coordinates
                if hasattr(self, 'rotation_params'):
                    center = self.rotation_params['center']
                    rotation_angle = self.rotation_params['angle']  # opencv: positive rotation means rotate counter-clockwise

                    # Convert point relative to center
                    x_rel = restored_circle['x'] - center[0]
                    y_rel = restored_circle['y'] - center[1]

                    # Rotate back (negative angle)
                    cos_theta = np.cos(np.deg2rad(rotation_angle))  # just use the selected value, to do a reverse rotation  
                    sin_theta = np.sin(np.deg2rad(rotation_angle))

                    # Apply inverse rotation
                    x_restored = x_rel * cos_theta - y_rel * sin_theta
                    y_restored = x_rel * sin_theta + y_rel * cos_theta

                    # Restore absolute coordinates
                    restored_circle['x'] = int(x_restored + center[0])
                    restored_circle['y'] = int(y_restored + center[1])

                restored_circles.append(restored_circle)

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

    def discard_tube(self):
        """
        Discard current tube, set the temperature to 0 C
        """
        bright_range = self.current_tube_brightness  # already a numpy array
        time_range = self.current_tube_timestamps  # already a numpy array

        # Find the index of the largest decrease
        freezing_index = 0
        freezing_temp = 0
        freezing_brightness = bright_range[freezing_index]
        freezing_timestamp = time_range[freezing_index]

        # Update the freezing temperature for this tube
        self.freezing_temperatures[self.current_tube] = {
            'temperature': freezing_temp,
            'timestamp': freezing_timestamp
        }

        # Remove old freezing point if it exists
        if hasattr(self, 'freezing_point'):
            # self.freezing_point.remove()
            # Note @2025-04-06 : NotImplementedError from matplotlib artist.py. Try another way to delete it.
            self.freezing_point.set_data([], [])  # 清空数据
            self.freezing_point.set_label("")

        # Plot new freezing point
        self.freezing_point, = self.ax2.plot(freezing_temp, freezing_brightness, 'ro', markersize=10,
                                            label=f"Freezing Point: {freezing_temp:.2f}°C")
        self.ax2.legend()
        self.canvas2.draw()
        self.log_text_edit2.append(f"Updated freezing point for tube {self.current_tube}: {freezing_temp:.2f}°C at timestamp {freezing_timestamp}")

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
            # self.freezing_point.remove()
            # Note @2025-04-06 : NotImplementedError from matplotlib artist.py. Try another way to delete it.
            self.freezing_point.set_data([], [])  # 清空数据
            self.freezing_point.set_label("")
        
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
        self.original_image = cv2.imread(self.sample_image_path)
        self.rotated_image = self.original_image.copy()
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
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

    def apply_rotation(self):
        try:
            # Check if an image is loaded
            if self.original_image is None:
                self.log_text_edit.append("No image loaded. Please load an image first.")
                return

            rotation_angle = float(self.rotation_input_crop.text())

            # Get image center
            height, width = self.original_image.shape[:2]
            center = (width / 2, height / 2)

            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)  # A positive rotation_angle rotates the image counter-clockwise

            # Perform rotation
            self.rotated_image = cv2.warpAffine(
                self.original_image,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR
            )

            # Display rotated image
            self.ax_crop.clear()
            img_rgb = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2RGB)
            self.ax_crop.imshow(img_rgb)
            self.ax_crop.axis('off')

            # Reinitialize crop selector
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

            # Store rotation parameters for later use
            self.rotation_params = {
                'angle': rotation_angle,
                'center': center,
                'matrix': rotation_matrix
            }

        except ValueError:
            self.log_text_edit.append("Invalid rotation angle. Please enter a number.")
        except Exception as e:
            self.log_text_edit.append(f"Error during rotation: {str(e)}")

    def restore_original_image(self):
        if self.original_image is not None:
            self.rotated_image = self.original_image.copy()
            self.ax_crop.clear()
            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.ax_crop.imshow(img_rgb)
            self.ax_crop.axis('off')

            # Reinitialize crop selector
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
            self.crop_region = None

    def on_crop_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.log_text_edit.append(f"Crop region set to: {self.crop_region}")

    def apply_crop(self):
        try:
            # Check if an image is loaded and crop region is selected
            if self.rotated_image is None:
                self.log_text_edit.append("No image loaded. Please load an image first.")
                return

            if self.crop_region is None:
                self.log_text_edit.append("No crop region selected. Please select a region.")
                return

            # Crop the rotated image
            x, y, w, h = self.crop_region
            cropped_img = self.rotated_image[y:y+h, x:x+w]

            # Prepare for tube detection by modifying the original workflow
            self.processed_image = cropped_img

            # Update UI to show what was done
            self.log_text_edit.append(f"Crop region: {self.crop_region}")
            self.log_text_edit.append(f"Cropped image size: {cropped_img.shape[:2]}")

            # If rotation was applied, store the rotation parameters
            if hasattr(self, 'rotation_params'):
                rotation_angle = self.rotation_params['angle']
                self.log_text_edit.append(f"Rotation angle: {rotation_angle} degrees")

            # Switch to Tube Locating tab
            self.tab_widget.setCurrentWidget(self.tab1)

            # Run tube detection with the processed image
            self.plot_tube_detection_results()

            self.log_text_edit.append("Crop and rotation applied to tube detection")

        except Exception as e:
            self.log_text_edit.append(f"Error during crop: {str(e)}")

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
    if getattr(main_window, 'is_compact_screen', False):
        main_window.showMaximized()
    else:
        main_window.show()
    sys.exit(app.exec_())
