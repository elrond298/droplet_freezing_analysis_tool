from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class FullMessageNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if hasattr(self, 'locLabel'):
            self.locLabel.hide()

    def set_message(self, message):
        full_message = message or ""
        if hasattr(self, 'locLabel'):
            self.locLabel.clear()

        window = self.window()
        if window is not None and hasattr(window, 'statusBar'):
            window.statusBar().showMessage(full_message)


def create_log_group(window, title, attribute_name):
    log_group = QGroupBox(title)
    log_layout = QVBoxLayout(log_group)
    log_text_edit = QTextEdit()
    log_text_edit.setReadOnly(True)
    log_text_edit.setMinimumHeight(240)
    setattr(window, attribute_name, log_text_edit)
    log_layout.addWidget(log_text_edit)
    return log_group

def build_tube_locating_tab(window):
    tab1_layout = QHBoxLayout(window.tab1)
    tab1_layout.setContentsMargins(12, 12, 12, 12)
    tab1_layout.setSpacing(12)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(4)
    window.figure = Figure(figsize=(5, 4), dpi=100)
    window.configure_figure_padding(window.figure, image_mode=True, reserve_title_space=True)
    window.canvas = FigureCanvas(window.figure)
    window.toolbar = FullMessageNavigationToolbar(window.canvas, window)
    left_layout.addWidget(window.toolbar)
    left_layout.addWidget(window.canvas, 1)

    window.ax = window.figure.add_subplot(111)
    window.canvas.mpl_connect('button_press_event', window.handle_tube_detection_plot_click)

    right_widget = QWidget()
    right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    right_layout = QVBoxLayout(right_widget)
    right_layout.setSpacing(12)

    right_layout.addWidget(window.create_tab_header(
        "Locate tubes on the prepared image",
        "Review the cropped image, tune the detection settings, then save the inner-circle positions for the analysis step."
    ))

    window.tube_image_summary_label = window.create_status_label(
        window.format_selected_path("Tube detection source", window.sample_image_path)
    )
    right_layout.addWidget(window.tube_image_summary_label)

    detection_group = QGroupBox("Detection Settings")
    detection_layout = QVBoxLayout(detection_group)
    detection_layout.setSpacing(10)

    window.refresh_button = QPushButton("Run Tube Detection")
    window.refresh_button.clicked.connect(window.run_tube_detection_and_render_plot)
    detection_layout.addWidget(window.refresh_button)

    form_layout = QFormLayout()
    window.tubes_size_input = QLineEdit()
    window.tubes_size_input.setPlaceholderText("Example: 10, 8 (rows, columns)")
    window.tubes_size_input.setToolTip("Enter the tube grid as rows, columns. Example: 10, 8 means 10 rows and 8 columns.")
    window.tubes_size_input.setText('10, 8')
    window.tubes_size_input.textChanged.connect(window.update_tubes_size)
    form_layout.addRow("Tubes array size (rows, columns):", window.tubes_size_input)

    tubes_size_hint = QLabel("Example: 10, 8 means 10 rows and 8 columns.")
    tubes_size_hint.setObjectName("hintLabel")
    tubes_size_hint.setWordWrap(True)
    form_layout.addRow("", tubes_size_hint)

    window.rotation_input = QLineEdit()
    window.rotation_input.setPlaceholderText("auto or degrees, e.g. -1.5")
    window.rotation_input.setToolTip("Use 'auto' to estimate the tube-grid angle, or enter degrees manually. Positive values rotate counter-clockwise and negative values rotate clockwise.")
    window.rotation_input.setText('auto')
    window.rotation_input.textChanged.connect(window.schedule_update)
    form_layout.addRow("Grid rotation:", window.rotation_input)

    rotation_hint = QLabel("Use 'auto' to estimate the tube-grid angle. You can also enter degrees manually: positive values rotate counter-clockwise, negative values rotate clockwise.")
    rotation_hint.setObjectName("hintLabel")
    rotation_hint.setWordWrap(True)
    form_layout.addRow("", rotation_hint)

    detection_layout.addLayout(form_layout)

    min_area_group = QWidget()
    min_area_layout = QVBoxLayout(min_area_group)
    min_area_layout.setContentsMargins(0, 0, 0, 0)
    window.min_area_slider = QSlider(Qt.Orientation.Horizontal)
    window.min_area_slider.setMinimum(10)
    window.min_area_slider.setMaximum(1500)
    window.min_area_slider.setSingleStep(10)
    window.min_area_slider.setValue(800)
    window.min_area_label = QLabel("Min Area: 800")
    window.min_area_slider.valueChanged.connect(window.update_min_area)
    min_area_layout.addWidget(window.min_area_label)
    min_area_layout.addWidget(window.min_area_slider)
    detection_layout.addWidget(min_area_group)

    circularity_group = QWidget()
    circularity_layout = QVBoxLayout(circularity_group)
    circularity_layout.setContentsMargins(0, 0, 0, 0)
    window.circularity_slider = QSlider(Qt.Orientation.Horizontal)
    window.circularity_slider.setMinimum(10)
    window.circularity_slider.setMaximum(100)
    window.circularity_slider.setSingleStep(5)
    window.circularity_slider.setValue(20)
    window.circularity_label = QLabel("Circularity: 0.20")
    window.circularity_slider.valueChanged.connect(window.update_circularity)
    circularity_layout.addWidget(window.circularity_label)
    circularity_layout.addWidget(window.circularity_slider)
    detection_layout.addWidget(circularity_group)
    right_layout.addWidget(detection_group)

    review_group = QGroupBox("Manual Review")
    review_layout = QVBoxLayout(review_group)
    review_hint = QLabel("Left click removes an inner circle. Right click adds a new inner circle at the clicked location.")
    review_hint.setObjectName("hintLabel")
    review_hint.setWordWrap(True)
    review_layout.addWidget(review_hint)

    window.save_button = QPushButton("Save Inner-Circle Locations")
    window.save_button.clicked.connect(window.save_detected_inner_circles)
    review_layout.addWidget(window.save_button)
    right_layout.addWidget(review_group)

    log_group = create_log_group(window, "Detection Log", 'log_text_edit')
    right_layout.addWidget(log_group, 1)

    right_layout.addStretch(1)

    right_scroll_area = window.create_scrollable_panel(right_widget)

    tab1_layout.addWidget(left_widget, 5)
    tab1_layout.addWidget(right_scroll_area, 3)


def build_freezing_detection_tab(window):
    tab2_layout = QHBoxLayout(window.tab2)
    tab2_layout.setContentsMargins(12, 12, 12, 12)
    tab2_layout.setSpacing(12)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(4)
    window.figure2 = Figure(figsize=(5, 4), dpi=100)
    window.configure_figure_padding(window.figure2)
    window.canvas2 = FigureCanvas(window.figure2)
    window.toolbar2 = FullMessageNavigationToolbar(window.canvas2, window)
    left_layout.addWidget(window.toolbar2)
    left_layout.addWidget(window.canvas2, 1)

    window.ax2 = window.figure2.add_subplot(111)
    window.show_analysis_plot_instructions()

    right_widget = QWidget()
    right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    right_layout = QVBoxLayout(right_widget)
    right_layout.setSpacing(12)

    right_layout.addWidget(window.create_tab_header(
        "Review freezing events tube by tube",
        "Choose the required input files, run the timeseries analysis, then inspect or correct the freezing point for each tube."
    ))

    input_group = QGroupBox("Analysis Inputs")
    input_layout = QVBoxLayout(input_group)

    image_dir_group, window.image_directory_label = window.create_selection_group(
        "Image Directory", "Choose Image Folder For Analysis", window.select_image_directory)
    input_layout.addWidget(image_dir_group)

    temp_rec_group, window.temperature_recording_label = window.create_selection_group(
        "Temperature Recording", "Choose Temperature Recording File", window.select_temperature_recording)
    input_layout.addWidget(temp_rec_group)

    tube_loc_group, window.tube_locations_label = window.create_selection_group(
        "Tube Locations", "Choose Saved Tube-Location File", window.select_tube_locations)
    input_layout.addWidget(tube_loc_group)

    window.start_load_timeseries_button = QPushButton("Load Brightness Timeseries")
    window.start_load_timeseries_button.clicked.connect(window.start_brightness_series_analysis)
    input_layout.addWidget(window.start_load_timeseries_button)

    window.analysis_progress_bar = QProgressBar()
    window.analysis_progress_bar.setRange(0, 100)
    window.analysis_progress_bar.setValue(0)
    window.analysis_progress_bar.setFormat("Waiting for the image folder, temperature file, tube locations to start ...")
    input_layout.addWidget(window.analysis_progress_bar)

    right_layout.addWidget(input_group)

    review_group = QGroupBox("Tube Review")
    review_layout = QVBoxLayout(review_group)

    button_layout = QHBoxLayout()
    window.prev_button = QPushButton("Show Previous Tube")
    window.next_button = QPushButton("Show Next Tube")
    window.discard_button = QPushButton("Mark Current Tube As Not Available")
    window.prev_button.clicked.connect(window.previous_tube)
    window.next_button.clicked.connect(window.next_tube)
    window.discard_button.clicked.connect(window.discard_current_tube_freezing_point)
    button_layout.addWidget(window.prev_button)
    button_layout.addWidget(window.next_button)
    button_layout.addWidget(window.discard_button)
    review_layout.addLayout(button_layout)

    window.value_input = QLineEdit()
    window.value_input.setPlaceholderText("Enter a tube number, press Enter to review")
    window.value_input.returnPressed.connect(window.go_to_tube)
    review_layout.addWidget(QLabel("Jump to tube number:"))
    review_layout.addWidget(window.value_input)
    right_layout.addWidget(review_group)

    export_group = QGroupBox("Import / Export")
    export_layout = QVBoxLayout(export_group)
    window.save_button_freezing_temperatures = QPushButton("Export Reviewed Freezing Temperatures")
    window.save_button_freezing_temperatures.clicked.connect(window.save_freezing_events_data)
    window.load_button_freezing_temperatures = QPushButton("Import Saved Freezing Temperatures")
    window.load_button_freezing_temperatures.clicked.connect(window.load_freezing_events_data)
    export_layout.addWidget(window.save_button_freezing_temperatures)
    export_layout.addWidget(window.load_button_freezing_temperatures)
    right_layout.addWidget(export_group)

    log_group = create_log_group(window, "Analysis Log", 'log_text_edit2')
    right_layout.addWidget(log_group, 1)

    right_layout.addStretch(1)

    right_scroll_area = window.create_scrollable_panel(right_widget)

    tab2_layout.addWidget(left_widget, 5)
    tab2_layout.addWidget(right_scroll_area, 3)

    window.prev_button.setEnabled(False)
    window.next_button.setEnabled(False)
    window.discard_button.setEnabled(False)
    window.value_input.setEnabled(False)
    window.save_button_freezing_temperatures.setEnabled(False)
    window.load_button_freezing_temperatures.setEnabled(False)


def build_image_cropping_tab(window):
    tab3_layout = QHBoxLayout(window.tab3)
    tab3_layout.setContentsMargins(12, 12, 12, 12)
    tab3_layout.setSpacing(12)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(4)
    window.figure_crop = Figure(figsize=(5, 4), dpi=100)
    window.configure_figure_padding(window.figure_crop, image_mode=True)
    window.canvas_crop = FigureCanvas(window.figure_crop)
    window.toolbar_crop = FullMessageNavigationToolbar(window.canvas_crop, window)
    window.ax_crop = window.figure_crop.add_subplot(111)
    left_layout.addWidget(window.toolbar_crop)
    left_layout.addWidget(window.canvas_crop, 1)

    control_widget = QWidget()
    control_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    control_layout = QVBoxLayout(control_widget)
    control_layout.setSpacing(12)

    control_layout.addWidget(window.create_tab_header(
        "Prepare the image before tube detection",
        "Select a source image, adjust its angle if needed, then crop the useful region before moving to tube detection."
    ))

    image_group = QGroupBox("Step 1: Choose Image")
    image_layout = QVBoxLayout(image_group)

    window.sample_image_path_label = QLabel(
        window.format_highlighted_selected_path("Current image", window.sample_image_path)
    )
    window.sample_image_path_label.setWordWrap(True)
    image_layout.addWidget(window.sample_image_path_label)

    window.sample_image_path_button = QPushButton("Choose Source Image")
    window.sample_image_path_button.clicked.connect(window.select_sample_image_path)
    image_layout.addWidget(window.sample_image_path_button)

    window.load_crop_image_button = QPushButton("Load Selected Image Into Preview")
    window.load_crop_image_button.clicked.connect(window.load_selected_image_into_preparation_view)
    image_layout.addWidget(window.load_crop_image_button)
    control_layout.addWidget(image_group)

    rotation_group = QGroupBox("Step 2: Adjust Rotation")
    rotation_group_layout = QVBoxLayout(rotation_group)
    rotation_layout = QHBoxLayout()
    window.rotation_input_crop = QLineEdit()
    window.rotation_input_crop.setPlaceholderText("Example: 2.5 or -2.5 degrees")
    window.rotation_input_crop.setToolTip("Positive values rotate counter-clockwise. Negative values rotate clockwise.")
    window.rotation_input_crop.setText('0')
    rotation_layout.addWidget(QLabel("Rotation angle (degrees):"))
    rotation_layout.addWidget(window.rotation_input_crop)
    rotation_group_layout.addLayout(rotation_layout)

    rotation_hint = QLabel("Use positive values for counter-clockwise rotation and negative values for clockwise rotation. Example: 2.5 or -2.5.")
    rotation_hint.setObjectName("hintLabel")
    rotation_hint.setWordWrap(True)
    rotation_group_layout.addWidget(rotation_hint)

    window.apply_rotation_button = QPushButton("Apply Rotation To Preview")
    window.apply_rotation_button.clicked.connect(window.apply_preparation_image_rotation)
    rotation_group_layout.addWidget(window.apply_rotation_button)
    control_layout.addWidget(rotation_group)

    crop_group = QGroupBox("Step 3: Crop And Continue")
    crop_layout = QVBoxLayout(crop_group)
    crop_hint = QLabel("Drag a rectangle on the image to select the analysis region. Applying the crop will open the tube-detection tab.")
    crop_hint.setObjectName("hintLabel")
    crop_hint.setWordWrap(True)
    crop_layout.addWidget(crop_hint)

    window.apply_crop_button = QPushButton("Apply Crop And Open Tube Detection")
    window.apply_crop_button.clicked.connect(window.apply_selected_crop_to_tube_detection)
    crop_layout.addWidget(window.apply_crop_button)

    window.restore_image_button = QPushButton("Restore Original Image Preview")
    window.restore_image_button.clicked.connect(window.restore_original_preparation_image)
    crop_layout.addWidget(window.restore_image_button)
    control_layout.addWidget(crop_group)

    log_group = create_log_group(window, "Preparation Log", 'log_text_edit_prep')
    control_layout.addWidget(log_group, 1)

    control_layout.addStretch(1)

    control_scroll_area = window.create_scrollable_panel(control_widget)

    tab3_layout.addWidget(left_widget, 5)
    tab3_layout.addWidget(control_scroll_area, 3)


def build_settings_tab(window):
    tab4_layout = QHBoxLayout(window.tab4)
    tab4_layout.setContentsMargins(12, 12, 12, 12)
    tab4_layout.setSpacing(12)

    settings_widget = QWidget()
    settings_layout = QVBoxLayout(settings_widget)
    settings_layout.setSpacing(12)

    settings_layout.addWidget(window.create_tab_header(
        "Adjust display and shortcuts",
        "Keep the analysis tabs uncluttered while still making interface size and keyboard navigation easy to access."
    ))

    settings_layout.addWidget(window.create_display_controls())
    settings_layout.addWidget(window.create_shortcuts_summary_group())
    settings_layout.addStretch(1)

    settings_scroll_area = window.create_scrollable_panel(settings_widget)
    tab4_layout.addWidget(settings_scroll_area)