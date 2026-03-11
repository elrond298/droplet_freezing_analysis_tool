from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QPoint, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from gui import InteractivePlot


class FullMessageNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas: FigureCanvas, parent: QWidget) -> None:
        super().__init__(canvas, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if hasattr(self, 'locLabel'):
            self.locLabel.hide()

    def set_message(self, message: str) -> None:
        full_message = message or ""
        if hasattr(self, 'locLabel'):
            self.locLabel.clear()

        window = self.window()
        if window is not None and hasattr(window, 'statusBar'):
            if getattr(window, 'show_hover_coordinates_in_status_bar', True):
                window.statusBar().showMessage(full_message)
            else:
                window.statusBar().clearMessage()


class BoundedPopupComboBox(QComboBox):
    popup_margin = 12

    def showPopup(self) -> None:
        super().showPopup()
        self._bound_popup_to_screen()

    def _bound_popup_to_screen(self) -> None:
        popup_view = self.view()
        popup_window = popup_view.window() if popup_view is not None else None
        if popup_view is None or popup_window is None:
            return

        screen = self.screen()
        popup_handle = popup_window.windowHandle()
        if popup_handle is not None and popup_handle.screen() is not None:
            screen = popup_handle.screen()
        if screen is None:
            return

        screen_geometry = screen.availableGeometry()
        margin = self.popup_margin
        combo_top_left = self.mapToGlobal(QPoint(0, 0))
        combo_bottom_left = self.mapToGlobal(QPoint(0, self.height()))
        available_below = screen_geometry.bottom() - combo_bottom_left.y() - margin + 1
        available_above = combo_top_left.y() - screen_geometry.top() - margin

        row_height = popup_view.sizeHintForRow(0)
        if row_height <= 0:
            row_height = popup_view.fontMetrics().height() + 10

        visible_rows = self.maxVisibleItems() if self.maxVisibleItems() > 0 else self.count()
        visible_rows = max(1, min(max(self.count(), 1), max(visible_rows, 1)))
        desired_height = max(
            popup_window.sizeHint().height(),
            row_height * visible_rows + 8,
        )
        target_height = min(desired_height, max(available_below, available_above, row_height))

        desired_width = max(self.width(), popup_window.sizeHint().width())
        max_width = max(160, screen_geometry.width() - (margin * 2))
        target_width = min(desired_width, max_width)

        open_below = available_below >= target_height or available_below >= available_above
        if open_below:
            target_y = min(
                combo_bottom_left.y(),
                screen_geometry.bottom() - margin - target_height + 1,
            )
        else:
            target_y = max(
                screen_geometry.top() + margin,
                combo_top_left.y() - target_height,
            )

        max_x = screen_geometry.right() - margin - target_width + 1
        target_x = min(max(combo_top_left.x(), screen_geometry.left() + margin), max_x)

        popup_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        popup_view.setMinimumWidth(target_width)
        popup_view.setMaximumHeight(target_height)
        popup_window.resize(target_width, target_height)
        popup_window.move(target_x, target_y)


def create_log_group(window: InteractivePlot, title: str, attribute_name: str) -> QGroupBox:
    log_group = QGroupBox(title)
    log_layout = QVBoxLayout(log_group)
    log_text_edit = QTextEdit()
    log_text_edit.setReadOnly(True)
    log_text_edit.setMinimumHeight(240)
    setattr(window, attribute_name, log_text_edit)
    log_layout.addWidget(log_text_edit)
    return log_group

def build_tube_locating_tab(window: InteractivePlot) -> None:
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
    window.set_standard_button_icon(window.refresh_button, QStyle.StandardPixmap.SP_BrowserReload)
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
    window.set_standard_button_icon(window.save_button, QStyle.StandardPixmap.SP_DialogSaveButton)
    window.save_button.clicked.connect(window.save_detected_inner_circles)
    review_layout.addWidget(window.save_button)
    right_layout.addWidget(review_group)

    log_group = create_log_group(window, "Detection Log", 'log_text_edit')
    right_layout.addWidget(log_group, 1)

    right_layout.addStretch(1)

    right_scroll_area = window.create_scrollable_panel(right_widget)

    tab1_layout.addWidget(left_widget, 5)
    tab1_layout.addWidget(right_scroll_area, 3)


def build_freezing_detection_tab(window: InteractivePlot) -> None:
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
    input_layout = QGridLayout(input_group)
    input_layout.setHorizontalSpacing(12)
    input_layout.setVerticalSpacing(10)
    input_layout.setColumnStretch(0, 1)
    input_layout.setColumnStretch(1, 1)

    image_dir_group, window.image_directory_label = window.create_selection_group(
        "Image Directory",
        "Choose Image Folder For Analysis",
        window.select_image_directory,
        QStyle.StandardPixmap.SP_DirOpenIcon,
    )
    input_layout.addWidget(image_dir_group, 0, 0)

    temp_rec_group, window.temperature_recording_label = window.create_selection_group(
        "Temperature Recording",
        "Choose Temperature Recording File",
        window.select_temperature_recording,
        QStyle.StandardPixmap.SP_DialogOpenButton,
    )
    input_layout.addWidget(temp_rec_group, 0, 1)

    tube_loc_group, window.tube_locations_label = window.create_selection_group(
        "Tube Locations",
        "Choose Saved Tube-Location File",
        window.select_tube_locations,
        QStyle.StandardPixmap.SP_DialogOpenButton,
    )
    input_layout.addWidget(tube_loc_group, 1, 0)

    cutoff_group = QGroupBox("Temperature Cutoff Timestamp")
    cutoff_layout = QVBoxLayout(cutoff_group)
    cutoff_layout.setSpacing(8)

    window.analysis_temperature_cutoff_input = QLineEdit(window.analysis_temperature_cutoff_timestamp)
    window.analysis_temperature_cutoff_input.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
    window.analysis_temperature_cutoff_input.editingFinished.connect(window.update_analysis_temperature_cutoff_timestamp)
    cutoff_layout.addWidget(window.analysis_temperature_cutoff_input)

    cutoff_hint_label = QLabel("Format: YYYY-MM-DD HH:MM:SS")
    cutoff_hint_label.setWordWrap(True)
    cutoff_layout.addWidget(cutoff_hint_label)

    input_layout.addWidget(cutoff_group, 1, 1)

    window.start_load_timeseries_button = QPushButton("Load Brightness Timeseries")
    window.set_standard_button_icon(window.start_load_timeseries_button, QStyle.StandardPixmap.SP_MediaPlay)
    window.start_load_timeseries_button.clicked.connect(window.start_brightness_series_analysis)
    input_layout.addWidget(window.start_load_timeseries_button, 2, 0, 1, 2)

    window.analysis_progress_bar = QProgressBar()
    window.analysis_progress_bar.setObjectName("analysisProgressBar")
    window.analysis_progress_bar.setRange(0, 100)
    window.analysis_progress_bar.setValue(0)
    window.analysis_progress_bar.setProperty("completed", False)
    window.analysis_progress_bar.setFormat("Waiting for the image folder, temperature file, tube locations to start ...")
    input_layout.addWidget(window.analysis_progress_bar, 3, 0, 1, 2)

    right_layout.addWidget(input_group)

    review_group = QGroupBox("Tube Review")
    review_layout = QVBoxLayout(review_group)

    selector_form = QFormLayout()
    window.tube_sort_combo = BoundedPopupComboBox()
    window.tube_sort_combo.addItem("Tube index", "index")
    window.tube_sort_combo.addItem("Freezing temperature", "temperature")
    window.tube_sort_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    window.tube_sort_combo.setMaxVisibleItems(12)
    window.tube_sort_combo.currentIndexChanged.connect(window.handle_analysis_tube_sort_changed)
    selector_form.addRow("Sort tube list by:", window.tube_sort_combo)

    window.tube_selector_combo = BoundedPopupComboBox()
    window.tube_selector_combo.setPlaceholderText("Choose a tube to review")
    window.tube_selector_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    window.tube_selector_combo.setMaxVisibleItems(12)
    window.tube_selector_combo.currentIndexChanged.connect(window.go_to_tube)
    selector_form.addRow("Select tube:", window.tube_selector_combo)
    review_layout.addLayout(selector_form)

    button_layout = QHBoxLayout()
    window.prev_button = QPushButton("Show Previous Tube")
    window.next_button = QPushButton("Show Next Tube")
    window.discard_button = QPushButton("Mark Current Tube As Not Available")
    window.set_standard_button_icon(window.prev_button, QStyle.StandardPixmap.SP_ArrowBack)
    window.set_standard_button_icon(window.next_button, QStyle.StandardPixmap.SP_ArrowForward)
    window.set_standard_button_icon(window.discard_button, QStyle.StandardPixmap.SP_TrashIcon)
    window.prev_button.clicked.connect(window.previous_tube)
    window.next_button.clicked.connect(window.next_tube)
    window.discard_button.clicked.connect(window.discard_current_tube_freezing_point)
    button_layout.addWidget(window.prev_button)
    button_layout.addWidget(window.next_button)
    button_layout.addWidget(window.discard_button)
    review_layout.addLayout(button_layout)
    right_layout.addWidget(review_group)

    export_group = QGroupBox("Import / Export")
    export_layout = QVBoxLayout(export_group)
    window.send_to_inp_button = QPushButton("Add Current Results To INP Plot")
    window.set_standard_button_icon(window.send_to_inp_button, QStyle.StandardPixmap.SP_DialogApplyButton)
    window.send_to_inp_button.clicked.connect(window.prompt_and_add_current_analysis_to_inp)
    window.save_button_freezing_temperatures = QPushButton("Export Reviewed Freezing Temperatures")
    window.set_standard_button_icon(window.save_button_freezing_temperatures, QStyle.StandardPixmap.SP_DialogSaveButton)
    window.save_button_freezing_temperatures.clicked.connect(window.save_freezing_events_data)
    window.load_button_freezing_temperatures = QPushButton("Import Saved Freezing Temperatures")
    window.set_standard_button_icon(window.load_button_freezing_temperatures, QStyle.StandardPixmap.SP_DialogOpenButton)
    window.load_button_freezing_temperatures.clicked.connect(window.load_freezing_events_data)
    export_layout.addWidget(window.send_to_inp_button)
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
    window.tube_sort_combo.setEnabled(False)
    window.tube_selector_combo.setEnabled(False)
    window.send_to_inp_button.setEnabled(False)
    window.save_button_freezing_temperatures.setEnabled(False)
    window.load_button_freezing_temperatures.setEnabled(False)


def build_inp_tab(window: InteractivePlot) -> None:
    tab4_layout = QHBoxLayout(window.tab4)
    tab4_layout.setContentsMargins(12, 12, 12, 12)
    tab4_layout.setSpacing(12)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(4)
    window.figure_inp = Figure(figsize=(5, 4), dpi=100)
    window.configure_figure_padding(window.figure_inp)
    window.canvas_inp = FigureCanvas(window.figure_inp)
    window.toolbar_inp = FullMessageNavigationToolbar(window.canvas_inp, window)
    left_layout.addWidget(window.toolbar_inp)
    left_layout.addWidget(window.canvas_inp, 1)

    window.ax_inp = window.figure_inp.add_subplot(111)
    window.show_inp_plot_instructions()

    right_widget = QWidget()
    right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    right_layout = QVBoxLayout(right_widget)
    right_layout.setSpacing(12)

    right_layout.addWidget(window.create_tab_header(
        "Compare cumulative INP concentrations",
        "Load reviewed freezing temperatures, apply the droplet volume and dilution settings, then compare one or more INP curves on the same plot."
    ))

    sources_group = QGroupBox("Add Freezing Temperatures")
    sources_layout = QVBoxLayout(sources_group)

    window.add_inp_file_button = QPushButton("Add Freezing Temperatures File")
    window.set_standard_button_icon(window.add_inp_file_button, QStyle.StandardPixmap.SP_DialogOpenButton)
    window.add_inp_file_button.clicked.connect(window.prompt_and_add_inp_dataset_from_files)
    sources_layout.addWidget(window.add_inp_file_button)

    preset_row = QHBoxLayout()
    window.inp_preset_combo = QComboBox()
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
    window.inp_preset_combo.addItem(
        "Water control example",
        os.path.join(examples_dir, 'water_freezing_temperatures.txt'),
    )
    window.inp_preset_combo.addItem(
        "Yeast example",
        os.path.join(examples_dir, 'yeast_freezing_temperatures.txt'),
    )
    preset_row.addWidget(window.inp_preset_combo, 1)

    window.add_inp_preset_button = QPushButton("Add Preset")
    window.set_standard_button_icon(window.add_inp_preset_button, QStyle.StandardPixmap.SP_DialogApplyButton)
    window.add_inp_preset_button.clicked.connect(window.prompt_and_add_selected_inp_preset)
    preset_row.addWidget(window.add_inp_preset_button)
    sources_layout.addLayout(preset_row)

    window.add_current_analysis_to_inp_button = QPushButton("Add Current Results From Analyze Freezing")
    window.set_standard_button_icon(window.add_current_analysis_to_inp_button, QStyle.StandardPixmap.SP_DialogApplyButton)
    window.add_current_analysis_to_inp_button.clicked.connect(window.prompt_and_add_current_analysis_to_inp_from_tab4)
    sources_layout.addWidget(window.add_current_analysis_to_inp_button)
    right_layout.addWidget(sources_group)

    datasets_group = QGroupBox("Loaded INP Datasets")
    datasets_layout = QVBoxLayout(datasets_group)

    window.inp_dataset_list = QListWidget()
    window.inp_dataset_list.setMinimumHeight(180)
    datasets_layout.addWidget(window.inp_dataset_list)

    dataset_buttons = QHBoxLayout()
    window.remove_inp_dataset_button = QPushButton("Remove Selected Dataset")
    window.set_standard_button_icon(window.remove_inp_dataset_button, QStyle.StandardPixmap.SP_TrashIcon)
    window.remove_inp_dataset_button.clicked.connect(window.remove_selected_inp_dataset)
    dataset_buttons.addWidget(window.remove_inp_dataset_button)

    window.clear_inp_datasets_button = QPushButton("Clear All Datasets")
    window.set_standard_button_icon(window.clear_inp_datasets_button, QStyle.StandardPixmap.SP_DialogResetButton)
    window.clear_inp_datasets_button.clicked.connect(window.clear_inp_datasets)
    dataset_buttons.addWidget(window.clear_inp_datasets_button)
    datasets_layout.addLayout(dataset_buttons)
    right_layout.addWidget(datasets_group)

    log_group = create_log_group(window, "INP Log", 'log_text_edit_inp')
    right_layout.addWidget(log_group, 1)

    right_layout.addStretch(1)

    right_scroll_area = window.create_scrollable_panel(right_widget)

    tab4_layout.addWidget(left_widget, 5)
    tab4_layout.addWidget(right_scroll_area, 3)

    window.refresh_inp_plot()


def build_image_cropping_tab(window: InteractivePlot) -> None:
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
    window.set_standard_button_icon(window.sample_image_path_button, QStyle.StandardPixmap.SP_DialogOpenButton)
    window.sample_image_path_button.clicked.connect(window.select_sample_image_path)
    image_layout.addWidget(window.sample_image_path_button)

    window.load_crop_image_button = QPushButton("Load Selected Image Into Preview")
    window.set_standard_button_icon(window.load_crop_image_button, QStyle.StandardPixmap.SP_MediaPlay)
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
    window.set_standard_button_icon(window.apply_rotation_button, QStyle.StandardPixmap.SP_DialogApplyButton)
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
    window.set_standard_button_icon(window.apply_crop_button, QStyle.StandardPixmap.SP_ArrowForward)
    window.apply_crop_button.clicked.connect(window.apply_selected_crop_to_tube_detection)
    crop_layout.addWidget(window.apply_crop_button)

    window.restore_image_button = QPushButton("Restore Original Image Preview")
    window.set_standard_button_icon(window.restore_image_button, QStyle.StandardPixmap.SP_DialogResetButton)
    window.restore_image_button.clicked.connect(window.restore_original_preparation_image)
    crop_layout.addWidget(window.restore_image_button)
    control_layout.addWidget(crop_group)

    log_group = create_log_group(window, "Preparation Log", 'log_text_edit_prep')
    control_layout.addWidget(log_group, 1)

    control_layout.addStretch(1)

    control_scroll_area = window.create_scrollable_panel(control_widget)

    tab3_layout.addWidget(left_widget, 5)
    tab3_layout.addWidget(control_scroll_area, 3)


def build_settings_tab(window: InteractivePlot) -> None:
    tab5_layout = QHBoxLayout(window.tab5)
    tab5_layout.setContentsMargins(12, 12, 12, 12)
    tab5_layout.setSpacing(12)

    settings_widget = QWidget()
    settings_layout = QVBoxLayout(settings_widget)
    settings_layout.setSpacing(12)

    settings_layout.addWidget(window.create_tab_header(
        "Adjust defaults, display, and shortcuts",
        "Set reusable detection defaults, choose how selections are remembered, and control display behavior without crowding the workflow tabs."
    ))

    settings_layout.addWidget(window.create_detection_defaults_group())
    settings_layout.addWidget(window.create_inp_defaults_group())
    settings_layout.addWidget(window.create_session_behavior_group())
    settings_layout.addWidget(window.create_plot_behavior_group())
    settings_layout.addWidget(window.create_display_controls())
    settings_layout.addWidget(window.create_shortcuts_summary_group())
    settings_layout.addStretch(1)

    settings_scroll_area = window.create_scrollable_panel(settings_widget)
    tab5_layout.addWidget(settings_scroll_area)