from __future__ import annotations

import sys
import os
import html
import datetime
from typing import TYPE_CHECKING, Any, Callable
from matplotlib import rcParams
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QSlider, QLabel, QSpinBox, QCheckBox,
                             QComboBox, QListWidget,
                             QFileDialog, QTextEdit, QTabWidget, QFrame, QGroupBox, QFormLayout,
                             QSizePolicy, QScrollArea, QProgressBar, QDoubleSpinBox, QDialog,
                             QStyle,
                             QDialogButtonBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from matplotlib.widgets import RectangleSelector

from gui_analysis_controller import (
    discard_current_tube_freezing_point as analysis_discard_current_tube_freezing_point,
    enable_analysis_review_controls as analysis_enable_analysis_review_controls,
    load_analysis_inner_circle_locations as analysis_load_analysis_inner_circle_locations,
    start_brightness_series_analysis as analysis_start_brightness_series_analysis,
    load_freezing_events_data as analysis_load_freezing_events_data,
    apply_analysis_results as analysis_apply_analysis_results,
    save_freezing_events_data as analysis_save_freezing_events_data,
    refresh_current_tube_brightness_plot as analysis_refresh_current_tube_brightness_plot,
    refresh_current_tube_freezing_marker as analysis_refresh_current_tube_freezing_marker,
)
from gui_inp_controller import (
    add_current_analysis_to_inp as inp_add_current_analysis_to_inp,
    add_inp_dataset_from_files as inp_add_inp_dataset_from_files,
    add_selected_inp_preset as inp_add_selected_inp_preset,
    clear_inp_datasets as inp_clear_inp_datasets,
    refresh_inp_plot as inp_refresh_inp_plot,
    remove_selected_inp_dataset as inp_remove_selected_inp_dataset,
)
from gui_detection_controller import (
    handle_tube_detection_plot_click as detection_handle_tube_detection_plot_click,
    redraw_manual_tube_detection_plot as detection_redraw_manual_tube_detection_plot,
    reset_tube_detection_view as detection_reset_tube_detection_view,
    run_tube_detection_and_render_plot as detection_run_tube_detection_and_render_plot,
    save_detected_inner_circles as detection_save_detected_inner_circles,
)
from gui_image_controller import (
    apply_preparation_image_rotation as image_apply_preparation_image_rotation,
    apply_selected_crop_to_tube_detection as image_apply_selected_crop_to_tube_detection,
    load_selected_image_into_preparation_view as image_load_selected_image_into_preparation_view,
    restore_original_preparation_image as image_restore_original_preparation_image,
)
from gui_logging import (
    append_log_message as logging_append_log_message,
    format_log_message as logging_format_log_message,
    get_log_widget as logging_get_log_widget,
    write_log_entry as logging_write_log_entry,
)
from gui_selection_cache import (
    load_selection_cache as cache_load_selection_cache,
    refresh_analysis_input_labels as cache_refresh_analysis_input_labels,
    refresh_image_path_labels as cache_refresh_image_path_labels,
    restore_cached_selections as cache_restore_cached_selections,
    save_selection_cache as cache_save_selection_cache,
)
from gui_services import (
    normalize_inner_circles,
    restore_circle_to_original_image,
)
from gui_state import AnalysisState, DetectionState, ImagePrepState, InpState, SelectionState
from gui_tabs import (
    build_freezing_detection_tab,
    build_image_cropping_tab,
    build_inp_tab,
    build_settings_tab,
    build_tube_locating_tab,
)
from gui_workers import StreamToTextEdit
import cv2

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    from gui_tabs import BoundedPopupComboBox, FullMessageNavigationToolbar
        
class InteractivePlot(QMainWindow):
    """
    A main window class for the Droplet Freezing Assay Offline Analysis application.

    This class sets up the main window with multiple tabs for different functionalities
    such as tube locating, freezing detection, and image cropping. It handles user
    interactions, updates the log, and manages the display of various plots and images.

    Attributes:
        update_log_signal (pyqtSignal): Signal emitted to update the log with a message and tab number.
    """
    update_log_signal = pyqtSignal(str, int, str)  # str for message, int for tab number, str for level
    LOG_TAB_PREPARE = 1
    LOG_TAB_LOCATE = 2
    LOG_TAB_ANALYZE = 3
    LOG_TAB_INP = 4
    LOG_TAB_ALL = 0
    LOG_LEVEL_DEBUG = "DEBUG"
    LOG_LEVEL_INFO = "INFO"
    LOG_LEVEL_SUCCESS = "SUCCESS"
    LOG_LEVEL_WARNING = "WARNING"
    LOG_LEVEL_ERROR = "ERROR"
    LOG_LEVEL_STYLES = {
        LOG_LEVEL_DEBUG: {"label": "DEBUG", "badge": "#7a8694", "text": "#516579"},
        LOG_LEVEL_INFO: {"label": "INFO", "badge": "#1f6fb2", "text": "#17324d"},
        LOG_LEVEL_SUCCESS: {"label": "SUCCESS", "badge": "#2e8b57", "text": "#1f5133"},
        LOG_LEVEL_WARNING: {"label": "WARNING", "badge": "#c17c00", "text": "#7a4f00"},
        LOG_LEVEL_ERROR: {"label": "ERROR", "badge": "#c0392b", "text": "#7b241c"},
    }
    SELECTION_CACHE_FILENAME = ".gui_selection_cache.json"
    DEFAULT_FONT_SIZE = 10
    MIN_FONT_SIZE = 8
    MAX_FONT_SIZE = 24

    # These UI attributes are attached during tab construction in gui_tabs.py.
    # Declaring them here gives the editor a stable definition site for navigation and typing.
    figure: Figure
    canvas: FigureCanvas
    toolbar: FullMessageNavigationToolbar
    ax: Axes
    tube_image_summary_label: QLabel
    refresh_button: QPushButton
    tubes_size_input: QLineEdit
    rotation_input: QLineEdit
    min_area_slider: QSlider
    min_area_label: QLabel
    circularity_slider: QSlider
    circularity_label: QLabel
    save_button: QPushButton
    log_text_edit: QTextEdit

    figure2: Figure
    canvas2: FigureCanvas
    toolbar2: FullMessageNavigationToolbar
    ax2: Axes
    image_directory_label: QLabel
    temperature_recording_label: QLabel
    tube_locations_label: QLabel
    analysis_temperature_cutoff_input: QLineEdit
    start_load_timeseries_button: QPushButton
    analysis_progress_bar: QProgressBar
    tube_sort_combo: BoundedPopupComboBox
    tube_selector_combo: BoundedPopupComboBox
    prev_button: QPushButton
    next_button: QPushButton
    discard_button: QPushButton
    send_to_inp_button: QPushButton
    save_button_freezing_temperatures: QPushButton
    load_button_freezing_temperatures: QPushButton
    log_text_edit2: QTextEdit

    figure_inp: Figure
    canvas_inp: FigureCanvas
    toolbar_inp: FullMessageNavigationToolbar
    ax_inp: Axes
    add_inp_file_button: QPushButton
    inp_preset_combo: QComboBox
    add_inp_preset_button: QPushButton
    add_current_analysis_to_inp_button: QPushButton
    inp_dataset_list: QListWidget
    remove_inp_dataset_button: QPushButton
    clear_inp_datasets_button: QPushButton
    log_text_edit_inp: QTextEdit

    figure_crop: Figure
    canvas_crop: FigureCanvas
    toolbar_crop: FullMessageNavigationToolbar
    ax_crop: Axes
    sample_image_path_label: QLabel
    sample_image_path_button: QPushButton
    load_crop_image_button: QPushButton
    rotation_input_crop: QLineEdit
    apply_rotation_button: QPushButton
    apply_crop_button: QPushButton
    restore_image_button: QPushButton
    log_text_edit_prep: QTextEdit

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Droplet Freezing Assay Offline Analysis')
        self.initialize_state()
        self.apply_matplotlib_font_defaults()
        self.configure_window_for_screen()
        self.apply_styles()
        self.selection_cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.SELECTION_CACHE_FILENAME,
        )

        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(True)
        self.tab_widget.setElideMode(Qt.TextElideMode.ElideNone)
        self.tab_widget.tabBar().setExpanding(False)
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tab_widget.addTab(self.tab3, "1. Prepare Image")
        self.tab_widget.addTab(self.tab1, "2. Locate Tubes")
        self.tab_widget.addTab(self.tab2, "3. Analyze Freezing")
        self.tab_widget.addTab(self.tab4, "4. INP Concentration")
        self.tab_widget.addTab(self.tab5, "5. Settings")

        # Set up tab layouts
        self.setup_tube_locating_tab()
        self.setup_freezing_detection_tab()
        self.setup_image_cropping_tab()
        self.setup_inp_tab()
        self.setup_settings_tab()
        self.configure_console_redirect()
        self.configure_shortcuts()

        # 初始化更新定时器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.run_tube_detection_and_render_plot)

        self.restore_cached_selections()

        # 绘制初始图表
        self.run_tube_detection_and_render_plot()
        
        self.update_log_signal.connect(self.update_log)

    def configure_console_redirect(self) -> None:
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = StreamToTextEdit(self.update_log_signal, self.LOG_TAB_ALL, self.LOG_LEVEL_INFO)
        sys.stderr = StreamToTextEdit(self.update_log_signal, self.LOG_TAB_ALL, self.LOG_LEVEL_ERROR)

    def initialize_state(self) -> None:
        default_font_size = self.get_default_font_size()
        self.selection_state = SelectionState(ui_font_size=default_font_size)
        self.image_prep_state = ImagePrepState()
        self.detection_state = DetectionState()
        self.analysis_state = AnalysisState()
        self.inp_state = InpState()

    @property
    def ui_font_size(self) -> int:
        return self.selection_state.ui_font_size

    @ui_font_size.setter
    def ui_font_size(self, value: int) -> None:
        self.selection_state.ui_font_size = value

    @property
    def sample_image_path(self) -> str | None:
        return self.selection_state.sample_image_path

    @sample_image_path.setter
    def sample_image_path(self, value: str | None) -> None:
        self.selection_state.sample_image_path = value

    @property
    def image_directory(self) -> str | None:
        return self.selection_state.image_directory

    @image_directory.setter
    def image_directory(self, value: str | None) -> None:
        self.selection_state.image_directory = value

    @property
    def temperature_recording_file(self) -> str | None:
        return self.selection_state.temperature_recording_file

    @temperature_recording_file.setter
    def temperature_recording_file(self, value: str | None) -> None:
        self.selection_state.temperature_recording_file = value

    @property
    def tube_location_file(self) -> str | None:
        return self.selection_state.tube_location_file

    @tube_location_file.setter
    def tube_location_file(self, value: str | None) -> None:
        self.selection_state.tube_location_file = value

    @property
    def analysis_temperature_cutoff_timestamp(self) -> str:
        return self.selection_state.analysis_temperature_cutoff_timestamp

    @analysis_temperature_cutoff_timestamp.setter
    def analysis_temperature_cutoff_timestamp(self, value: str) -> None:
        self.selection_state.analysis_temperature_cutoff_timestamp = value

    @property
    def detection_default_tubes_size(self) -> tuple[int, int]:
        return self.selection_state.detection_default_tubes_size

    @detection_default_tubes_size.setter
    def detection_default_tubes_size(self, value: tuple[int, int]) -> None:
        self.selection_state.detection_default_tubes_size = value

    @property
    def detection_default_rotation(self) -> str:
        return self.selection_state.detection_default_rotation

    @detection_default_rotation.setter
    def detection_default_rotation(self, value: str) -> None:
        self.selection_state.detection_default_rotation = value

    @property
    def detection_default_min_area(self) -> int:
        return self.selection_state.detection_default_min_area

    @detection_default_min_area.setter
    def detection_default_min_area(self, value: int) -> None:
        self.selection_state.detection_default_min_area = value

    @property
    def detection_default_circularity(self) -> int:
        return self.selection_state.detection_default_circularity

    @detection_default_circularity.setter
    def detection_default_circularity(self, value: int) -> None:
        self.selection_state.detection_default_circularity = value

    @property
    def restore_last_selected_inputs(self) -> bool:
        return self.selection_state.restore_last_selected_inputs

    @restore_last_selected_inputs.setter
    def restore_last_selected_inputs(self, value: bool) -> None:
        self.selection_state.restore_last_selected_inputs = value

    @property
    def auto_save_selected_inputs(self) -> bool:
        return self.selection_state.auto_save_selected_inputs

    @auto_save_selected_inputs.setter
    def auto_save_selected_inputs(self, value: bool) -> None:
        self.selection_state.auto_save_selected_inputs = value

    @property
    def auto_open_tube_detection_after_crop(self) -> bool:
        return self.selection_state.auto_open_tube_detection_after_crop

    @auto_open_tube_detection_after_crop.setter
    def auto_open_tube_detection_after_crop(self, value: bool) -> None:
        self.selection_state.auto_open_tube_detection_after_crop = value

    @property
    def advance_after_marking_tube_not_available(self) -> bool:
        return self.selection_state.advance_after_marking_tube_not_available

    @advance_after_marking_tube_not_available.setter
    def advance_after_marking_tube_not_available(self, value: bool) -> None:
        self.selection_state.advance_after_marking_tube_not_available = value

    @property
    def inp_default_droplet_volume_ul(self) -> float:
        return self.selection_state.inp_default_droplet_volume_ul

    @inp_default_droplet_volume_ul.setter
    def inp_default_droplet_volume_ul(self, value: float) -> None:
        self.selection_state.inp_default_droplet_volume_ul = float(value)

    @property
    def inp_default_dilution_factor(self) -> float:
        return self.selection_state.inp_default_dilution_factor

    @inp_default_dilution_factor.setter
    def inp_default_dilution_factor(self, value: float) -> None:
        self.selection_state.inp_default_dilution_factor = float(value)

    @property
    def show_hover_coordinates_in_status_bar(self) -> bool:
        return self.selection_state.show_hover_coordinates_in_status_bar

    @show_hover_coordinates_in_status_bar.setter
    def show_hover_coordinates_in_status_bar(self, value: bool) -> None:
        self.selection_state.show_hover_coordinates_in_status_bar = value

    @property
    def img(self) -> Any:
        return self.image_prep_state.img

    @img.setter
    def img(self, value: Any) -> None:
        self.image_prep_state.img = value

    @property
    def original_image(self) -> Any:
        return self.image_prep_state.original_image

    @original_image.setter
    def original_image(self, value: Any) -> None:
        self.image_prep_state.original_image = value

    @property
    def rotated_image(self) -> Any:
        return self.image_prep_state.rotated_image

    @rotated_image.setter
    def rotated_image(self, value: Any) -> None:
        self.image_prep_state.rotated_image = value

    @property
    def processed_image(self) -> Any:
        return self.image_prep_state.processed_image

    @processed_image.setter
    def processed_image(self, value: Any) -> None:
        self.image_prep_state.processed_image = value

    @property
    def crop_region(self) -> tuple[int, int, int, int] | None:
        return self.image_prep_state.crop_region

    @crop_region.setter
    def crop_region(self, value: tuple[int, int, int, int] | None) -> None:
        self.image_prep_state.crop_region = value

    @property
    def crop_selector(self) -> Any:
        return self.image_prep_state.crop_selector

    @crop_selector.setter
    def crop_selector(self, value: Any) -> None:
        self.image_prep_state.crop_selector = value

    @property
    def rotation_params(self) -> dict[str, Any] | None:
        return self.image_prep_state.rotation_params

    @rotation_params.setter
    def rotation_params(self, value: dict[str, Any] | None) -> None:
        self.image_prep_state.rotation_params = value

    @property
    def pcr_tubes(self) -> list[dict[str, Any]]:
        return self.detection_state.pcr_tubes

    @pcr_tubes.setter
    def pcr_tubes(self, value: list[dict[str, Any]]) -> None:
        self.detection_state.pcr_tubes = value

    @property
    def inferred_tubes(self) -> list[dict[str, Any]]:
        return self.detection_state.inferred_tubes

    @inferred_tubes.setter
    def inferred_tubes(self, value: list[dict[str, Any]]) -> None:
        self.detection_state.inferred_tubes = value

    @property
    def all_tubes(self) -> list[dict[str, Any]]:
        return self.detection_state.all_tubes

    @all_tubes.setter
    def all_tubes(self, value: list[dict[str, Any]]) -> None:
        self.detection_state.all_tubes = value

    @property
    def inner_circles(self) -> list[dict[str, Any]]:
        return self.detection_state.inner_circles

    @inner_circles.setter
    def inner_circles(self, value: list[dict[str, Any]]) -> None:
        self.detection_state.inner_circles = value

    @property
    def tubes_size(self) -> tuple[int, int]:
        return self.detection_state.tubes_size

    @tubes_size.setter
    def tubes_size(self, value: tuple[int, int]) -> None:
        self.detection_state.tubes_size = value

    @property
    def temperature_recordings(self) -> Any:
        return self.analysis_state.temperature_recordings

    @temperature_recordings.setter
    def temperature_recordings(self, value: Any) -> None:
        self.analysis_state.temperature_recordings = value

    @property
    def brightness_timeseries(self) -> Any:
        return self.analysis_state.brightness_timeseries

    @brightness_timeseries.setter
    def brightness_timeseries(self, value: Any) -> None:
        self.analysis_state.brightness_timeseries = value

    @property
    def freezing_temperatures(self) -> dict[int, dict[str, Any]]:
        return self.analysis_state.freezing_temperatures

    @freezing_temperatures.setter
    def freezing_temperatures(self, value: dict[int, dict[str, Any]]) -> None:
        self.analysis_state.freezing_temperatures = value

    @property
    def num_tubes(self) -> int:
        return self.analysis_state.num_tubes

    @num_tubes.setter
    def num_tubes(self, value: int) -> None:
        self.analysis_state.num_tubes = value

    @property
    def current_tube(self) -> int:
        return self.analysis_state.current_tube

    @current_tube.setter
    def current_tube(self, value: int) -> None:
        self.analysis_state.current_tube = value

    @property
    def sorted_tube_indices(self) -> list[int]:
        return self.analysis_state.sorted_tube_indices

    @sorted_tube_indices.setter
    def sorted_tube_indices(self, value: list[int]) -> None:
        self.analysis_state.sorted_tube_indices = value

    @property
    def sorted_tube_position(self) -> int:
        return self.analysis_state.sorted_tube_position

    @sorted_tube_position.setter
    def sorted_tube_position(self, value: int) -> None:
        self.analysis_state.sorted_tube_position = value

    @property
    def current_tube_temperature(self) -> Any:
        return self.analysis_state.current_tube_temperature

    @current_tube_temperature.setter
    def current_tube_temperature(self, value: Any) -> None:
        self.analysis_state.current_tube_temperature = value

    @property
    def current_tube_brightness(self) -> Any:
        return self.analysis_state.current_tube_brightness

    @current_tube_brightness.setter
    def current_tube_brightness(self, value: Any) -> None:
        self.analysis_state.current_tube_brightness = value

    @property
    def current_tube_timestamps(self) -> Any:
        return self.analysis_state.current_tube_timestamps

    @current_tube_timestamps.setter
    def current_tube_timestamps(self, value: Any) -> None:
        self.analysis_state.current_tube_timestamps = value

    @property
    def inp_datasets(self) -> list[dict[str, Any]]:
        return self.inp_state.datasets

    @inp_datasets.setter
    def inp_datasets(self, value: list[dict[str, Any]]) -> None:
        self.inp_state.datasets = value

    def get_default_font_size(self) -> int:
        app = QApplication.instance()
        if app is not None:
            point_size = app.font().pointSize()
            if point_size > 0:
                return point_size
        return self.DEFAULT_FONT_SIZE

    def format_tubes_size(self, tubes_size: tuple[int, int]) -> str:
        return f"{tubes_size[0]}, {tubes_size[1]}"

    def format_analysis_tube_option(self, tube_index: int) -> str:
        freezing_data = self.freezing_temperatures.get(tube_index)
        if not isinstance(freezing_data, dict):
            return f"Tube {tube_index} | No freezing data"

        freezing_temp = freezing_data.get('temperature')
        if isinstance(freezing_temp, (int, float)):
            return f"Tube {tube_index} | {float(freezing_temp):.2f} °C"

        return f"Tube {tube_index} | Not available"

    def get_sorted_tube_indices(self) -> list[int]:
        tube_indices = list(range(self.num_tubes))
        tube_indices.sort(key=self._analysis_tube_sort_key)
        return tube_indices

    def _analysis_tube_sort_key(self, tube_index: int) -> tuple[float, int]:
        sort_mode = 'index'
        if hasattr(self, 'tube_sort_combo'):
            current_data = self.tube_sort_combo.currentData()
            if isinstance(current_data, str):
                sort_mode = current_data

        if sort_mode == 'temperature':
            freezing_data = self.freezing_temperatures.get(tube_index, {})
            freezing_temp = freezing_data.get('temperature') if isinstance(freezing_data, dict) else None
            if isinstance(freezing_temp, (int, float)):
                return (float(freezing_temp), tube_index)
            return (float('inf'), tube_index)

        return (float(tube_index), tube_index)

    def refresh_analysis_tube_selector(self, preserve_navigation_position: bool = False) -> None:
        if not hasattr(self, 'tube_selector_combo'):
            return

        current_tube = self.current_tube if self.num_tubes > 0 else None
        previous_position = self.sorted_tube_position
        tube_indices = self.get_sorted_tube_indices()
        self.sorted_tube_indices = tube_indices

        if tube_indices:
            if preserve_navigation_position:
                self.sorted_tube_position = min(max(previous_position, 0), len(tube_indices) - 1)
            elif current_tube in tube_indices:
                self.sorted_tube_position = tube_indices.index(current_tube)
            else:
                self.sorted_tube_position = 0
        else:
            self.sorted_tube_position = 0

        self.tube_selector_combo.blockSignals(True)
        self.tube_selector_combo.clear()
        for tube_index in tube_indices:
            self.tube_selector_combo.addItem(self.format_analysis_tube_option(tube_index), tube_index)

        has_tubes = bool(tube_indices)
        self.tube_selector_combo.setEnabled(has_tubes and self.next_button.isEnabled())
        if hasattr(self, 'tube_sort_combo'):
            self.tube_sort_combo.setEnabled(has_tubes and self.next_button.isEnabled())

        combo_view = self.tube_selector_combo.view()
        if combo_view is not None:
            combo_view.setMinimumWidth(max(self.tube_selector_combo.sizeHint().width(), self.tube_selector_combo.minimumSizeHint().width()))

        if current_tube is not None:
            selected_index = self.tube_selector_combo.findData(current_tube)
            if selected_index >= 0:
                self.tube_selector_combo.setCurrentIndex(selected_index)

        self.tube_selector_combo.blockSignals(False)

    def handle_analysis_tube_sort_changed(self, *_args: object) -> None:
        self.refresh_analysis_tube_selector(preserve_navigation_position=False)

    def parse_tubes_size_text(self, text: str) -> tuple[int, int]:
        rows_text, columns_text = [part.strip() for part in text.split(',')]
        rows = int(rows_text)
        columns = int(columns_text)
        if rows <= 0 or columns <= 0:
            raise ValueError()
        return rows, columns

    def refresh_settings_controls(self) -> None:
        if hasattr(self, 'settings_tubes_size_input'):
            self.settings_tubes_size_input.blockSignals(True)
            self.settings_tubes_size_input.setText(self.format_tubes_size(self.detection_default_tubes_size))
            self.settings_tubes_size_input.blockSignals(False)

        if hasattr(self, 'settings_rotation_input'):
            self.settings_rotation_input.blockSignals(True)
            self.settings_rotation_input.setText(self.detection_default_rotation)
            self.settings_rotation_input.blockSignals(False)

        if hasattr(self, 'settings_min_area_spinbox'):
            self.settings_min_area_spinbox.blockSignals(True)
            self.settings_min_area_spinbox.setValue(self.detection_default_min_area)
            self.settings_min_area_spinbox.blockSignals(False)

        if hasattr(self, 'settings_circularity_spinbox'):
            self.settings_circularity_spinbox.blockSignals(True)
            self.settings_circularity_spinbox.setValue(self.detection_default_circularity)
            self.settings_circularity_spinbox.blockSignals(False)

        if hasattr(self, 'restore_last_selected_inputs_checkbox'):
            self.restore_last_selected_inputs_checkbox.blockSignals(True)
            self.restore_last_selected_inputs_checkbox.setChecked(self.restore_last_selected_inputs)
            self.restore_last_selected_inputs_checkbox.blockSignals(False)

        if hasattr(self, 'auto_save_selected_inputs_checkbox'):
            self.auto_save_selected_inputs_checkbox.blockSignals(True)
            self.auto_save_selected_inputs_checkbox.setChecked(self.auto_save_selected_inputs)
            self.auto_save_selected_inputs_checkbox.blockSignals(False)

        if hasattr(self, 'auto_open_tube_detection_after_crop_checkbox'):
            self.auto_open_tube_detection_after_crop_checkbox.blockSignals(True)
            self.auto_open_tube_detection_after_crop_checkbox.setChecked(self.auto_open_tube_detection_after_crop)
            self.auto_open_tube_detection_after_crop_checkbox.blockSignals(False)

        if hasattr(self, 'advance_after_marking_tube_not_available_checkbox'):
            self.advance_after_marking_tube_not_available_checkbox.blockSignals(True)
            self.advance_after_marking_tube_not_available_checkbox.setChecked(self.advance_after_marking_tube_not_available)
            self.advance_after_marking_tube_not_available_checkbox.blockSignals(False)

        if hasattr(self, 'show_hover_coordinates_checkbox'):
            self.show_hover_coordinates_checkbox.blockSignals(True)
            self.show_hover_coordinates_checkbox.setChecked(self.show_hover_coordinates_in_status_bar)
            self.show_hover_coordinates_checkbox.blockSignals(False)

        if hasattr(self, 'settings_inp_volume_spinbox'):
            self.settings_inp_volume_spinbox.blockSignals(True)
            self.settings_inp_volume_spinbox.setValue(self.inp_default_droplet_volume_ul)
            self.settings_inp_volume_spinbox.blockSignals(False)

        if hasattr(self, 'settings_inp_dilution_spinbox'):
            self.settings_inp_dilution_spinbox.blockSignals(True)
            self.settings_inp_dilution_spinbox.setValue(self.inp_default_dilution_factor)
            self.settings_inp_dilution_spinbox.blockSignals(False)

    def apply_detection_defaults_to_locate_controls(self, schedule: bool = False) -> None:
        self.tubes_size = self.detection_default_tubes_size

        if hasattr(self, 'tubes_size_input'):
            self.tubes_size_input.blockSignals(True)
            self.tubes_size_input.setText(self.format_tubes_size(self.detection_default_tubes_size))
            self.tubes_size_input.blockSignals(False)

        if hasattr(self, 'rotation_input'):
            self.rotation_input.blockSignals(True)
            self.rotation_input.setText(self.detection_default_rotation)
            self.rotation_input.blockSignals(False)

        if hasattr(self, 'min_area_slider'):
            self.min_area_slider.blockSignals(True)
            self.min_area_slider.setValue(self.detection_default_min_area)
            self.min_area_slider.blockSignals(False)
        if hasattr(self, 'min_area_label'):
            self.min_area_label.setText(f"Min Area: {self.detection_default_min_area}")

        if hasattr(self, 'circularity_slider'):
            self.circularity_slider.blockSignals(True)
            self.circularity_slider.setValue(self.detection_default_circularity)
            self.circularity_slider.blockSignals(False)
        if hasattr(self, 'circularity_label'):
            self.circularity_label.setText(f"Circularity: {self.detection_default_circularity / 100:.2f}")

        if schedule:
            self.schedule_update()

    def create_detection_defaults_group(self) -> QGroupBox:
        detection_group = QGroupBox("Detection Defaults")
        layout = QFormLayout(detection_group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        self.settings_tubes_size_input = QLineEdit(self.format_tubes_size(self.detection_default_tubes_size))
        self.settings_tubes_size_input.setPlaceholderText("Example: 10, 8")
        self.settings_tubes_size_input.editingFinished.connect(self.update_detection_default_tubes_size)
        layout.addRow("Default tube grid (rows, columns):", self.settings_tubes_size_input)

        self.settings_rotation_input = QLineEdit(self.detection_default_rotation)
        self.settings_rotation_input.setPlaceholderText("auto or degrees, e.g. -1.5")
        self.settings_rotation_input.editingFinished.connect(self.update_detection_default_rotation)
        layout.addRow("Default grid rotation:", self.settings_rotation_input)

        self.settings_min_area_spinbox = QSpinBox()
        self.settings_min_area_spinbox.setRange(10, 1500)
        self.settings_min_area_spinbox.setSingleStep(10)
        self.settings_min_area_spinbox.setValue(self.detection_default_min_area)
        self.settings_min_area_spinbox.valueChanged.connect(self.update_detection_default_min_area)
        layout.addRow("Default minimum area:", self.settings_min_area_spinbox)

        self.settings_circularity_spinbox = QSpinBox()
        self.settings_circularity_spinbox.setRange(10, 100)
        self.settings_circularity_spinbox.setSingleStep(5)
        self.settings_circularity_spinbox.setSuffix(" %")
        self.settings_circularity_spinbox.setValue(self.detection_default_circularity)
        self.settings_circularity_spinbox.valueChanged.connect(self.update_detection_default_circularity)
        layout.addRow("Default circularity threshold:", self.settings_circularity_spinbox)

        return detection_group

    def create_inp_defaults_group(self) -> QGroupBox:
        inp_group = QGroupBox("INP Defaults")
        layout = QFormLayout(inp_group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        self.settings_inp_volume_spinbox = QDoubleSpinBox()
        self.settings_inp_volume_spinbox.setRange(0.01, 1000000.0)
        self.settings_inp_volume_spinbox.setDecimals(3)
        self.settings_inp_volume_spinbox.setSingleStep(1.0)
        self.settings_inp_volume_spinbox.setSuffix(" uL")
        self.settings_inp_volume_spinbox.setValue(self.inp_default_droplet_volume_ul)
        self.settings_inp_volume_spinbox.valueChanged.connect(self.update_inp_default_droplet_volume_ul)
        layout.addRow("Default droplet volume:", self.settings_inp_volume_spinbox)

        self.settings_inp_dilution_spinbox = QDoubleSpinBox()
        self.settings_inp_dilution_spinbox.setRange(0.000001, 1000000.0)
        self.settings_inp_dilution_spinbox.setDecimals(6)
        self.settings_inp_dilution_spinbox.setSingleStep(0.1)
        self.settings_inp_dilution_spinbox.setValue(self.inp_default_dilution_factor)
        self.settings_inp_dilution_spinbox.valueChanged.connect(self.update_inp_default_dilution_factor)
        layout.addRow("Default dilution factor:", self.settings_inp_dilution_spinbox)

        return inp_group

    def create_session_behavior_group(self) -> QGroupBox:
        session_group = QGroupBox("Session And File Behavior")
        layout = QVBoxLayout(session_group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.restore_last_selected_inputs_checkbox = QCheckBox("Restore the last selected input files and folders on startup")
        self.restore_last_selected_inputs_checkbox.setChecked(self.restore_last_selected_inputs)
        self.restore_last_selected_inputs_checkbox.toggled.connect(self.update_restore_last_selected_inputs)
        layout.addWidget(self.restore_last_selected_inputs_checkbox)

        self.auto_save_selected_inputs_checkbox = QCheckBox("Auto-save selected input files and folders as you change them")
        self.auto_save_selected_inputs_checkbox.setChecked(self.auto_save_selected_inputs)
        self.auto_save_selected_inputs_checkbox.toggled.connect(self.update_auto_save_selected_inputs)
        layout.addWidget(self.auto_save_selected_inputs_checkbox)

        self.auto_open_tube_detection_after_crop_checkbox = QCheckBox("Open the Locate Tubes tab automatically after applying a crop")
        self.auto_open_tube_detection_after_crop_checkbox.setChecked(self.auto_open_tube_detection_after_crop)
        self.auto_open_tube_detection_after_crop_checkbox.toggled.connect(self.update_auto_open_tube_detection_after_crop)
        layout.addWidget(self.auto_open_tube_detection_after_crop_checkbox)

        self.advance_after_marking_tube_not_available_checkbox = QCheckBox(
            "After 'Mark Current Tube As Not Available' in Analyze Freezing, move to the next tube"
        )
        self.advance_after_marking_tube_not_available_checkbox.setChecked(self.advance_after_marking_tube_not_available)
        self.advance_after_marking_tube_not_available_checkbox.toggled.connect(self.update_advance_after_marking_tube_not_available)
        layout.addWidget(self.advance_after_marking_tube_not_available_checkbox)

        return session_group

    def create_plot_behavior_group(self) -> QGroupBox:
        plot_group = QGroupBox("Plot And Status Bar")
        layout = QVBoxLayout(plot_group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.show_hover_coordinates_checkbox = QCheckBox("Show matplotlib hover coordinates in the status bar")
        self.show_hover_coordinates_checkbox.setChecked(self.show_hover_coordinates_in_status_bar)
        self.show_hover_coordinates_checkbox.toggled.connect(self.update_show_hover_coordinates_in_status_bar)
        layout.addWidget(self.show_hover_coordinates_checkbox)

        hint_label = QLabel("Hover readouts from the embedded plots appear in the main window status bar instead of the matplotlib toolbar.")
        hint_label.setObjectName("hintLabel")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        return plot_group

    def update_detection_default_tubes_size(self) -> None:
        text = self.settings_tubes_size_input.text().strip()
        try:
            self.detection_default_tubes_size = self.parse_tubes_size_text(text)
            self.apply_detection_defaults_to_locate_controls(schedule=True)
            self.save_selection_cache()
            self.append_log_message(
                f"Default tube grid updated to {self.detection_default_tubes_size}",
                self.LOG_TAB_ALL,
                self.LOG_LEVEL_INFO,
            )
        except ValueError:
            self.refresh_settings_controls()
            self.append_log_message(
                "Invalid default tube grid. Please enter two positive integers separated by a comma.",
                self.LOG_TAB_ALL,
                self.LOG_LEVEL_WARNING,
            )

    def update_detection_default_rotation(self) -> None:
        rotation = self.settings_rotation_input.text().strip() or 'auto'
        self.detection_default_rotation = rotation
        self.apply_detection_defaults_to_locate_controls(schedule=True)
        self.refresh_settings_controls()
        self.save_selection_cache()
        self.append_log_message(
            f"Default grid rotation updated to {rotation}",
            self.LOG_TAB_ALL,
            self.LOG_LEVEL_INFO,
        )

    def update_detection_default_min_area(self, value: int) -> None:
        self.detection_default_min_area = int(value)
        self.apply_detection_defaults_to_locate_controls(schedule=True)
        self.save_selection_cache()

    def update_detection_default_circularity(self, value: int) -> None:
        self.detection_default_circularity = int(value)
        self.apply_detection_defaults_to_locate_controls(schedule=True)
        self.save_selection_cache()

    def update_restore_last_selected_inputs(self, checked: bool) -> None:
        self.restore_last_selected_inputs = bool(checked)
        self.save_selection_cache()

    def update_auto_save_selected_inputs(self, checked: bool) -> None:
        self.auto_save_selected_inputs = bool(checked)
        self.save_selection_cache()

    def update_auto_open_tube_detection_after_crop(self, checked: bool) -> None:
        self.auto_open_tube_detection_after_crop = bool(checked)
        self.save_selection_cache()

    def update_advance_after_marking_tube_not_available(self, checked: bool) -> None:
        self.advance_after_marking_tube_not_available = bool(checked)
        self.save_selection_cache()

    def update_show_hover_coordinates_in_status_bar(self, checked: bool) -> None:
        self.show_hover_coordinates_in_status_bar = bool(checked)
        if not self.show_hover_coordinates_in_status_bar:
            self.statusBar().clearMessage()
        self.save_selection_cache()

    def update_inp_default_droplet_volume_ul(self, value: float) -> None:
        self.inp_default_droplet_volume_ul = float(value)
        self.save_selection_cache()

    def update_inp_default_dilution_factor(self, value: float) -> None:
        self.inp_default_dilution_factor = float(value)
        self.save_selection_cache()

    def prompt_for_inp_dataset_parameters(
        self,
        label: str = "",
        initial_droplet_volume_ul: float | None = None,
        initial_dilution_factor: float | None = None,
    ) -> tuple[str, float, float] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Results To INP Plot")

        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        label_input = QLineEdit(label)
        label_input.setPlaceholderText("Optional dataset label")
        form_layout.addRow("Dataset label:", label_input)

        volume_spinbox = QDoubleSpinBox()
        volume_spinbox.setRange(0.01, 1000000.0)
        volume_spinbox.setDecimals(3)
        volume_spinbox.setSingleStep(1.0)
        volume_spinbox.setSuffix(" uL")
        volume_spinbox.setValue(self.inp_default_droplet_volume_ul if initial_droplet_volume_ul is None else initial_droplet_volume_ul)
        form_layout.addRow("Droplet volume:", volume_spinbox)

        dilution_spinbox = QDoubleSpinBox()
        dilution_spinbox.setRange(0.000001, 1000000.0)
        dilution_spinbox.setDecimals(6)
        dilution_spinbox.setSingleStep(0.1)
        dilution_spinbox.setValue(self.inp_default_dilution_factor if initial_dilution_factor is None else initial_dilution_factor)
        form_layout.addRow("Dilution factor:", dilution_spinbox)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return None

        return label_input.text().strip(), float(volume_spinbox.value()), float(dilution_spinbox.value())

    def create_display_controls(self) -> QGroupBox:
        display_group = QGroupBox("Display")
        layout = QHBoxLayout(display_group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        label = QLabel("Font size:")
        layout.addWidget(label)

        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(self.MIN_FONT_SIZE, self.MAX_FONT_SIZE)
        self.font_size_spinbox.setValue(self.ui_font_size)
        self.font_size_spinbox.setSuffix(" pt")
        self.font_size_spinbox.setToolTip("Adjust the UI font size for all controls.")
        self.font_size_spinbox.valueChanged.connect(self.set_ui_font_size)
        layout.addWidget(self.font_size_spinbox)
        layout.addStretch(1)

        return display_group

    def create_shortcuts_summary_group(self) -> QGroupBox:
        shortcuts_group = QGroupBox("Keyboard Shortcuts")
        layout = QVBoxLayout(shortcuts_group)
        layout.setSpacing(8)

        shortcuts = (
            "Ctrl+=: Increase font size",
            "Ctrl+-: Decrease font size",
            "Ctrl+0: Reset font size",
            "Ctrl+,: Open Settings tab",
            "Ctrl+1: Open Prepare Image tab",
            "Ctrl+2: Open Locate Tubes tab",
            "Ctrl+3: Open Analyze Freezing tab",
            "Ctrl+4: Open INP Concentration tab",
            "Ctrl+5: Open Settings tab",
        )

        for shortcut_text in shortcuts:
            shortcut_label = QLabel(shortcut_text)
            shortcut_label.setWordWrap(True)
            layout.addWidget(shortcut_label)

        return shortcuts_group

    def configure_shortcuts(self) -> None:
        self.shortcuts = []

        shortcut_definitions = (
            ("Ctrl+=", self.increase_ui_font_size),
            ("Ctrl++", self.increase_ui_font_size),
            ("Ctrl+-", self.decrease_ui_font_size),
            ("Ctrl+0", self.reset_ui_font_size),
            ("Ctrl+,", self.open_settings_tab),
            ("Ctrl+1", lambda: self.select_tab_by_index(0)),
            ("Ctrl+2", lambda: self.select_tab_by_index(1)),
            ("Ctrl+3", lambda: self.select_tab_by_index(2)),
            ("Ctrl+4", lambda: self.select_tab_by_index(3)),
            ("Ctrl+5", lambda: self.select_tab_by_index(4)),
        )

        for key_sequence, callback in shortcut_definitions:
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def increase_ui_font_size(self) -> None:
        self.set_ui_font_size(self.ui_font_size + 1)

    def decrease_ui_font_size(self) -> None:
        self.set_ui_font_size(self.ui_font_size - 1)

    def reset_ui_font_size(self) -> None:
        self.set_ui_font_size(self.get_default_font_size())

    def open_settings_tab(self) -> None:
        self.select_tab_by_index(4)

    def select_tab_by_index(self, index: int) -> None:
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index)

    def set_ui_font_size(self, font_size: int, persist: bool = True) -> None:
        font_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, int(font_size)))
        self.ui_font_size = font_size

        app = QApplication.instance()
        if app is not None:
            app_font = app.font()
            app_font.setPointSize(font_size)
            app.setFont(app_font)

        if hasattr(self, 'font_size_spinbox') and self.font_size_spinbox.value() != font_size:
            self.font_size_spinbox.blockSignals(True)
            self.font_size_spinbox.setValue(font_size)
            self.font_size_spinbox.blockSignals(False)

        self.apply_matplotlib_font_defaults()
        self.apply_styles()
        self.refresh_figure_fonts()

        if hasattr(self, 'ax2') and self.brightness_timeseries is None:
            self.show_analysis_plot_instructions()

        if persist and hasattr(self, 'selection_cache_path'):
            self.save_selection_cache()

    def apply_matplotlib_font_defaults(self) -> None:
        base_font_size = float(self.ui_font_size)
        rcParams['font.size'] = base_font_size
        rcParams['axes.titlesize'] = base_font_size + 1
        rcParams['axes.labelsize'] = base_font_size
        rcParams['xtick.labelsize'] = max(base_font_size - 1, float(self.MIN_FONT_SIZE))
        rcParams['ytick.labelsize'] = max(base_font_size - 1, float(self.MIN_FONT_SIZE))
        rcParams['legend.fontsize'] = max(base_font_size - 1, float(self.MIN_FONT_SIZE))
        rcParams['legend.title_fontsize'] = base_font_size

    def refresh_figure_fonts(self) -> None:
        figure_entries = (
            ('ax', 'canvas'),
            ('ax2', 'canvas2'),
            ('ax_crop', 'canvas_crop'),
            ('ax_inp', 'canvas_inp'),
        )
        for axes_name, canvas_name in figure_entries:
            axes = getattr(self, axes_name, None)
            if axes is None:
                continue
            self.apply_figure_font_sizes(axes)
            canvas = getattr(self, canvas_name, None)
            if canvas is not None:
                canvas.draw_idle()

    def apply_figure_font_sizes(self, axes: Any) -> None:
        title_size = self.ui_font_size + 1
        label_size = self.ui_font_size
        tick_size = max(self.ui_font_size - 1, self.MIN_FONT_SIZE)
        body_size = self.ui_font_size

        axes.title.set_fontsize(title_size)
        axes.xaxis.label.set_fontsize(label_size)
        axes.yaxis.label.set_fontsize(label_size)
        axes.tick_params(axis='both', which='major', labelsize=tick_size)
        axes.tick_params(axis='both', which='minor', labelsize=tick_size)

        x_offset_text = axes.xaxis.get_offset_text()
        y_offset_text = axes.yaxis.get_offset_text()
        if x_offset_text is not None:
            x_offset_text.set_fontsize(tick_size)
        if y_offset_text is not None:
            y_offset_text.set_fontsize(tick_size)

        for text in axes.texts:
            if text.get_fontfamily() == ['monospace']:
                text.set_fontsize(max(self.ui_font_size - 1, self.MIN_FONT_SIZE))
            else:
                current_size = text.get_fontsize() or body_size
                if current_size > body_size:
                    text.set_fontsize(title_size)
                else:
                    text.set_fontsize(body_size)

        legend = axes.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(tick_size)
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontsize(label_size)

    def configure_window_for_screen(self) -> None:
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

    def create_scrollable_panel(self, content_widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(content_widget)
        return scroll_area

    def configure_figure_padding(self, figure: Any, image_mode: bool = False, reserve_title_space: bool = False) -> None:
        if image_mode:
            # top_padding = 0.97 if reserve_title_space else 0.99
            top_padding = 0.97  # consistently reserve space across tabs
            figure.subplots_adjust(left=0.01, right=0.99, top=top_padding, bottom=0.01)
        else:
            figure.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.12)

    def show_plot_error(self, axes: Any, title: str, error_msg: str) -> None:
        axes.clear()
        axes.set_title(title, pad=14)
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.text(
            0.02,
            0.96,
            error_msg,
            transform=axes.transAxes,
            ha='left',
            va='top',
            wrap=True,
            fontsize=max(self.ui_font_size - 1, self.MIN_FONT_SIZE),
            family='monospace',
            bbox=dict(facecolor='red', alpha=0.12, edgecolor='red', boxstyle='round,pad=0.5')
        )
        self.apply_figure_font_sizes(axes)

    def show_analysis_plot_instructions(self) -> None:
        title_font_size = self.ui_font_size + 3
        body_font_size = max(self.ui_font_size, self.MIN_FONT_SIZE)

        self.ax2.clear()
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        # for spine in self.ax2.spines.values():
        #     spine.set_visible(False)

        self.ax2.text(
            0.5,
            0.78,
            "No analysis data loaded yet",
            transform=self.ax2.transAxes,
            ha='center',
            va='center',
            fontsize=title_font_size,
            fontweight='bold',
            color='#17324d',
        )
        self.ax2.text(
            0.5,
            0.52,
            "1. Select the image folder, temperature file, tube-location file, and cutoff timestamp on the right.\n"
            "2. Click 'Load Brightness Timeseries' to generate the plot for each tube.\n"
            "3. Review one tube at a time, then adjust or export the freezing temperatures.",
            transform=self.ax2.transAxes,
            ha='center',
            va='center',
            wrap=True,
            fontsize=body_font_size,
            color='#516579',
            bbox=dict(facecolor='#eef5fb', edgecolor='#d6e4f2', boxstyle='round,pad=0.7'),
        )
        self.apply_figure_font_sizes(self.ax2)
        self.canvas2.draw()

    def show_inp_plot_instructions(self) -> None:
        title_font_size = self.ui_font_size + 3
        body_font_size = max(self.ui_font_size, self.MIN_FONT_SIZE)

        self.ax_inp.clear()
        self.ax_inp.set_xticks([])
        self.ax_inp.set_yticks([])

        self.ax_inp.text(
            0.5,
            0.78,
            "No INP datasets loaded yet",
            transform=self.ax_inp.transAxes,
            ha='center',
            va='center',
            fontsize=title_font_size,
            fontweight='bold',
            color='#17324d',
        )
        self.ax_inp.text(
            0.5,
            0.50,
            "1. Set the droplet volume and dilution factor on the right.\n"
            "2. Add freezing-temperature files, example presets, or the current Analyze Freezing results.\n"
            "3. Compare multiple cumulative INP concentration curves on the same plot.",
            transform=self.ax_inp.transAxes,
            ha='center',
            va='center',
            wrap=True,
            fontsize=body_font_size,
            color='#516579',
            bbox=dict(facecolor='#eef5fb', edgecolor='#d6e4f2', boxstyle='round,pad=0.7'),
        )
        self.apply_figure_font_sizes(self.ax_inp)
        self.canvas_inp.draw()
        
    def get_log_widget(self, tab_number: int) -> QTextEdit | None:
        return logging_get_log_widget(self, tab_number)

    def format_log_message(self, message: object, level: str) -> str:
        return logging_format_log_message(self, message, level)

    def write_log_entry(self, widget: QTextEdit, formatted_message: str) -> None:
        logging_write_log_entry(widget, formatted_message)

    def update_log(self, message: object, tab_number: int, level: str) -> None:
        self.append_log_message(message, tab_number, level)

    def closeEvent(self, event: Any) -> None:
        if getattr(self, 'original_stdout', None) is not None:
            sys.stdout = self.original_stdout
        if getattr(self, 'original_stderr', None) is not None:
            sys.stderr = self.original_stderr
        super().closeEvent(event)

    def apply_styles(self) -> None:
        base_font_size = self.ui_font_size
        tab_header_title_size = base_font_size + 6
        self.setStyleSheet(f"""
            QWidget {{
                font-size: {base_font_size}pt;
            }}
            QMainWindow {{
                background: #f4f7fb;
            }}
            QTabWidget::pane {{
                border: 1px solid #d6dde8;
                background: #ffffff;
                top: -1px;
            }}
            QTabBar::tab {{
                background: #e8eef6;
                border: 1px solid #d6dde8;
                min-width: 180px;
                padding: 10px 18px;
                margin-right: 4px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: #ffffff;
                color: #17324d;
            }}
            QGroupBox {{
                border: 1px solid #d6dde8;
                border-radius: 8px;
                margin-top: 12px;
                padding: 16px 12px 12px 12px;
                font-weight: 600;
                color: #17324d;
                background: #fbfdff;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QPushButton {{
                background: #1f6fb2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                min-height: 22px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #195d95;
            }}
            QPushButton:disabled {{
                background: #b8c5d3;
                color: #eef3f8;
            }}
            QLineEdit, QTextEdit {{
                border: 1px solid #c9d5e2;
                border-radius: 6px;
                padding: 6px;
                background: #ffffff;
            }}
            QLabel#tabHeaderTitle {{
                font-size: {tab_header_title_size}pt;
                font-weight: 700;
                color: #17324d;
            }}
            QLabel#tabHeaderDescription {{
                color: #516579;
                line-height: 1.4;
            }}
            QLabel#statusLabel {{
                color: #2c4f6f;
                background: #eef5fb;
                border: 1px solid #d6e4f2;
                border-radius: 6px;
                padding: 8px 10px;
            }}
            QLabel#hintLabel {{
                color: #516579;
            }}
            QProgressBar#analysisProgressBar {{
                border: 1px solid #c9d5e2;
                border-radius: 6px;
                background: rgba(238, 243, 248, 0.50);
                color: #17324d;
                text-align: center;
                min-height: 22px;
            }}
            QProgressBar#analysisProgressBar::chunk {{
                background: rgba(31, 111, 178, 0.50);
                border-radius: 5px;
            }}
            QProgressBar#analysisProgressBar[completed="true"]::chunk {{
                background: rgba(46, 139, 87, 0.50);
            }}
        """)

    def create_tab_header(self, title: str, description: str) -> QWidget:
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

    def create_status_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("statusLabel")
        label.setWordWrap(True)
        return label

    def format_selected_path(self, prefix: str, path: str | None) -> str:
        selected_name = os.path.basename(path) if path else "Not selected"
        return f"{prefix}: {selected_name}"

    def format_highlighted_selected_path(self, prefix: str, path: str | None) -> str:
        if path:
            selected_name = html.escape(os.path.basename(path))
            return (
                f"{html.escape(prefix)}: "
                f"<span style=\"color:#0b7a75; font-weight:600;\">{selected_name}</span>"
            )

        return (
            f"{html.escape(prefix)}: "
            f"<span style=\"color:#7a8694;\">Not selected</span>"
        )

    def append_log_message(self, message: object, tab_number: int, level: str) -> None:
        logging_append_log_message(self, message, tab_number, level)

    def validate_file_path(self, path: str | None, label: str, tab_number: int) -> bool:
        if not path:
            self.append_log_message(f"{label} is not selected.", tab_number, self.LOG_LEVEL_WARNING)
            return False

        if not os.path.isfile(path):
            self.append_log_message(f"{label} does not exist or is not a file: {path}", tab_number, self.LOG_LEVEL_ERROR)
            return False

        return True

    def validate_directory_path(self, path: str | None, label: str, tab_number: int) -> bool:
        if not path:
            self.append_log_message(f"{label} is not selected.", tab_number, self.LOG_LEVEL_WARNING)
            return False

        if not os.path.isdir(path):
            self.append_log_message(f"{label} does not exist or is not a directory: {path}", tab_number, self.LOG_LEVEL_ERROR)
            return False

        return True

    def load_image_from_path(self, path: str | None, label: str, tab_number: int) -> Any | None:
        if not self.validate_file_path(path, label, tab_number):
            return None

        image = cv2.imread(path)
        if image is None:
            self.append_log_message(f"{label} could not be opened as an image: {path}", tab_number, self.LOG_LEVEL_ERROR)
            return None

        return image

    def validate_analysis_inputs(self) -> bool:
        checks = (
            self.validate_directory_path(self.image_directory, "Image directory", self.LOG_TAB_ANALYZE),
            self.validate_file_path(self.temperature_recording_file, "Temperature recording file", self.LOG_TAB_ANALYZE),
            self.validate_file_path(self.tube_location_file, "Tube locations file", self.LOG_TAB_ANALYZE),
            self.validate_analysis_temperature_cutoff_timestamp(),
        )
        return all(checks)

    def validate_analysis_temperature_cutoff_timestamp(self) -> bool:
        value = (self.analysis_temperature_cutoff_timestamp or "").strip()
        if not value:
            self.append_log_message(
                "Temperature cutoff timestamp is required. Use the format YYYY-MM-DD HH:MM:SS.",
                self.LOG_TAB_ANALYZE,
                self.LOG_LEVEL_ERROR,
            )
            return False

        try:
            normalized_value = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            self.append_log_message(
                f"Invalid temperature cutoff timestamp: {value}. Expected format: YYYY-MM-DD HH:MM:SS.",
                self.LOG_TAB_ANALYZE,
                self.LOG_LEVEL_ERROR,
            )
            return False

        self.analysis_temperature_cutoff_timestamp = normalized_value
        if hasattr(self, 'analysis_temperature_cutoff_input'):
            self.analysis_temperature_cutoff_input.setText(normalized_value)
        if self.auto_save_selected_inputs:
            self.save_selection_cache()
        return True

    def refresh_image_path_labels(self) -> None:
        cache_refresh_image_path_labels(self)

    def refresh_analysis_input_labels(self) -> None:
        cache_refresh_analysis_input_labels(self)

    def load_selection_cache(self) -> dict[str, Any]:
        return cache_load_selection_cache(self)

    def save_selection_cache(self) -> None:
        cache_save_selection_cache(self)

    def restore_cached_selections(self) -> None:
        cache_restore_cached_selections(self)

    def set_standard_button_icon(self, button: QPushButton, icon: QStyle.StandardPixmap) -> None:
        button.setIcon(self.style().standardIcon(icon))

    def create_selection_group(
        self,
        title: str,
        button_text: str,
        selection_method: Callable[[], None],
        icon: QStyle.StandardPixmap | None = None,
    ) -> tuple[QGroupBox, QLabel]:
        group = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        button = QPushButton(button_text)
        if icon is not None:
            self.set_standard_button_icon(button, icon)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.clicked.connect(selection_method)
        
        label = QLabel("Selected item: Not selected")
        label.setWordWrap(True)
        
        layout.addWidget(button)
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group, label

    def setup_tube_locating_tab(self) -> None:
        build_tube_locating_tab(self)

    def setup_freezing_detection_tab(self) -> None:
        build_freezing_detection_tab(self)
    
    def setup_image_cropping_tab(self) -> None:
        build_image_cropping_tab(self)

    def setup_inp_tab(self) -> None:
        build_inp_tab(self)

    def setup_settings_tab(self) -> None:
        build_settings_tab(self)

    def create_crop_selector(self) -> None:
        if self.crop_selector is not None:
            self.crop_selector.set_active(False)

        self.crop_selector = RectangleSelector(
            self.ax_crop,
            self.on_crop_select,
            useblit=True,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
        )

    
    def select_sample_image_path(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Select an Image")
        if file:
            self.sample_image_path = file
            self.refresh_image_path_labels()
            if self.auto_save_selected_inputs:
                self.save_selection_cache()
            self.append_log_message(f"Selected image: {file}", self.LOG_TAB_PREPARE, self.LOG_LEVEL_INFO)
            self.load_selected_image_into_preparation_view()

    def select_image_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if folder:
            self.image_directory = folder
            self.refresh_analysis_input_labels()
            if self.auto_save_selected_inputs:
                self.save_selection_cache()
            self.append_log_message(f"Image directory selected: {folder}", self.LOG_TAB_ANALYZE, self.LOG_LEVEL_INFO)

    def select_temperature_recording(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Select Temperature Recording")
        if file:
            self.temperature_recording_file = file
            self.refresh_analysis_input_labels()
            if self.auto_save_selected_inputs:
                self.save_selection_cache()
            self.append_log_message(f"Temperature recording selected: {file}", self.LOG_TAB_ANALYZE, self.LOG_LEVEL_INFO)

    def select_tube_locations(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Select Tube Locations")
        if file:
            self.tube_location_file = file
            self.refresh_analysis_input_labels()
            if self.auto_save_selected_inputs:
                self.save_selection_cache()
            self.append_log_message(f"Tube locations selected: {file}", self.LOG_TAB_ANALYZE, self.LOG_LEVEL_INFO)

    def update_analysis_temperature_cutoff_timestamp(self) -> None:
        if not hasattr(self, 'analysis_temperature_cutoff_input'):
            return

        self.analysis_temperature_cutoff_timestamp = self.analysis_temperature_cutoff_input.text().strip()
        if self.auto_save_selected_inputs:
            self.save_selection_cache()

    def run_tube_detection_and_render_plot(self) -> None:
        detection_run_tube_detection_and_render_plot(self)

    def handle_tube_detection_plot_click(self, event: Any) -> None:
        detection_handle_tube_detection_plot_click(self, event)

    def redraw_manual_tube_detection_plot(self) -> None:
        detection_redraw_manual_tube_detection_plot(self)

    def reset_tube_detection_view(self) -> None:
        detection_reset_tube_detection_view(self)

    def schedule_update(self) -> None:
        self.update_timer.start(300)  # 300ms 延迟

    def update_min_area(self, value: int) -> None:
        self.min_area_label.setText(f"Min Area: {value}")
        self.schedule_update()
        self.append_log_message(f"Min area updated to {value}", self.LOG_TAB_LOCATE, self.LOG_LEVEL_DEBUG)

    def update_circularity(self, value: int) -> None:
        self.circularity_label.setText(f"Circularity: {value/100:.2f}")
        self.schedule_update()
        self.append_log_message(
            f"Circularity updated to {value/100:.2f}",
            self.LOG_TAB_LOCATE,
            self.LOG_LEVEL_DEBUG,
        )

    def update_tubes_size(self, text: str) -> None:
        try:
            rows, columns = self.parse_tubes_size_text(text)
            self.tubes_size = (rows, columns)
            self.schedule_update()
            self.append_log_message(
                f"Tube grid updated to {self.tubes_size}",
                self.LOG_TAB_LOCATE,
                self.LOG_LEVEL_DEBUG,
            )
        except ValueError:
            self.append_log_message(
                "Invalid input for tubes size. Please enter rows and columns as two integers separated by a comma.",
                self.LOG_TAB_LOCATE,
                self.LOG_LEVEL_WARNING,
            )

    def normalize_inner_circles(self, circles: list[dict[str, Any]], default_method: str = "loaded") -> list[dict[str, Any]]:
        return normalize_inner_circles(circles, default_method=default_method)

    def restore_circle_to_original_image(self, circle: dict[str, Any]) -> dict[str, Any]:
        return restore_circle_to_original_image(circle, self.crop_region, self.rotation_params)

    def save_detected_inner_circles(self) -> None:
        detection_save_detected_inner_circles(self)

    def save_freezing_events_data(self) -> None:
        analysis_save_freezing_events_data(self)
            
    def load_freezing_events_data(self) -> None:
        analysis_load_freezing_events_data(self)
        
    def start_brightness_series_analysis(self) -> None:
        analysis_start_brightness_series_analysis(self)

    def set_analysis_progress_completed(self, completed: bool) -> None:
        if not hasattr(self, 'analysis_progress_bar'):
            return

        self.analysis_progress_bar.setProperty("completed", bool(completed))
        style = self.analysis_progress_bar.style()
        style.unpolish(self.analysis_progress_bar)
        style.polish(self.analysis_progress_bar)
        self.analysis_progress_bar.update()
    
    def update_progress(self, value: int) -> None:
        self.analysis_progress_bar.setValue(max(0, min(100, value)))

    def apply_analysis_results(self, temperature_recordings: Any, brightness_timeseries: Any) -> None:
        analysis_apply_analysis_results(self, temperature_recordings, brightness_timeseries)

    def update_subprocess_log(self, message: str) -> None:
        self.update_log_signal.emit(message, self.LOG_TAB_ANALYZE, self.LOG_LEVEL_INFO)
        
    def enable_analysis_review_controls(self) -> None:
        analysis_enable_analysis_review_controls(self)

    def next_tube(self) -> None:
        if not self.sorted_tube_indices:
            self.sorted_tube_indices = self.get_sorted_tube_indices()

        if self.sorted_tube_position < len(self.sorted_tube_indices) - 1:
            self.sorted_tube_position += 1
            self.current_tube = self.sorted_tube_indices[self.sorted_tube_position]
            self.refresh_current_tube_brightness_plot()

    def previous_tube(self) -> None:
        if not self.sorted_tube_indices:
            self.sorted_tube_indices = self.get_sorted_tube_indices()

        if self.sorted_tube_position > 0:
            self.sorted_tube_position -= 1
            self.current_tube = self.sorted_tube_indices[self.sorted_tube_position]
            self.refresh_current_tube_brightness_plot()

    def discard_current_tube_freezing_point(self) -> None:
        analysis_discard_current_tube_freezing_point(self)

    def go_to_tube(self, *_args: object) -> None:
        tube_number = None
        if hasattr(self, 'tube_selector_combo'):
            selected_data = self.tube_selector_combo.currentData()
            if isinstance(selected_data, int):
                tube_number = selected_data

        if tube_number is None:
            self.append_log_message("Select a tube from the list.", self.LOG_TAB_ANALYZE, self.LOG_LEVEL_WARNING)
            return

        if 0 <= tube_number < self.num_tubes:
            self.current_tube = tube_number
            if not self.sorted_tube_indices:
                self.sorted_tube_indices = self.get_sorted_tube_indices()
            if tube_number in self.sorted_tube_indices:
                self.sorted_tube_position = self.sorted_tube_indices.index(tube_number)
            self.refresh_current_tube_brightness_plot()
            return

        self.append_log_message(
            f"Invalid tube number. Please select a value between 0 and {self.num_tubes - 1}.",
            self.LOG_TAB_ANALYZE,
            self.LOG_LEVEL_WARNING,
        )
            
    def refresh_current_tube_brightness_plot(self) -> None:
        analysis_refresh_current_tube_brightness_plot(self)

    def refresh_current_tube_freezing_marker(self, xmin: float | None = None, xmax: float | None = None) -> None:
        analysis_refresh_current_tube_freezing_marker(self, xmin=xmin, xmax=xmax)

    def on_brightness_span_select(self, xmin: float, xmax: float) -> None:
        self.refresh_current_tube_freezing_marker(xmin, xmax)

    def load_selected_image_into_preparation_view(self) -> None:
        image_load_selected_image_into_preparation_view(self)

    def apply_preparation_image_rotation(self) -> None:
        image_apply_preparation_image_rotation(self)

    def restore_original_preparation_image(self) -> None:
        image_restore_original_preparation_image(self)

    def on_crop_select(self, eclick: Any, erelease: Any) -> None:
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self.append_log_message(f"Crop region set to: {self.crop_region}", self.LOG_TAB_PREPARE, self.LOG_LEVEL_INFO)

    def apply_selected_crop_to_tube_detection(self) -> None:
        image_apply_selected_crop_to_tube_detection(self)

    def load_analysis_inner_circle_locations(self) -> bool:
        return analysis_load_analysis_inner_circle_locations(self)

    def add_inp_dataset_from_files(self) -> None:
        inp_add_inp_dataset_from_files(self)

    def prompt_and_add_inp_dataset_from_files(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Freezing Temperatures",
            ".",
            "Text Files (*.txt)",
        )
        if not file_paths:
            self.append_log_message("No freezing-temperature file selected.", self.LOG_TAB_INP, self.LOG_LEVEL_WARNING)
            return

        initial_label = os.path.basename(file_paths[0]) if len(file_paths) == 1 else ""
        parameters = self.prompt_for_inp_dataset_parameters(label=initial_label)
        if parameters is None:
            return

        label, droplet_volume_ul, dilution_factor = parameters
        inp_add_inp_dataset_from_files(
            self,
            label=label,
            droplet_volume_ul=droplet_volume_ul,
            dilution_factor=dilution_factor,
            file_paths=file_paths,
        )

    def add_selected_inp_preset(self) -> None:
        inp_add_selected_inp_preset(self)

    def prompt_and_add_selected_inp_preset(self) -> None:
        parameters = self.prompt_for_inp_dataset_parameters(label=self.inp_preset_combo.currentText())
        if parameters is None:
            return

        label, droplet_volume_ul, dilution_factor = parameters
        inp_add_selected_inp_preset(
            self,
            label=label,
            droplet_volume_ul=droplet_volume_ul,
            dilution_factor=dilution_factor,
        )

    def add_current_analysis_to_inp(self) -> None:
        inp_add_current_analysis_to_inp(self)

    def prompt_and_add_current_analysis_to_inp(self) -> None:
        parameters = self.prompt_for_inp_dataset_parameters(label="Analyze Freezing")
        if parameters is None:
            return

        label, droplet_volume_ul, dilution_factor = parameters

        inp_add_current_analysis_to_inp(
            self,
            label=label,
            droplet_volume_ul=droplet_volume_ul,
            dilution_factor=dilution_factor,
            auto_export=True,
        )

    def prompt_and_add_current_analysis_to_inp_from_tab4(self) -> None:
        parameters = self.prompt_for_inp_dataset_parameters(label="Analyze Freezing")
        if parameters is None:
            return

        label, droplet_volume_ul, dilution_factor = parameters

        inp_add_current_analysis_to_inp(
            self,
            label=label,
            droplet_volume_ul=droplet_volume_ul,
            dilution_factor=dilution_factor,
            auto_export=False,
        )

    def remove_selected_inp_dataset(self) -> None:
        inp_remove_selected_inp_dataset(self)

    def clear_inp_datasets(self) -> None:
        inp_clear_inp_datasets(self)

    def refresh_inp_plot(self) -> None:
        inp_refresh_inp_plot(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = InteractivePlot()
    if getattr(main_window, 'is_compact_screen', False):
        main_window.showMaximized()
    else:
        main_window.show()
    sys.exit(app.exec())
