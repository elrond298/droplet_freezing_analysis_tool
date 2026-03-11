from __future__ import annotations

import datetime
import os
import traceback
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QFileDialog
from matplotlib.widgets import SpanSelector

from gui_services import (
    build_current_tube_series,
    compute_analysis_results,
    deserialize_freezing_temperatures,
    discard_freezing_point,
    load_inner_circles_from_pickle,
    recalculate_freezing_point_in_range,
    resolve_existing_freezing_point,
    serialize_freezing_temperatures,
)
from gui_workers import BrightnessWorker

if TYPE_CHECKING:
    from gui import InteractivePlot


def save_freezing_events_data(window: InteractivePlot) -> None:
    default_filename = f"freezing_temperatures_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    default_filepath = os.path.join(".", default_filename)

    file_path, _ = QFileDialog.getSaveFileName(window, "Save Freezing Temperatures", default_filepath, "Text Files (*.txt)")
    if file_path:
        with open(file_path, 'w') as file_handle:
            file_handle.writelines(serialize_freezing_temperatures(window.freezing_temperatures))

        window.append_log_message(
            f"Freezing temperatures saved to {file_path}",
            window.LOG_TAB_ANALYZE,
            window.LOG_LEVEL_SUCCESS,
        )


def load_freezing_events_data(window: InteractivePlot) -> None:
    file_path, _ = QFileDialog.getOpenFileName(window, "Load Freezing Temperatures", ".", "Text Files (*.txt)")
    if file_path:
        try:
            window.freezing_temperatures, errors = deserialize_freezing_temperatures(file_path)
            for line, error in errors:
                window.append_log_message(
                    f"Error parsing line: {line}. Error: {str(error)}",
                    window.LOG_TAB_ANALYZE,
                    window.LOG_LEVEL_ERROR,
                )

            window.append_log_message(
                f"Freezing temperatures loaded from {file_path}",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_SUCCESS,
            )
            window.append_log_message(
                f"Loaded data for {len(window.freezing_temperatures)} tubes",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_INFO,
            )

            if hasattr(window, 'current_tube'):
                refresh_current_tube_brightness_plot(window)

        except Exception as error:
            window.append_log_message(
                f"Error loading file: {str(error)}",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_ERROR,
            )
    else:
        window.append_log_message("No file selected", window.LOG_TAB_ANALYZE, window.LOG_LEVEL_WARNING)


def load_analysis_inner_circle_locations(window: InteractivePlot) -> bool:
    try:
        if not window.validate_file_path(window.tube_location_file, "Tube locations file", window.LOG_TAB_ANALYZE):
            return False

        window.inner_circles = load_inner_circles_from_pickle(window.tube_location_file)
        window.append_log_message(
            f"Loaded {len(window.inner_circles)} inner circles from {window.tube_location_file}",
            window.LOG_TAB_ANALYZE,
            window.LOG_LEVEL_SUCCESS,
        )
        return True
    except Exception as error:
        window.append_log_message(
            f"Error loading inner circles: {str(error)}",
            window.LOG_TAB_ANALYZE,
            window.LOG_LEVEL_ERROR,
        )
        return False


def start_brightness_series_analysis(window: InteractivePlot) -> None:
    if not window.validate_analysis_inputs():
        return

    if not load_analysis_inner_circle_locations(window):
        return

    window.start_load_timeseries_button.setEnabled(False)
    window.analysis_progress_bar.setValue(0)
    window.analysis_progress_bar.setFormat("Starting brightness-timeseries analysis for the selected inputs... %p%")

    window.thread = QThread()
    window.worker = BrightnessWorker(window.image_directory, window.tube_location_file, window.temperature_recording_file)
    window.worker.moveToThread(window.thread)
    window.thread.started.connect(window.worker.run)
    window.worker.finished.connect(window.thread.quit)
    window.worker.finished.connect(window.worker.deleteLater)
    window.thread.finished.connect(window.thread.deleteLater)
    window.worker.progress.connect(window.update_progress)
    window.worker.finished.connect(window.apply_analysis_results)
    window.worker.log.connect(window.update_subprocess_log)
    window.thread.start()

    window.thread.finished.connect(lambda: window.start_load_timeseries_button.setEnabled(True))
    window.thread.finished.connect(
        lambda: window.append_log_message("Analysis completed!", window.LOG_TAB_ANALYZE, window.LOG_LEVEL_SUCCESS)
    )


def apply_analysis_results(window: InteractivePlot, temperature_recordings: Any, brightness_timeseries: Any) -> None:
    window.temperature_recordings = temperature_recordings
    window.brightness_timeseries = brightness_timeseries
    window.analysis_progress_bar.setValue(100)
    window.analysis_progress_bar.setFormat("Analysis complete. Drag a span to adjust temperature below. (%p%)")
    window.freezing_temperatures, valid_freezing_points = compute_analysis_results(
        window.temperature_recordings,
        window.brightness_timeseries,
    )
    window.num_tubes = len(window.inner_circles)
    window.current_tube = 0
    window.append_log_message("Data loaded successfully!", window.LOG_TAB_ANALYZE, window.LOG_LEVEL_SUCCESS)
    window.append_log_message(
        f"Detected freezing points for {valid_freezing_points} out of {window.num_tubes} tubes",
        window.LOG_TAB_ANALYZE,
        window.LOG_LEVEL_INFO,
    )

    refresh_current_tube_brightness_plot(window)
    enable_analysis_review_controls(window)


def enable_analysis_review_controls(window: InteractivePlot) -> None:
    window.next_button.setEnabled(True)
    window.prev_button.setEnabled(True)
    window.discard_button.setEnabled(True)
    window.value_input.setEnabled(True)
    window.save_button_freezing_temperatures.setEnabled(True)
    window.load_button_freezing_temperatures.setEnabled(True)


def discard_current_tube_freezing_point(window: InteractivePlot) -> None:
    if not hasattr(window, 'current_tube_brightness') or not hasattr(window, 'current_tube_timestamps'):
        window.append_log_message(
            "No tube data is loaded. Run the analysis before discarding a tube.",
            window.LOG_TAB_ANALYZE,
            window.LOG_LEVEL_WARNING,
        )
        return

    freezing_data = discard_freezing_point(window.current_tube_brightness, window.current_tube_timestamps)
    _render_freezing_point(window, freezing_data, persist=True)


def refresh_current_tube_brightness_plot(window: InteractivePlot) -> None:
    try:
        if not hasattr(window, 'brightness_timeseries') or not window.brightness_timeseries:
            window.append_log_message(
                "No data loaded. Please run the analysis first.",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_WARNING,
            )
            return

        if window.current_tube < len(window.inner_circles):
            window.ax2.clear()
            (
                window.current_tube_temperature,
                window.current_tube_brightness,
                window.current_tube_timestamps,
            ) = build_current_tube_series(
                window.temperature_recordings,
                window.brightness_timeseries,
                window.current_tube,
            )

            window.line, = window.ax2.plot(window.current_tube_temperature, window.current_tube_brightness, 'b-')
            window.ax2.invert_xaxis()
            window.ax2.set_xlabel("Temperature (°C)")
            window.ax2.set_ylabel("Brightness")
            window.ax2.set_title(f"Brightness vs Temperature for Tube {window.current_tube}")
            window.ax2.set_xlim((0, window.temperature_recordings['temperature'].min()))

            refresh_current_tube_freezing_marker(window)

            window.span = SpanSelector(
                window.ax2,
                window.on_brightness_span_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.5, facecolor='red'),
                interactive=True,
                drag_from_anywhere=True,
            )

            window.canvas2.draw()
            window.append_log_message(
                f"Displaying data for tube {window.current_tube}",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_DEBUG,
            )
        else:
            window.append_log_message(
                f"Invalid index: {window.current_tube}",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_WARNING,
            )

    except Exception as error:
        error_msg = f"Error: {str(error)}\n\n{traceback.format_exc()}"
        window.show_plot_error(window.ax2, "An error occurred", error_msg)
        window.append_log_message(error_msg, window.LOG_TAB_ANALYZE, window.LOG_LEVEL_ERROR)

    finally:
        window.canvas2.draw()


def refresh_current_tube_freezing_marker(window: InteractivePlot, xmin: float | None = None, xmax: float | None = None) -> None:
    if xmin is None and xmax is None:
        freezing_data = resolve_existing_freezing_point(
            window.freezing_temperatures,
            window.current_tube,
            window.current_tube_timestamps,
            window.current_tube_brightness,
        )
        if freezing_data is None:
            if window.current_tube not in window.freezing_temperatures:
                window.append_log_message(
                    f"No freezing data available for tube {window.current_tube}",
                    window.LOG_TAB_ANALYZE,
                    window.LOG_LEVEL_WARNING,
                )
            else:
                window.append_log_message(
                    f"No freezing point detected for tube {window.current_tube}",
                    window.LOG_TAB_ANALYZE,
                    window.LOG_LEVEL_WARNING,
                )
            return
    else:
        freezing_data = recalculate_freezing_point_in_range(
            window.current_tube_temperature,
            window.current_tube_brightness,
            window.current_tube_timestamps,
            xmin,
            xmax,
        )
        if freezing_data is None:
            window.append_log_message(
                "Selected range is too small. Please select a larger range.",
                window.LOG_TAB_ANALYZE,
                window.LOG_LEVEL_WARNING,
            )
            return

    _render_freezing_point(window, freezing_data, persist=xmin is not None or xmax is not None)


def _render_freezing_point(window: InteractivePlot, freezing_data: dict[str, Any], persist: bool) -> None:
    freezing_temp = freezing_data['temperature']
    freezing_timestamp = freezing_data['timestamp']
    freezing_brightness = freezing_data['brightness']

    if persist:
        window.freezing_temperatures[window.current_tube] = {
            'temperature': freezing_temp,
            'timestamp': freezing_timestamp,
        }

    if hasattr(window, 'freezing_point'):
        window.freezing_point.set_data([], [])
        window.freezing_point.set_label("")

    if freezing_temp is None or freezing_timestamp is None or freezing_brightness is None:
        if window.ax2.get_legend() is not None:
            window.ax2.get_legend().remove()
        window.canvas2.draw()
        window.append_log_message(
            f"Marked tube {window.current_tube} as Not Available.",
            window.LOG_TAB_ANALYZE,
            window.LOG_LEVEL_INFO,
        )
        return

    window.freezing_point, = window.ax2.plot(
        freezing_temp,
        freezing_brightness,
        'ro',
        markersize=10,
        label=f"Freezing Point: {freezing_temp:.2f}°C",
    )
    window.ax2.legend()
    window.canvas2.draw()
    window.append_log_message(
        f"Updated freezing point for tube {window.current_tube}: {freezing_temp:.2f}°C at timestamp {freezing_timestamp}",
        window.LOG_TAB_ANALYZE,
        window.LOG_LEVEL_INFO,
    )