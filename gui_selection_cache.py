from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gui import InteractivePlot


def refresh_image_path_labels(window: InteractivePlot) -> None:
    if hasattr(window, 'sample_image_path_label'):
        window.sample_image_path_label.setText(
            window.format_highlighted_selected_path("Current image", window.sample_image_path)
        )
    if hasattr(window, 'tube_image_summary_label'):
        window.tube_image_summary_label.setText(
            window.format_highlighted_selected_path("Tube detection source", window.sample_image_path)
        )


def refresh_analysis_input_labels(window: InteractivePlot) -> None:
    if hasattr(window, 'image_directory_label'):
        window.image_directory_label.setText(
            window.format_highlighted_selected_path("Selected image folder", window.image_directory)
        )
    if hasattr(window, 'temperature_recording_label'):
        window.temperature_recording_label.setText(
            window.format_highlighted_selected_path("Selected temperature file", window.temperature_recording_file)
        )
    if hasattr(window, 'tube_locations_label'):
        window.tube_locations_label.setText(
            window.format_highlighted_selected_path("Selected tube-location file", window.tube_location_file)
        )


def load_selection_cache(window: InteractivePlot) -> dict[str, Any]:
    if not os.path.isfile(window.selection_cache_path):
        return {}

    try:
        with open(window.selection_cache_path, 'r', encoding='utf-8') as cache_file:
            cached_data = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return cached_data if isinstance(cached_data, dict) else {}


def save_selection_cache(window: InteractivePlot) -> None:
    existing_cached_data = load_selection_cache(window)

    sample_image_path = window.sample_image_path
    image_directory = window.image_directory
    temperature_recording_file = window.temperature_recording_file
    tube_location_file = window.tube_location_file

    if not window.auto_save_selected_inputs:
        sample_image_path = existing_cached_data.get('sample_image_path')
        image_directory = existing_cached_data.get('image_directory')
        temperature_recording_file = existing_cached_data.get('temperature_recording_file')
        tube_location_file = existing_cached_data.get('tube_location_file')

    cached_data = {
        'sample_image_path': sample_image_path,
        'image_directory': image_directory,
        'temperature_recording_file': temperature_recording_file,
        'tube_location_file': tube_location_file,
        'ui_font_size': window.ui_font_size,
        'detection_default_tubes_size': list(window.detection_default_tubes_size),
        'detection_default_rotation': window.detection_default_rotation,
        'detection_default_min_area': window.detection_default_min_area,
        'detection_default_circularity': window.detection_default_circularity,
        'restore_last_selected_inputs': window.restore_last_selected_inputs,
        'auto_save_selected_inputs': window.auto_save_selected_inputs,
        'auto_open_tube_detection_after_crop': window.auto_open_tube_detection_after_crop,
        'show_hover_coordinates_in_status_bar': window.show_hover_coordinates_in_status_bar,
        'inp_default_droplet_volume_ul': window.inp_default_droplet_volume_ul,
        'inp_default_dilution_factor': window.inp_default_dilution_factor,
    }
    temp_cache_path = f"{window.selection_cache_path}.tmp"

    try:
        with open(temp_cache_path, 'w', encoding='utf-8') as cache_file:
            json.dump(cached_data, cache_file, indent=2)
        os.replace(temp_cache_path, window.selection_cache_path)
    except OSError:
        if os.path.exists(temp_cache_path):
            try:
                os.remove(temp_cache_path)
            except OSError:
                pass


def restore_cached_selections(window: InteractivePlot) -> None:
    cached_data = load_selection_cache(window)

    detection_default_tubes_size = cached_data.get('detection_default_tubes_size')
    if (
        isinstance(detection_default_tubes_size, (list, tuple))
        and len(detection_default_tubes_size) == 2
        and all(isinstance(value, int) and value > 0 for value in detection_default_tubes_size)
    ):
        window.detection_default_tubes_size = tuple(detection_default_tubes_size)

    detection_default_rotation = cached_data.get('detection_default_rotation')
    if isinstance(detection_default_rotation, str) and detection_default_rotation.strip():
        window.detection_default_rotation = detection_default_rotation.strip()

    detection_default_min_area = cached_data.get('detection_default_min_area')
    if isinstance(detection_default_min_area, int):
        window.detection_default_min_area = detection_default_min_area

    detection_default_circularity = cached_data.get('detection_default_circularity')
    if isinstance(detection_default_circularity, int):
        window.detection_default_circularity = detection_default_circularity

    restore_last_selected_inputs = cached_data.get('restore_last_selected_inputs')
    if isinstance(restore_last_selected_inputs, bool):
        window.restore_last_selected_inputs = restore_last_selected_inputs

    auto_save_selected_inputs = cached_data.get('auto_save_selected_inputs')
    if isinstance(auto_save_selected_inputs, bool):
        window.auto_save_selected_inputs = auto_save_selected_inputs

    auto_open_tube_detection_after_crop = cached_data.get('auto_open_tube_detection_after_crop')
    if isinstance(auto_open_tube_detection_after_crop, bool):
        window.auto_open_tube_detection_after_crop = auto_open_tube_detection_after_crop

    show_hover_coordinates_in_status_bar = cached_data.get('show_hover_coordinates_in_status_bar')
    if isinstance(show_hover_coordinates_in_status_bar, bool):
        window.show_hover_coordinates_in_status_bar = show_hover_coordinates_in_status_bar

    inp_default_droplet_volume_ul = cached_data.get('inp_default_droplet_volume_ul')
    if isinstance(inp_default_droplet_volume_ul, (int, float)) and inp_default_droplet_volume_ul > 0:
        window.inp_default_droplet_volume_ul = float(inp_default_droplet_volume_ul)

    inp_default_dilution_factor = cached_data.get('inp_default_dilution_factor')
    if isinstance(inp_default_dilution_factor, (int, float)) and inp_default_dilution_factor > 0:
        window.inp_default_dilution_factor = float(inp_default_dilution_factor)

    ui_font_size = cached_data.get('ui_font_size')
    if isinstance(ui_font_size, int):
        window.set_ui_font_size(ui_font_size, persist=False)

    # Restore settings-driven controls before reapplying cached inputs so the UI and defaults stay in sync.
    window.refresh_settings_controls()
    window.apply_detection_defaults_to_locate_controls(schedule=False)

    if window.restore_last_selected_inputs:
        sample_image_path = cached_data.get('sample_image_path')
        if isinstance(sample_image_path, str) and os.path.isfile(sample_image_path):
            window.sample_image_path = sample_image_path

        image_directory = cached_data.get('image_directory')
        if isinstance(image_directory, str) and os.path.isdir(image_directory):
            window.image_directory = image_directory

        temperature_recording_file = cached_data.get('temperature_recording_file')
        if isinstance(temperature_recording_file, str) and os.path.isfile(temperature_recording_file):
            window.temperature_recording_file = temperature_recording_file

        tube_location_file = cached_data.get('tube_location_file')
        if isinstance(tube_location_file, str) and os.path.isfile(tube_location_file):
            window.tube_location_file = tube_location_file

    refresh_image_path_labels(window)
    refresh_analysis_input_labels(window)

    if window.sample_image_path:
        window.load_selected_image_into_preparation_view()