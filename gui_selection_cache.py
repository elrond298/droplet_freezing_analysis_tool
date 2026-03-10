import json
import os


def refresh_image_path_labels(window):
    if hasattr(window, 'sample_image_path_label'):
        window.sample_image_path_label.setText(
            window.format_highlighted_selected_path("Current image", window.sample_image_path)
        )
    if hasattr(window, 'tube_image_summary_label'):
        window.tube_image_summary_label.setText(
            window.format_highlighted_selected_path("Tube detection source", window.sample_image_path)
        )


def refresh_analysis_input_labels(window):
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


def load_selection_cache(window):
    if not os.path.isfile(window.selection_cache_path):
        return {}

    try:
        with open(window.selection_cache_path, 'r', encoding='utf-8') as cache_file:
            cached_data = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return cached_data if isinstance(cached_data, dict) else {}


def save_selection_cache(window):
    cached_data = {
        'sample_image_path': window.sample_image_path,
        'image_directory': window.image_directory,
        'temperature_recording_file': window.temperature_recording_file,
        'tube_location_file': window.tube_location_file,
        'ui_font_size': window.ui_font_size,
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


def restore_cached_selections(window):
    cached_data = load_selection_cache(window)

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

    ui_font_size = cached_data.get('ui_font_size')
    if isinstance(ui_font_size, int):
        window.set_ui_font_size(ui_font_size, persist=False)

    refresh_image_path_labels(window)
    refresh_analysis_input_labels(window)

    if window.sample_image_path:
        window.load_selected_image_into_preparation_view()