import pickle

import cv2
import numpy as np
import pandas as pd

from freezing_detection import get_freezing_temperature
from tube_detection import detect_inner_circles, infer_missing_tubes, locate_pcr_tubes


def run_tube_detection(image, min_area, circularity_threshold, tubes_size, rotation):
    pcr_tubes, _ = locate_pcr_tubes(image, min_area, circularity_threshold)
    inferred_tubes = infer_missing_tubes(
        pcr_tubes,
        image.shape,
        tubes_size=tubes_size,
        rotate=rotation,
    )
    all_tubes = pcr_tubes + inferred_tubes
    inner_circles = detect_inner_circles(image, all_tubes)
    return pcr_tubes, inferred_tubes, all_tubes, inner_circles


def render_tube_detection_overlay(image, all_tubes, inner_circles):
    img_with_tubes = image.copy()

    for tube, inner_circle in zip(all_tubes, inner_circles):
        color = (0, 255, 0) if 'inferred' not in tube else (0, 0, 255)
        cv2.circle(img_with_tubes, (tube['x'], tube['y']), tube['radius'], color, 2)

        if inner_circle:
            cv2.circle(
                img_with_tubes,
                (inner_circle['x'], inner_circle['y']),
                inner_circle['radius'],
                (0, 0, 0),
                1,
            )

    return img_with_tubes


def render_manual_detection_overlay(image, all_tubes, inner_circles):
    img_with_tubes = image.copy()

    for tube in all_tubes:
        color = (0, 255, 0) if 'inferred' not in tube else (0, 0, 255)
        cv2.circle(img_with_tubes, (tube['x'], tube['y']), tube['radius'], color, 2)

    for circle in inner_circles:
        cv2.circle(img_with_tubes, (circle['x'], circle['y']), circle['radius'], (0, 0, 0), 1)

    return img_with_tubes


def normalize_inner_circles(circles, default_method="loaded"):
    normalized_circles = []
    for circle in circles:
        normalized_circle = dict(circle)
        normalized_circle['x'] = int(round(normalized_circle['x']))
        normalized_circle['y'] = int(round(normalized_circle['y']))
        normalized_circle['radius'] = int(round(normalized_circle.get('radius', 10)))
        normalized_circle['method'] = normalized_circle.get('method', default_method)
        normalized_circles.append(normalized_circle)
    return normalized_circles


def restore_circle_to_original_image(circle, crop_region, rotation_params):
    restored_circle = dict(circle)
    restored_circle['x'] = int(round(restored_circle['x']))
    restored_circle['y'] = int(round(restored_circle['y']))
    restored_circle['radius'] = int(round(restored_circle.get('radius', 10)))

    if crop_region is not None:
        x_offset, y_offset, _, _ = crop_region
        restored_circle['x'] += x_offset
        restored_circle['y'] += y_offset

    if rotation_params is not None:
        inverse_matrix = cv2.invertAffineTransform(rotation_params['matrix'])
        original_x, original_y = inverse_matrix @ np.array(
            [restored_circle['x'], restored_circle['y'], 1.0],
            dtype=float,
        )
        restored_circle['x'] = int(round(original_x))
        restored_circle['y'] = int(round(original_y))

    return restored_circle


def dump_inner_circles(file_path, circles):
    with open(file_path, 'wb') as file_handle:
        pickle.dump(circles, file_handle)


def serialize_freezing_temperatures(freezing_temperatures):
    lines = ["Tube,Temperature,Timestamp\n"]

    for tube, data in freezing_temperatures.items():
        temperature = data['temperature']
        timestamp = data['timestamp']
        if temperature is not None and timestamp is not None:
            datetime_str = pd.Timestamp(timestamp).isoformat()
            lines.append(f"{tube},{temperature:.4f},{datetime_str}\n")
        else:
            lines.append(f"{tube},N/A,N/A\n")

    return lines


def deserialize_freezing_temperatures(file_path):
    freezing_temperatures = {}
    errors = []

    with open(file_path, 'r') as file_handle:
        next(file_handle)
        for line in file_handle:
            try:
                tube, temperature, datetime_str = line.strip().split(',')
                tube = int(tube)
                if temperature != 'N/A':
                    freezing_temperatures[tube] = {
                        'temperature': float(temperature),
                        'timestamp': pd.to_datetime(datetime_str).to_numpy(),
                    }
                else:
                    freezing_temperatures[tube] = {
                        'temperature': None,
                        'timestamp': None,
                    }
            except ValueError as error:
                errors.append((line, error))

    return freezing_temperatures, errors


def compute_analysis_results(temperature_recordings, brightness_timeseries):
    freezing_temperatures = get_freezing_temperature(temperature_recordings, brightness_timeseries)
    valid_freezing_points = sum(
        1
        for data in freezing_temperatures.values()
        if data['temperature'] is not None and data['timestamp'] is not None
    )
    return freezing_temperatures, valid_freezing_points


def build_current_tube_series(temperature_recordings, brightness_timeseries, current_tube):
    common_timestamps = np.intersect1d(
        temperature_recordings['timestamp'],
        brightness_timeseries['timestamp'],
    )
    temp_indices = np.searchsorted(temperature_recordings['timestamp'], common_timestamps)
    bright_indices = np.searchsorted(brightness_timeseries['timestamp'], common_timestamps)

    current_tube_temperature = temperature_recordings['temperature'][temp_indices]
    current_tube_brightness = brightness_timeseries[current_tube][bright_indices]
    current_tube_timestamps = common_timestamps

    return current_tube_temperature, current_tube_brightness, current_tube_timestamps


def resolve_existing_freezing_point(freezing_temperatures, current_tube, timestamps, brightness):
    if current_tube not in freezing_temperatures:
        return None

    freezing_data = freezing_temperatures[current_tube]
    freezing_temp = freezing_data['temperature']
    freezing_timestamp = freezing_data['timestamp']
    if freezing_temp is None or freezing_timestamp is None:
        return None

    freezing_temp_index = np.argmin(np.abs(timestamps - freezing_timestamp))
    freezing_brightness = brightness[freezing_temp_index]
    return {
        'temperature': freezing_temp,
        'timestamp': freezing_timestamp,
        'brightness': freezing_brightness,
    }


def recalculate_freezing_point_in_range(temperature, brightness, timestamps, xmin, xmax):
    mask = (temperature >= xmin) & (temperature <= xmax)
    if np.sum(mask) < 3:
        return None

    temp_range = temperature[mask]
    if hasattr(temp_range, 'to_numpy'):
        temp_range = temp_range.to_numpy()

    bright_range = brightness[mask]
    time_range = timestamps[mask]
    brightness_derivative = np.diff(bright_range)
    freezing_index = np.argmin(brightness_derivative)
    return {
        'temperature': temp_range[freezing_index],
        'timestamp': time_range[freezing_index],
        'brightness': bright_range[freezing_index],
    }


def discard_freezing_point(brightness, timestamps):
    return {
        'temperature': None,
        'timestamp': None,
        'brightness': None,
    }


def rotate_image(image, rotation_angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    rotation_params = {
        'angle': rotation_angle,
        'center': center,
        'matrix': rotation_matrix,
    }
    return rotated_image, rotation_params


def crop_rotated_image(rotated_image, crop_region):
    x_pos, y_pos, width, height = crop_region
    return rotated_image[y_pos:y_pos + height, x_pos:x_pos + width]


def load_inner_circles_from_pickle(file_path):
    with open(file_path, 'rb') as file_handle:
        return normalize_inner_circles(pickle.load(file_handle))