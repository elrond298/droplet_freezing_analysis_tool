import pandas as pd
import numpy as np
import os
import datetime
from PIL import Image
import pickle
from collections import defaultdict
import multiprocessing as mp
from scipy.signal import find_peaks

def load_temperature_timeseries(temperature_recordings):
    """
    Loads and processes temperature timeseries data from a CSV file.

    Args:
        temperature_recordings (str): Path to the CSV file containing temperature recordings.

    Returns:
        pd.DataFrame: A DataFrame with columns 'timestamp' and 'temperature'.
                      'timestamp' is a datetime object.
                      'temperature' is the average temperature across multiple sensors.
    """
    # Read the CSV file, skipping the specified rows
    df = pd.read_csv(temperature_recordings, skiprows=[0,2,3])

    # Convert TIMESTAMP to datetime
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    # Apply the cutoff timestamp
    TEMPERATURE_CUTOFF_TIMESTAMP = "2023-04-02 14:00:00"
    cutoff_timestamp = pd.to_datetime(TEMPERATURE_CUTOFF_TIMESTAMP)
    df = df[df["TIMESTAMP"] >= cutoff_timestamp]

    # Create a new DataFrame with the timestamp column
    new_df = pd.DataFrame()
    new_df["timestamp"] = df["TIMESTAMP"]

    # Add temperature columns, converting "NAN" to np.nan
    for i in range(1, 9):
        column = f"RT_C_Avg({i})"
        new_df[column] = pd.to_numeric(df[column], errors='coerce')

    # Calculate the mean temperature, excluding column 7
    temperature_columns = [f"RT_C_Avg({i})" for i in range(1, 9) if i != 7]
    new_df["temperature"] = new_df[temperature_columns].mean(axis=1)

    # Drop rows with any NaN values
    new_df = new_df.dropna()

    # Reset the index
    new_df = new_df.reset_index(drop=True)

    return new_df

def parse_timestamp_from_filename(filename):
    """
    Parses a timestamp from the filename.

    Args:
        filename (str): The filename to parse.

    Returns:
        datetime.datetime: The parsed timestamp if successful, otherwise None.
    """
    filename = os.path.basename(filename)
    try:
        date_time_str = filename.split('.')[0]
        return datetime.datetime.strptime(date_time_str, '%Y-%m-%d_%H-%M-%S')
    except ValueError:
        return None

def process_image(args):
    """
    Processes a single image to calculate the average brightness around specified tube locations.

    Args:
        args (tuple): A tuple containing:
            file_path (str): Path to the image file.
            tube_locations (list): List of dictionaries containing 'x' and 'y' coordinates of tube locations.
            zero_t_timestamp (datetime.datetime): Timestamp to filter images taken before this time.
            use_filename_timestamp (bool): Whether to use the filename timestamp or file modification time.

    Returns:
        dict: A dictionary with timestamps as keys and nested dictionaries as values.
              Each nested dictionary contains tube indices as keys and average brightness values as values.
              Returns None if the timestamp is invalid or before the zero_t_timestamp.
    """
    file_path, tube_locations, zero_t_timestamp, use_filename_timestamp = args
    
    if use_filename_timestamp:
        filename = os.path.basename(file_path)
        timestamp = parse_timestamp_from_filename(filename)
        if timestamp is None:
            return None
    else:
        # Assuming the file modification time is also in UTC+8
        timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    
    if timestamp <= zero_t_timestamp:
        return None

    image = Image.open(file_path).convert('L')
    image_array = np.array(image)
    
    height, width = image_array.shape

    result = {timestamp: {}}
    for i, location in enumerate(tube_locations):
        x, y = int(location['x']), int(location['y'])
        box_size = 10
        
        y_start = max(0, y - box_size)
        y_end = min(height, y + box_size + 1)
        x_start = max(0, x - box_size)
        x_end = min(width, x + box_size + 1)
        
        box = image_array[y_start:y_end, x_start:x_end]
        average_brightness = np.mean(box)
        result[timestamp][i] = average_brightness

    return result

def load_brightness_timeseries(image_directory, tube_location, temperature_recordings, use_filename_timestamp=True, log_callback=None):
    """
    Loads and processes brightness timeseries data from image files.

    Args:
        image_directory (str): Directory containing the image files.
        tube_location (str): Path to the file containing tube locations.
        temperature_recordings (pd.DataFrame): DataFrame containing temperature recordings.
        use_filename_timestamp (bool): Whether to use the filename timestamp or file modification time.
        log_callback (function): Optional callback function for logging messages.

    Returns:
        dict: A dictionary with 'timestamp' as a key and an array of datetime objects as values.
              Other keys are tube indices, each containing an array of average brightness values.
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
            
    with open(tube_location, 'rb') as f:
        tube_locations = pickle.load(f)

    zero_t_index = (temperature_recordings['temperature'] < 0).idxmax()
    zero_t_timestamp = temperature_recordings['timestamp'].iloc[zero_t_index]
    print(zero_t_timestamp)

    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    args_list = [(os.path.join(image_directory, image_file), tube_locations, zero_t_timestamp, use_filename_timestamp) 
                 for image_file in image_files]

    total_images = len(args_list)
    log(f"Starting to process {total_images} images")

    second_brightness = defaultdict(lambda: defaultdict(list))
    with mp.Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(process_image, args_list, chunksize=10)):
            if i % 100 == 0:
                progress = int((i / total_images) * 95)
                log(f"Processed {i} / {total_images} images ({progress}%)")
            
            if result is not None:
                for timestamp, brightness_dict in result.items():
                    for tube_id, brightness in brightness_dict.items():
                        second_brightness[timestamp][tube_id].append(brightness)

    log("Image processing complete. Calculating average brightness...")

    brightness_timeseries = {'timestamp': []}
    for i in range(len(tube_locations)):
        brightness_timeseries[i] = []

    for timestamp in sorted(second_brightness.keys()):
        brightness_timeseries['timestamp'].append(timestamp)
        for i in range(len(tube_locations)):
            avg_brightness = np.mean(second_brightness[timestamp][i])
            brightness_timeseries[i].append(avg_brightness)

    brightness_timeseries['timestamp'] = np.array(brightness_timeseries['timestamp'])
    for i in range(len(tube_locations)):
        brightness_timeseries[i] = np.array(brightness_timeseries[i])

    brightness_timeseries['timestamp'] = pd.to_datetime(brightness_timeseries['timestamp'])

    log("Brightness time series calculation complete")
    return brightness_timeseries

def get_freezing_temperature(temperature_recordings, brightness_timeseries):
    """
    Determines the freezing temperature for each tube based on the brightness timeseries.

    Args:
        temperature_recordings (pd.DataFrame): DataFrame containing temperature recordings with 'timestamp' and 'temperature' columns.
        brightness_timeseries (dict): Dictionary with 'timestamp' as a key and an array of datetime objects as values.
                                      Other keys are tube indices, each containing an array of average brightness values.

    Returns:
        dict: A dictionary with tube indices as keys and nested dictionaries as values.
              Each nested dictionary contains 'temperature' and 'timestamp' keys.
              'temperature' is the freezing temperature, and 'timestamp' is the corresponding timestamp.
              If no freezing point is found, 'temperature' and 'timestamp' are set to None.
    """
    results = {}
    
    # Convert datetime objects to Unix timestamps if necessary
    temp_timestamps = temperature_recordings['timestamp']
    
    # Ensure brightness timestamps are in Unix timestamp format
    bright_timestamps = brightness_timeseries['timestamp']
    
    common_timestamps = np.intersect1d(temp_timestamps, bright_timestamps)
    temp_indices = np.searchsorted(temp_timestamps, common_timestamps)
    bright_indices = np.searchsorted(bright_timestamps, common_timestamps)
    
    for tube_index in range(len(brightness_timeseries) - 1):  # Exclude 'timestamp' key
        tube_brightness = brightness_timeseries[tube_index][bright_indices]
        
        # Calculate the derivative (rate of change) of brightness
        brightness_derivative = np.diff(tube_brightness)
        
        if len(brightness_derivative) > 0:
            # Find the index of the largest absolute decrease
            freezing_index = np.argmin(brightness_derivative)
            
            # Get the timestamp and temperature at the freezing point
            freezing_timestamp = common_timestamps[freezing_index]
            freezing_temperature = np.interp(common_timestamps[freezing_index], 
                                             temp_timestamps, 
                                             temperature_recordings['temperature'])
            
            results[tube_index] = {
                'temperature': freezing_temperature,
                'timestamp': freezing_timestamp
            }
        else:
            # Handle the case where there's not enough data to determine a freezing point
            results[tube_index] = {
                'temperature': None,
                'timestamp': None
            }
    
    return results
