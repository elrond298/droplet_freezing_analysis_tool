import pandas as pd
import numpy as np
import os
import datetime
import numpy as np
from PIL import Image
import pickle

from collections import defaultdict
import multiprocessing as mp

from scipy.signal import find_peaks


def load_temperature_timeseries(temperature_recordings):
    """
    temperature_recordings: string, file of temperature storage
    
    return:
    dictionary
        timestamp: time
        temperature: average temperature
    """
    df = pd.read_excel(temperature_recordings)
    def to_timestamp(lst):
        bias = 0
        for i in range(len(lst)):
            lst[i] = datetime.datetime.fromisoformat(lst[i])
            lst[i] = datetime.datetime.timestamp(lst[i]) - bias
        return lst

    df["时间"] = to_timestamp(df["时间"].to_string(index=False).split("\n"))

    temperature_recordings = {}
    temperature_recordings["timestamp"] = df["时间"].to_numpy()
    if df["通道01(℃)"].min() < 0:
        temperature_recordings["temperature"] = (df["通道01(℃)"] + df["通道02(℃)"] + df["通道03(℃)"] + df["通道04(℃)"]) / 4
    else:
        temperature_recordings["temperature"] = (df["通道05(℃)"] + df["通道02(℃)"] + df["通道03(℃)"] + df["通道06(℃)"]) / 4
    
    temperature_recordings["temperature"] = temperature_recordings["temperature"].to_numpy()
    return temperature_recordings # a 1D array, contains the temperature recording


def process_image(args):
    file_path, tube_locations, zero_t_timestamp = args
    timestamp = os.path.getmtime(file_path)
    if timestamp <= zero_t_timestamp:
        return None

    rounded_timestamp = int(timestamp)
    image = Image.open(file_path).convert('L')
    image_array = np.array(image)
    
    height, width = image_array.shape

    result = {rounded_timestamp: {}}
    for i, location in enumerate(tube_locations):
        x, y = int(location['x']), int(location['y'])
        box_size = 3
        
        # Ensure box boundaries are within image dimensions
        y_start = max(0, y - box_size)
        y_end = min(height, y + box_size + 1)
        x_start = max(0, x - box_size)
        x_end = min(width, x + box_size + 1)
        
        box = image_array[y_start:y_end, x_start:x_end]
        average_brightness = np.mean(box)
        result[rounded_timestamp][i] = average_brightness

    return result

def load_brightness_timeseries(image_directory, tube_location, temperature_recordings, log_callback=None):
    """
    image_directory: string
    tube_location: list of dictionary, {'x': float, 'y': float}, pickle saved data
    temperature_recordings: dictionary with 'timestamp' and 'temperature' numpy arrays
    log_callback: function to handle logging messages
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
            
    # Load tube locations
    with open(tube_location, 'rb') as f:
        tube_locations = pickle.load(f)

    zero_t_timestamp = temperature_recordings['timestamp'][np.argmax(temperature_recordings['temperature'] < 0)]

    # Walkthrough images in image_directory, identified by file suffix: png, jpg
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    # Prepare arguments for multiprocessing
    args_list = [(os.path.join(image_directory, image_file), tube_locations, zero_t_timestamp) 
                 for image_file in image_files]

    # Use multiprocessing to process images
    total_images = len(args_list)
    log(f"Starting to process {total_images} images")

    second_brightness = defaultdict(lambda: defaultdict(list))
    with mp.Pool() as pool:
        for i, result in enumerate(pool.imap_unordered(process_image, args_list, chunksize=10)):
            if i % 100 == 0:  # Increase frequency of updates
                progress = int((i / total_images) * 95)  # Leave 5% for final processing
                log(f"Processed {i} / {total_images} images ({progress}%)")
            
            if result is not None:
                for timestamp, brightness_dict in result.items():
                    for tube_id, brightness in brightness_dict.items():
                        second_brightness[timestamp][tube_id].append(brightness)

    log("Image processing complete. Calculating average brightness...")

    # Calculate average brightness for each second
    brightness_timeseries = {'timestamp': []}
    for i in range(len(tube_locations)):
        brightness_timeseries[i] = []

    for timestamp in sorted(second_brightness.keys()):
        brightness_timeseries['timestamp'].append(timestamp)
        for i in range(len(tube_locations)):
            avg_brightness = np.mean(second_brightness[timestamp][i])
            brightness_timeseries[i].append(avg_brightness)

    # Convert lists to numpy arrays for easier manipulation
    brightness_timeseries['timestamp'] = np.array(brightness_timeseries['timestamp'])
    for i in range(len(tube_locations)):
        brightness_timeseries[i] = np.array(brightness_timeseries[i])

    log("Brightness time series calculation complete")
    return brightness_timeseries



def get_freezing_temperature(temperature_recordings, brightness_timeseries):
    """
    temperature_recordings: dictionary with 'timestamp' and 'temperature' numpy arrays
    brightness_timeseries:
        timestamp: numpy array of timestamps
        tube_index: a array of brightness for each tube
    
    Returns:
    A dictionary with tube indices as keys and dictionaries containing 'temperature' and 'timestamp' as values
    """
    results = {}
    
    # Ensure timestamps are aligned
    common_timestamps = np.intersect1d(temperature_recordings['timestamp'], brightness_timeseries['timestamp'])
    temp_indices = np.searchsorted(temperature_recordings['timestamp'], common_timestamps)
    bright_indices = np.searchsorted(brightness_timeseries['timestamp'], common_timestamps)
    
    for tube_index in range(len(brightness_timeseries) - 1):  # Exclude 'timestamp' key
        tube_brightness = brightness_timeseries[tube_index][bright_indices]
        
        # Calculate the derivative (rate of change) of brightness
        brightness_derivative = np.diff(tube_brightness)
        
        # Find the index of the largest absolute decrease
        freezing_index = np.argmin(brightness_derivative)
        
        # Get the timestamp and temperature at the freezing point
        freezing_timestamp = common_timestamps[freezing_index]
        freezing_temperature = np.interp(freezing_timestamp, 
                                         temperature_recordings['timestamp'], 
                                         temperature_recordings['temperature'])
        
        results[tube_index] = {
            'temperature': freezing_temperature,
            'timestamp': freezing_timestamp
        }
    
    return results