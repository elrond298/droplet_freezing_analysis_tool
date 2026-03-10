import io

from PyQt6.QtCore import QObject, pyqtSignal

from freezing_detection import load_brightness_timeseries, load_temperature_timeseries


class StreamToTextEdit(io.StringIO):
    def __init__(self, signal, tab_number, level):
        super().__init__()
        self.signal = signal
        self.tab_number = tab_number
        self.level = level

    def write(self, text):
        normalized_text = text.rstrip()
        if normalized_text:
            self.signal.emit(normalized_text, self.tab_number, self.level)


class BrightnessWorker(QObject):
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, image_directory, tube_location_file, temperature_recording_file):
        super().__init__()
        self.image_directory = image_directory
        self.tube_location_file = tube_location_file
        self.temperature_recording_file = temperature_recording_file

    def run(self):
        temperature_recordings = load_temperature_timeseries(self.temperature_recording_file)
        self.progress.emit(5)

        brightness_timeseries = load_brightness_timeseries(
            self.image_directory,
            self.tube_location_file,
            temperature_recordings,
            progress_callback=lambda value: self.progress.emit(value),
            log_callback=lambda message: self.log.emit(message),
        )

        self.progress.emit(100)
        self.finished.emit(temperature_recordings, brightness_timeseries)