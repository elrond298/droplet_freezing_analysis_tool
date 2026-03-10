from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectionState:
    sample_image_path: str | None = None
    image_directory: str | None = None
    temperature_recording_file: str | None = None
    tube_location_file: str | None = None
    ui_font_size: int = 10


@dataclass
class ImagePrepState:
    img: Any = None
    original_image: Any = None
    rotated_image: Any = None
    processed_image: Any = None
    crop_region: tuple[int, int, int, int] | None = None
    crop_selector: Any = None
    rotation_params: dict[str, Any] | None = None


@dataclass
class DetectionState:
    pcr_tubes: list[dict[str, Any]] = field(default_factory=list)
    inferred_tubes: list[dict[str, Any]] = field(default_factory=list)
    all_tubes: list[dict[str, Any]] = field(default_factory=list)
    inner_circles: list[dict[str, Any]] = field(default_factory=list)
    tubes_size: tuple[int, int] = (10, 8)


@dataclass
class AnalysisState:
    temperature_recordings: Any = None
    brightness_timeseries: Any = None
    freezing_temperatures: dict[int, dict[str, Any]] = field(default_factory=dict)
    num_tubes: int = 0
    current_tube: int = 0
    current_tube_temperature: Any = None
    current_tube_brightness: Any = None
    current_tube_timestamps: Any = None