
# CircleDetection

CircleDetection is a desktop tool for droplet-freezing assay analysis. It provides a GUI workflow for preparing images, locating PCR tubes, extracting brightness timeseries, and reviewing freezing temperatures tube by tube.

## Quick Start

If you already use `uv`, the shortest setup is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.14
uv sync
uv run python gui.py
```

If `uv` is already installed, you only need the last three commands.

## Installation

Choose one environment setup method.

### uv

Use the dependencies defined in `pyproject.toml`:

```bash
uv python install 3.14
uv sync
uv run python gui.py
```

### Conda

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate inp
python3 gui.py
```

### Virtual Environment (venv)

Create a virtual environment and install from `requirements.txt`:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python3 gui.py
```

## GUI Workflow

Launch the application with:

```python
python3 gui.py
```

Then follow the workflow in the GUI or the additional usage notes in `usage/使用说明.pdf`.

### Main Tabs

- `Prepare Image`: load a source image, rotate it if needed, and crop the region that should be used for tube detection.
- `Locate Tubes`: detect PCR tubes, adjust detection parameters such as minimum area and circularity, manually review inner circles, and save the tube locations.
- `Analyze Freezing`: load the image directory, temperature recording, and saved tube locations, run the analysis, inspect each tube, and save or reload freezing-temperature results.
- `Settings`: adjust UI font size and review keyboard shortcuts.

### GUI Features

- Selection paths are cached between sessions in `.gui_selection_cache.json`.
- Each workflow tab includes its own log panel for progress updates and errors.
- Inner-circle positions can be exported and reused in the analysis step.
- Freezing-temperature results can be saved and loaded later.

## GUI Module Structure

The GUI code is split across `gui*.py` files by responsibility so that UI layout, state, workflow logic, and helpers can evolve separately.

```text
gui.py
gui_state.py
gui_tabs.py
gui_services.py
gui_workers.py
gui_detection_controller.py
gui_analysis_controller.py
gui_image_controller.py
gui_logging.py
gui_selection_cache.py
```

### File Responsibilities

- `gui.py`: main application entry point and `InteractivePlot` window. It wires together the tabs, state containers, controllers, logging, cache restore/save, keyboard shortcuts, and Qt signal plumbing.
- `gui_state.py`: dataclasses for persistent window state, split into selection state, image-preparation state, tube-detection state, and analysis state.
- `gui_tabs.py`: builds the four main tabs and their widgets. This file should stay focused on layout creation and signal connections rather than analysis logic.
- `gui_services.py`: pure helper functions used by the GUI, including tube detection orchestration, overlay rendering, image rotation/cropping helpers, coordinate restoration, and freezing-temperature serialization/deserialization.
- `gui_workers.py`: background and stream utilities. It contains the brightness-loading worker that runs in a `QThread` and the stream adapter used to push console output into GUI log panes.
- `gui_detection_controller.py`: handlers for the `Locate Tubes` tab, including running detection, responding to plot clicks, redrawing manual edits, and saving inner-circle locations.
- `gui_analysis_controller.py`: handlers for the `Analyze Freezing` tab, including loading saved tube locations, starting brightness extraction, applying analysis results, reviewing each tube, updating the selected freezing point, and importing/exporting freezing results.
- `gui_image_controller.py`: handlers for the `Prepare Image` tab, including loading the selected image, rotating it, restoring the original view, and applying the selected crop to downstream tube detection.
- `gui_logging.py`: shared log formatting and routing helpers. It chooses the correct log widget for a tab, formats timestamped log messages, and supports broadcasting messages to all GUI log panes.
- `gui_selection_cache.py`: shared helpers for `.gui_selection_cache.json`, including restoring cached input paths, refreshing labels, and persisting the last-used files and folders.

### Dependency Direction

- `gui.py` is the coordinator and imports the other `gui*.py` modules.
- `gui_tabs.py` depends on the window methods exposed by `gui.py`, but it should not implement business logic itself.
- The controller modules call into `gui_services.py` and update the `InteractivePlot` window state.
- `gui_workers.py`, `gui_logging.py`, and `gui_selection_cache.py` provide cross-cutting support used by `gui.py` and the controllers.
- `gui_state.py` stays data-only and should not depend on Qt widgets or controller code.

### Saved Inner-Circle Coordinate Fix

Earlier versions of the GUI could save incorrect `y` coordinates for inner circles when the preview image had been rotated before export. The symptom was that saved circles looked mirrored on the `y` axis when reloaded or visualized with `test.py`.

Cause:

- The GUI detected circles in rotated preview coordinates.
- During export, it used hand-written trigonometric inverse-rotation logic to map those coordinates back to the original image.
- That manual inverse did not match OpenCV's actual image-space affine transform, so `y` values could be mirrored in saved files.

Fix:

- The GUI now restores exported circle coordinates by inverting the exact OpenCV affine matrix created for the preview rotation.
- Crop offsets are applied before the inverse transform so saved circles end up in original-image coordinates.
- Loaded circle records are also normalized to integer image coordinates before analysis.

Impact on existing files:

- Inner-circle location files saved before this fix may still contain incorrect `y` coordinates.
- Re-export those files from the updated GUI, or repair them separately, before using them for freezing analysis.

### Keyboard Shortcuts

- `Ctrl+1` to `Ctrl+4`: switch between the main tabs.
- `Ctrl+,`: open the Settings tab.
- `Ctrl+=` or `Ctrl++`: increase font size.
- `Ctrl+-`: decrease font size.
- `Ctrl+0`: reset font size.

## Qt and Platform Notes

The GUI targets PyQt6. If you are updating an existing virtual environment, reinstall the Qt dependency with:

```bash
pip install --upgrade PyQt6
```

## Linux Qt Troubleshooting

If the GUI fails with an `xcb` platform plugin error such as `libxcb-cursor.so.0 => not found`, install the missing system package:

```bash
sudo apt update
sudo apt install libxcb-cursor0
```

On Debian and Ubuntu systems, this is required by Qt 6.5+ for the `xcb` platform plugin.
