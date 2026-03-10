
# Overview

This software is intended to simplify the process of droplet freezing experiment freezing temperature detection.

## Installation

Follow one of these installation method below to create the environment.

### Using uv

If you use `uv`, install `uv`, create the project environment from `pyproject.toml`, and run the GUI with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.14
uv sync
uv run python gui.py
```

If `uv` is already installed, you only need the last three commands.

### Using Conda

To install the required packages using Conda, you can create a new environment from the `environment.yml` file. Run the following commands:

```bash
conda env create -f environment.yml
conda activate inp
```

### Using Virtual Environment (venv)

To install the required packages using a virtual environment, you can create a new virtual environment and install the dependencies from the `requirements.txt` file. Run the following commands:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Usage

start the gui by:

``` python
python3 gui.py
```

Then follow the instructions in `usage/使用说明.pdf` to complete the recognition.

## GUI Overview

The desktop GUI provides a complete workflow for droplet-freezing analysis:

- `Prepare Image`: load a source image, rotate it if needed, and crop the useful region before tube detection.
- `Locate Tubes`: detect PCR tubes, tune detection settings such as minimum area and circularity, manually review inner circles, and save the detected inner-circle positions.
- `Analyze Freezing`: load the image folder, temperature recording, and saved tube locations, run the timeseries analysis, review each tube one by one, and save or reload freezing-temperature results.
- `Settings`: adjust UI font size and review available keyboard shortcuts.

Additional GUI behavior:

- Selection paths are cached between sessions for the sample image, image directory, temperature file, and tube-location file.
- Each major workflow tab includes its own log panel for progress messages and errors.
- Keyboard shortcuts are available for tab switching and font-size changes, including `Ctrl+1` to `Ctrl+4`, `Ctrl+,`, `Ctrl+=`, `Ctrl+-`, and `Ctrl+0`.

## Qt Version

The GUI now targets PyQt6. If you are updating an existing virtual environment, reinstall the Qt dependency with:

```bash
pip install --upgrade PyQt6
```

## Linux Qt Troubleshooting

If the GUI fails with an `xcb` platform plugin error such as `libxcb-cursor.so.0 => not found`, install the missing system package:

```bash
sudo apt update
sudo apt install libxcb-cursor0
```

On Debian/Ubuntu systems, this is required by Qt 6.5+ for the `xcb` platform plugin.
