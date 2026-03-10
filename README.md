
# Overview

This software is intended to simplify the process of droplet freezing experiment freezing temperature detection.

## Installation

Follow one of these installation method below to create the environment.

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
