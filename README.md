
# Overview

This software is intended to simplify the process of droplet freezing experiment freezing temperature detection.

## Installation

### Using Conda

To install the required packages using Conda, you can create a new environment and install the dependencies from the `requirements.txt` file. Run the following commands:

```bash
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt
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
