
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
