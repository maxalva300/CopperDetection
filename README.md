# CopperImageWeb

Local Dash app for optical copper particle detection in Eddy Current Separation images.

## What it does

- Upload one JPG/PNG image.
- Detect copper particles using HSV + RGB-ratio criteria.
- Highlight only detected copper in red.
- Keep rejects / non-copper material unchanged.
- Show basic metrics and allow downloads of the overlay and Excel result.

## Folder structure

```text
CopperImageWeb/
├── app.py
├── copper_core.py
├── config.py
├── assets/
│   ├── style.css
│   ├── MetsoLogo.png
│   └── Background.png
├── data/
│   └── calibrated_copper_model.json
├── uploads/
├── results/
│   ├── overlays/
│   └── reports/
├── requirements.txt
├── runtime.txt
└── Procfile
```

## Installation with Anaconda Prompt

```bash
conda create -n copperweb python=3.11 -y
conda activate copperweb
conda install -c conda-forge dash dash-bootstrap-components pandas numpy pillow openpyxl plotly opencv -y
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
cd "C:\Users\sagit\OneDrive - University of Oulu and Oamk\Project Study 2026\Code 21-5\CopperImageWeb"
python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

## Calibration model

The app reads:

```text
data/calibrated_copper_model.json
```

For now, this file contains temporary default parameters. Once the hand-sort calibration finishes, replace it with the final calibrated JSON model.
