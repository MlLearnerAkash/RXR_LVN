# RxR Heatmap Generation Project

This project implements a segmentation-like heatmap generation approach for the RxR dataset, highlighting "appealing" and "repelling" areas for visual language navigation. The implementation is inspired by CLIP and VLMaps.

## Project Structure

```
rxr_research/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── rxr_dataset.py        # RxR dataset loading and preprocessing
│   │   └── data_utils.py         # Utility functions for data handling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_adapter.py       # CLIP model integration
│   │   ├── attention_model.py    # Multi-head attention localization
│   │   └── heatmap_generator.py  # Heatmap generation module
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py      # Visualization utilities
│   │   └── evaluation.py         # Evaluation metrics
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── test.py                   # Testing script
├── config/
│   ├── default.yaml              # Default configuration
│   └── paths.yaml                # Path configuration
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

```bash
# Create and activate a virtual environment
conda create -n rxr_heatmap python=3.8 -y
conda activate rxr_heatmap

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Download and prepare the RxR dataset
2. Configure paths in `config/paths.yaml`
3. Train the model: `python src/train.py`
4. Evaluate the model: `python src/evaluate.py`
5. Test on test set: `python src/test.py`
# RXR_LVN
