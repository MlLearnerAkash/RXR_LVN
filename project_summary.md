# RxR Heatmap Generation Project Summary

## Project Overview

This project implements a segmentation-like heatmap generation approach for the RxR dataset, highlighting "appealing" and "repelling" areas for visual language navigation. The implementation is inspired by CLIP and VLMaps, integrating attention-based mechanisms with pose trace analysis and ground truth path information.

## Key Components

### 1. Multi-Head Attention Localization (MHAL)

The core of the approach leverages recent findings that specific attention heads in vision-language models can effectively localize objects mentioned in text descriptions. The implementation:

- Identifies specific attention heads that consistently focus on navigation-relevant regions
- Extracts and combines attention maps from these localization heads
- Aligns attention with instruction semantics and visual features

### 2. Pose Trace-Based Temporal Grounding (PTTG)

Utilizes RxR's unique pose trace annotations to:

- Convert annotator gaze patterns into spatial density maps
- Map specific instruction phrases to corresponding visual regions
- Analyze field-of-view to identify regions that receive consistent visual attention

### 3. Ground Truth Path Segmentation (GTPS)

Creates objective segmentation maps based on:

- Successful navigation paths (higher pixel values)
- Areas with navigation errors or obstacles (lower pixel values)
- Success metrics like Navigation Error and Dynamic Time Warping

## Implementation Details

The project is structured as follows:

```
rxr_research/
├── src/
│   ├── data/
│   │   ├── rxr_dataset.py        # RxR dataset loading and preprocessing
│   │   └── data_utils.py         # Utility functions for data handling
│   ├── models/
│   │   ├── clip_adapter.py       # CLIP model integration
│   │   ├── attention_model.py    # Multi-head attention localization
│   │   └── heatmap_generator.py  # Heatmap generation module
│   ├── utils/
│   │   ├── visualization.py      # Visualization utilities
│   │   └── evaluation.py         # Evaluation metrics
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── test.py                   # Testing script
│   └── setup.py                  # Project setup utilities
├── config/
│   ├── default.yaml              # Default configuration
│   └── paths.yaml                # Path configuration
└── requirements.txt              # Project dependencies
```

### Key Features

1. **CLIP Integration**: Leverages CLIP's powerful vision-language representations to extract features from both panoramic images and natural language instructions.

2. **Attention-Based Localization**: Implements the approach from "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding" to identify specific attention heads that excel at localizing navigation-relevant regions.

3. **VLMaps-Inspired Spatial Mapping**: Draws inspiration from VLMaps to create spatially-anchored visual language features that enable natural language indexing in the navigation environment.

4. **Comprehensive Evaluation**: Includes metrics for heatmap quality assessment, including IoU, Average Precision, and structural similarity.

5. **Visualization Tools**: Provides utilities for visualizing heatmaps, attention maps, and comparison between predicted and ground truth segmentations.

## Usage Instructions

1. **Setup**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd rxr_research
   
   # Create and activate a virtual environment
   conda create -n rxr_heatmap python=3.8 -y
   conda activate rxr_heatmap
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up project
   python src/setup.py --all
   ```

2. **Configure**:
   - Edit `config/paths.yaml` to set the path to your RxR dataset
   - Adjust model parameters in `config/default.yaml` if needed

3. **Train**:
   ```bash
   python src/train.py --data_root /path/to/rxr_dataset --output_dir outputs
   ```

4. **Evaluate**:
   ```bash
   python src/evaluate.py --checkpoint outputs/checkpoints/checkpoint_best.pth --split val_unseen
   ```

5. **Test**:
   ```bash
   python src/test.py --checkpoint outputs/checkpoints/checkpoint_best.pth --save_visualizations
   ```

## Future Directions

1. **Improved Temporal Modeling**: Enhance the temporal consistency of heatmaps across sequential viewpoints.

2. **Cross-Lingual Analysis**: Leverage RxR's multilingual nature to compare navigation patterns across different languages.

3. **Integration with Navigation Agents**: Use the generated heatmaps to guide navigation decision-making in embodied AI agents.

4. **Fine-Grained Instruction Grounding**: Develop more precise word-level grounding of instructions to specific spatial regions.

5. **Real-Time Processing**: Optimize the pipeline for real-time heatmap generation during navigation.

## Conclusion

This project demonstrates a novel approach to generating segmentation-like heatmaps for visual language navigation using the RxR dataset. By integrating attention mechanisms, pose trace analysis, and ground truth path information, the system provides valuable insights into the relationship between natural language instructions and navigation decisions, potentially advancing the state-of-the-art in Visual Language Navigation and contributing to more interpretable and effective embodied AI systems.
