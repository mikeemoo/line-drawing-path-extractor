# Line Drawing Path Extractor

A deep learning technique for extracting clean paths from line drawings, featuring two complementary models:

1. Skeletonization Model: Extracts clean, single-pixel-wide skeleton lines from input drawings
2. Repair Model: Fixes damaged or incomplete skeleton lines

## Models

### Skeletonization Model
- **Input**: Grayscale line drawings with varying line thickness
- **Output**: Single-pixel-wide skeleton lines
- **Architecture**: Enhanced deep neural network with:
  - Encoder path with dilated convolutions for larger receptive field
  - Decoder path with skip connections for precise localization
  - Residual blocks for better gradient flow
  - Final sigmoid activation for binary output

### Repair Model
- **Input**: Damaged skeleton lines (with gaps, overdraw, or missing segments)
- **Output**: Clean, continuous skeleton lines
- **Architecture**: Specialized U-Net style network with:
  - Dilated convolutions in middle layers for context awareness
  - Skip connections to preserve fine details
  - Focus on local and global line connectivity

## Training Data

The models are trained on procedurally generated data with the following characteristics:
- Random line drawings with varying thickness
- Controlled line complexity (number of points, curves)
- Simulated damage for repair model training:
  - Random pixel removal
  - Line thickness variations
  - Overdraw effects

## Installation
`pip install -r requirements.txt`

## Usage

### Basic Training
`python -m src.main --wandb_mode disabled`

## Requirements
- PyTorch (with CUDA support recommended)
- NumPy
- Pillow
- Matplotlib
- tqdm
- Weights & Biases (optional, for experiment tracking)

## Project Structure

```
src/
├── data/
│ └── line_drawing_dataset.py # Dataset and data loading
├── models/
│ ├── skeletonization.py # Main skeletonization model
│ └── skeleton_repair_net.py # Repair model for fixing damaged lines
├── losses/
│ ├── f1_loss.py # F1 score based loss function
│ └── repair_loss.py # Specialized loss for repair model
└── utils/
├── data_generation.py # Procedural training data generation
├── training.py # Training loop and utilities
└── visualization.py # Visualization tools
```

## Model Performance
- Skeletonization model achieves clean path extraction with:
  - High precision in line placement
  - Consistent single-pixel width
  - Preservation of line connectivity
- Repair model successfully handles:
  - Gap filling in broken lines
  - Removal of overdraw artifacts
  - Recovery of missing line segments
