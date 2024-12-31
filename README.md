# Line Drawing Path Extractor

A deep learning model for extracting skeleton lines from images using query points.

## Model Architecture

The PathExtractionNet is designed to extract a single skeleton line given an input image and a query point. It uses a combination of:

- Dilated residual blocks for multi-scale feature extraction
- Self-attention mechanism for global context
- Skip connections for preserving spatial information
- Gradual channel reduction for final prediction

### Input
- Grayscale image: [B, 1, H, W]
- Query point: [B, 2] containing (y,x) coordinates

### Output
- Probability map: [B, 1, H, W] indicating the likelihood of each pixel being part of the skeleton line

## Training Data

The models are trained on procedurally generated data with the following characteristics:
- Random line drawings with varying thickness
- Controlled line complexity (number of points, curves)
- Generated using line_drawing_dataset.py with customizable parameters

## Installation
`pip install -r requirements.txt`

## Usage

### Basic Training
`python -m src.main`

### Custom Training
The training process can be customized with various command line arguments:
- --batch_size: Number of samples per batch
- --epochs: Number of training epochs
- --learning_rate: Initial learning rate

## Requirements
- PyTorch (with CUDA support recommended)
- NumPy
- Pillow
- Matplotlib
- tqdm

## Project Structure
- src/
  - data/: Dataset generation and loading
  - losses/: Custom loss functions (F1 and repair losses)
  - models/: Neural network architecture
  - main.py: Training and evaluation scripts
- models/: Pretrained model weights
- requirements.txt: Project dependencies
- setup.py: Package installation configuration

## Model Performance
The PathExtractionNet achieves:
- High accuracy in skeleton line extraction
- Robust performance with varying line thicknesses
- Real-time inference capability
- Effective handling of complex line intersections and curves

## License
This project is licensed under the included LICENSE file.

