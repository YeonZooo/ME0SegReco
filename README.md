# ME0SegReco

ME0 Segment Reconstruction using Deep Learning

## Overview

ME0SegReco is a deep learning framework for ME0 (Muon Endcap 0) segment reconstruction in particle physics experiments. This project provides implementations of multiple neural network architectures (CNN and Transformer) for hit-level segment reconstruction tasks, built on PyTorch Lightning.

## Features

- **Multiple Model Architectures**:
  - 3D Convolutional Neural Network (ME0SegCNN3d)
  - Transformer Encoder-based model (ME0Transformer)
- **PyTorch Lightning Integration**: Simplified training, validation, and testing workflows
- **Flexible Configuration**: YAML-based configuration system using jsonargparse
- **Comprehensive Metrics**: Hit-level, segment-level
- **Data Processing**: HDF5-based dataset handling with efficient batching
- **Post-processing**: Clustering algorithms for segment reconstruction

## Installation

### Prerequisites

- [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or Conda
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YeonZooo/ME0SegReco.git
cd ME0SegReco
```

2. Create and activate the conda environment:
```bash
micromamba env create -f environment.yaml
micromamba activate ME0SegReco-py311
```

> **Note**: The `environment.yaml` file is configured for the University of Seoul GPU cluster. Users must modify dependency versions (e.g., PyTorch, CUDA) to ensure compatibility with their local environments.

3. Set up the Python path:
```bash
source setup.sh
```

## Usage

### Quick Start

Run a sanity check with a specific configuration:
```bash
./sanity-check.sh cnn        # For CNN model
./sanity-check.sh transformer # For Transformer model
```

### Training

Train a model using the main training script:
```bash
python train.py --config config/cnn.yaml
```

Or use the provided run script:
```bash
./run.sh train.py <log_dir> [checkpoint_path]
```

### Configuration

The project uses YAML configuration files located in the `config/` directory:
- `cnn.yaml`: Configuration for the 3D CNN model
- `transformer.yaml`: Configuration for the Transformer model

Key configuration sections:
- `seed_everything`: Random seed for reproducibility
- `model`: Model architecture and hyperparameters
- `data`: Dataset paths and preprocessing options
- `trainer`: PyTorch Lightning trainer settings (epochs, accelerator, devices, etc.)

Example configuration structure:
```yaml
seed_everything: 1234
model:
  class_path: me0.modules.cnn.modelmodule.ModelModule
  init_args:
    model:
      class_path: me0.modules.cnn.model.ME0SegCNN3d
      init_args:
        in_channels: 3
        out_channels: 1
        hidden_channels_list: [2, 4, 2]
        kernel_size: [3, 3, 7]
        activation: GELU
    criterion:
      class_path: me0.losses.ME0BCELoss
      init_args:
        pos_weight: 72.15
        reduction: mean
    optimizer_init:
      class_path: torch.optim.AdamW
      init_args:
        lr: 3.0e-4
```

### Jupyter Notebooks

Explore the project through the example notebooks in the `examples/` directory:
- `00.dataset.ipynb`: Dataset exploration and visualization
- `01.ME0SegCNN3d.ipynb`: CNN model examples
- `02.ME0Transformer.ipynb`: Transformer model examples
- `03.Metrics.ipynb`: Metrics computation and analysis

## Project Structure

```
ME0SegReco/
├── config/                 # Configuration files
│   ├── cnn.yaml
│   └── transformer.yaml
├── data/                   # Data directory (HDF5 files)
├── examples/               # Jupyter notebooks
├── src/me0/               # Source code
│   ├── data/              # Dataset implementations
│   ├── lightning/         # PyTorch Lightning modules
│   ├── losses.py          # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── modules/           # Model architectures
│   │   ├── cnn/          # CNN models
│   │   └── transformer/  # Transformer models
│   ├── postprocessing/    # Clustering and post-processing
│   └── utils/             # Utility functions
├── train.py               # Main training script
├── environment.yaml       # Conda environment specification
└── README.md             # This file
```

## Model Architectures

### 3D CNN (ME0SegCNN3d)
- Processes 3D spatial representations of detector hits
- Multiple convolutional layers with batch normalization
- Configurable channel dimensions and kernel sizes
- GELU activation functions

### Transformer (ME0Transformer)
- Transformer encoder architecture for sequence modeling
- Multi-head self-attention mechanism
- Configurable number of layers, attention heads, and feedforward dimensions
- Supports input padding masks for variable-length sequences

## Metrics

The framework provides comprehensive evaluation metrics:
- **Hit-level metrics**: Precision, recall, F1-score for individual hits
- **Segment-level metrics**: Segment reconstruction quality
- **Segment metrics for graphs**: torchmetrics to create performance graphs

## Dataset Generation

To generate datasets for training and evaluation, use the dataset generation code available at:
https://gitlab.cern.ch/yeonju/me0segreco

The repository includes comprehensive documentation for dataset generation. Please refer to its README for detailed instructions.

## Data Format

The project uses HDF5 format for data storage. Input data is expected to have:
- Hit positions and features
- Ground truth segment labels
- Event-level metadata

## Development

### Environment
- Python 3.11
- PyTorch 2.6.0
- PyTorch Lightning 2.5.1
- PyTorch Geometric 2.6.1

### Key Dependencies
- `h5py`: HDF5 data handling
- `tensordict`: Efficient tensor manipulation
- `einops`: Tensor operations
- `mplhep`: HEP-style plotting
- `scienceplots`: Scientific publication-quality plots

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the GitHub repository.
