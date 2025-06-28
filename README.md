# Orthogonal Gradient Descent Improves Neural Calibration

This repository contains the implementation and experiments for the OrthoGrad optimizer as explored in https://arxiv.org/abs/2506.04487. 

## Overview

OrthoGrad modifies standard gradient descent by orthogonalizing gradients with respect to current parameters:

```
g_orth = g - ((w·g)/(w·w + ε)) * w
```

where:
- `g` is the original gradient
- `w` are the current parameters 
- `ε` is a small constant for numerical stability (1e-30)

This implementation follows that of Prieto et al. in *Grokking at the Edge of Numerical Stability* and has been slightly modified to allow for the renormalization step to be skipped. Their repo can be found at https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/tree/main. 


## Key Features

- **Orthogonal Updates**: Gradients are projected to be orthogonal to current parameters
- **Gradient Renormalization**: Optional rescaling to preserve gradient magnitude (enabled by default)
- **Base Optimizer Agnostic**: Works with SGD, Adam, AdamW, etc.
- **Multi-Platform Support**: Supports CUDA, MPS (Apple Silicon), and CPU

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/OrthoGrad.git
cd OrthoGrad
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from utils.orthograd import OrthoGrad
import torch.optim as optim

# Create model
model = YourModel()

# Initialize OrthoGrad with SGD as base optimizer
optimizer = OrthoGrad(
    model.parameters(),
    base_optimizer_cls=optim.SGD,
    grad_renormalization=True,  # Enable gradient renormalization
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
```

### Run CIFAR-10 Experiments

Train ResNet18 on CIFAR-10 with both SGD and OrthoGrad:

```bash
python train.py --config configs/resnet18_cifar10.json
```

Or run with custom parameters:
```bash
python train.py --model resnet18 --optimizer both --epochs 100 --lr 0.01
```

For multiple seeds:
```bash
python train.py --config configs/resnet18_cifar10.json --seed 42
python train.py --config configs/resnet18_cifar10.json --seed 100
python train.py --config configs/resnet18_cifar10.json --seed 200
```

The training script will automatically save the best model checkpoints in the `outputs/` directory.

## Experimental Setup

The experiments in this repository were conducted on:
- **Hardware**: 2023 MacBook Pro with M3 Max chip
- **Framework**: PyTorch with MPS (Metal Performance Shaders) backend
- **Device**: Apple Silicon GPU acceleration via MPS

The training script automatically detects and uses the best available device (CUDA > MPS > CPU). On Apple Silicon Macs, it will use MPS for GPU acceleration.

## Results

Our experiments on CIFAR-10 show that OrthoGrad:
- Achieves competitive accuracy compared to standard SGD
- Demonstrates improved calibration properties
- Shows different optimization dynamics with orthogonal updates

## Repository Structure

```
OrthoGrad_Public_Repo/
│
├── README.md              # This file
├── LICENSE               # MIT License
├── requirements.txt      # Python dependencies
├── train.py             # Main training script
│
├── configs/             # Configuration files
│   ├── resnet18_cifar10.json
│   └── wrn28_cifar10.json
│
├── utils/              # Utility modules
│   ├── dataloaders.py  # Data loading utilities
│   ├── metrics.py      # Evaluation metrics
│   └── orthograd.py    # OrthoGrad optimizer
│
├── outputs/            # Training outputs
│   └── checkpoints/    # Model checkpoints
│
└── data/              # CIFAR-10 dataset (downloaded automatically)
```

**Note**: Model checkpoints are not included in this repository due to their large size. Run the training scripts to generate your own trained models, which will be saved in the `outputs/` directory.

## Citation

If you use this code in your research, please cite both the original OrthoGrad paper and our modified version:

```bibtex
@article{prieto2025grokking,
  title={Grokking at the Edge of Numerical Stability},
  author={Prieto, Lucas and Barsbey, Melih and Mediano, Pedro AM and Birdal, Tolga},
  journal={arXiv preprint arXiv:2501.04697},
  year={2025}
}

@article{orthograd-hedges-2025,
  title={Orthogonal Gradient Descent Improves Neural Calibration},
  author={C. Evans Hedges},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [evans.hedges@du.edu]. 
