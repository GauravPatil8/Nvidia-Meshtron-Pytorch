# MeshTron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale


> **Unofficial PyTorch Implementation** of ["MeshTron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale"](https://research.nvidia.com/labs/dir/meshtron) by NVIDIA Research

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2412.09548)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://research.nvidia.com/labs/dir/meshtron)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![MeshTron Banner](meshtron_architecture.png)

## Overview

MeshTron is a autoregressive mesh generation model capable of generating high-quality, artist-like meshes with up to **64K faces** at **1024-level coordinate resolution** – over an order of magnitude higher face count and 8× higher coordinate resolution than current state-of-the-art methods.
## Requirements
- Python 3.12
- PyTorch 2.8+
- CUDA 11.8+ (for GPU acceleration)
- **Operating System: Linux is required** due to the use of the **Flash Attention** library, which relies on platform-specific CUDA kernels.
- Additional dependencies listed in `requirements.txt`
## 1.Installation
### Clone the repository
```bash
git clone https://github.com/GauravPatil8/Nvidia-Meshtron-Pytorch.git
cd Nvidia-Meshtron-Pytorch
```

### Create conda environment
```bash
conda create -n meshtron python=3.12
conda activate meshtron
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## 2.Model Initialization
### To train on your own dataset with custom parameters, initialize the model components individually:
```python
import torch
from meshtron.model import Meshtron
from meshtron.encoder_conditioning import ConditioningEncoder
from meshtron.mesh_tokenizer import MeshTokenizer
```
### Initialize tokenizer
```python
tokenizer = MeshTokenizer(bins=128)
```
### Initialize conditioning encoder (Perceiver-based point cloud encoder)
```python
encoder = ConditioningEncoder(
    num_freq_bands = 4,
    depth = 4,
    max_freq = 6.0,
    input_channels = 6,
    input_axis = 1,
    num_latents = 256,
    latent_dim = 512,
    cross_heads = 8,
    latent_heads = 8,
    cross_dim_head = 32,
    latent_dim_head = 32,
    num_classes = 1,
    attn_dropout = 0.0,
    ff_dropout = 0.0,
    weight_tie_layers = 2,
    fourier_encode_data = True,
    self_per_cross_attn = 1,
    final_classifier_head = False,
    dim_ffn = 1024
)
```
### Initialize MeshTron model
```python
model = Meshtron(
    dim = 512,
    embedding_size = 131, #vocab size + special tokens
    n_heads = 16,
    head_dim = 32,
    window_size = 256,
    dim_ff = 1536,
    shortening_factor= 3,
    num_blocks_per_layer=[4,8,12], # hourglass structure
    ff_dropout = 0.0,
    attn_dropout = 0.0,
    pad_token = tokenizer.PAD.item(),
    condition_every_n_layers = 4,
    encoder=encoder
)
```
## Training

### Train MeshTron-Small on Primitive Dataset

Use the pre-built training pipeline to quickly get started:
```bash
python run.py
```

This command trains a MeshTron-Small model on the included primitive shapes dataset with default hyperparameters.
## Citation

```bibtex
@article{hao2024meshtron,
  title={Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale},
  author={Hao, Zekun and Romero, David W. and Lin, Tsung-Yi and Liu, Ming-Yu},
  journal={arXiv preprint arXiv:2412.09548},
  year={2024}
}
```

## Contact

For questions and discussions, please open an issue on GitHub.

---


**Note**: This is an unofficial implementation. For official code and models, please visit the [official project page](https://research.nvidia.com/labs/dir/meshtron).


