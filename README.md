# GRSN: Gate Recurrent Spiking Neuron for POMDP Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository implements **GRSN (Gate Recurrent Spiking Neuron)**, a spiking neural network architecture for reinforcement learning in partially observable environments (POMDPs). This work is based on research extending recurrent model-free RL methods with bio-inspired spiking neurons.

## Overview

### Key Features

- **Multiple Model Types**: Supports RNN (GRU/LSTM), SNN (LIF/RecurrentLIF/GRSNwoTAP/AdaptiveLIF), and MLP baselines
- **Unified Training Interface**: Single entry point for all experiments via `experiments/train.py`
- **Comprehensive Environments**: POMDP benchmarks, Meta-RL tasks, and Credit Assignment problems
- **Multiple RL Algorithms**: TD3, SAC, and SAC-Discrete

### SNN Neuron Types

| Neuron Type | Description |
|-------------|-------------|
| LIF | Leaky Integrate-and-Fire neuron |
| RecurrentLIF | LIF with recurrent connections and gating |
| GRSNwoTAP | GRSN without temporal attention mechanism |
| AdaptiveLIF | LIF with adaptive threshold |
| LIFwoTAP | Simplified LIF without TAP |

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GRSN.git
cd GRSN
```

2. Create a conda environment:
```bash
conda env create -f environments.yml
conda activate grsn
```

Or install via pip:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for specific environments:
```bash
# For PyBullet environments (Ant, HalfCheetah, etc.)
pip install pybullet
```

## Quick Start

### Basic Usage

Train an RNN agent on Pendulum-V:
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type rnn \
    --encoder gru \
    --algo sac \
    --seed 0
```

Train an SNN agent with RecurrentLIF:
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type snn \
    --snn_type RecurrentLIF \
    --algo sac \
    --seed 0 \
    --save_model
```

Train an MLP baseline:
```bash
python experiments/train.py \
    --env Pendulum-F-v0 \
    --model_type mlp \
    --algo sac \
    --seed 0
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Environment name (required) | - |
| `--model_type` | Model type: `mlp`, `rnn`, `snn` | `rnn` |
| `--snn_type` | SNN neuron type (for model_type=snn) | `RecurrentLIF` |
| `--encoder` | RNN encoder: `gru`, `lstm` (for model_type=rnn) | `gru` |
| `--algo` | RL algorithm: `td3`, `sac`, `sacd` | `sac` |
| `--seed` | Random seed | `0` |
| `--cuda` | CUDA device ID (-1 for CPU) | `0` |
| `--config` | Path to custom config file | Auto-detected |
| `--save_model` | Save trained model | False |

## Available Environments

### POMDP Benchmarks

| Environment | Description |
|-------------|-------------|
| `Pendulum-{F,P,V}-v0` | Classic control with partial observations |
| `CartPole-{F,P,V}-v0` | CartPole with partial observations |
| `HopperBLT-{F,P,V}-v0` | Hopper with partial observations |
| `WalkerBLT-{F,P,V}-v0` | Walker2D with partial observations |
| `AntBLT-{F,P,V}-v0` | Ant with partial observations |
| `HalfCheetahBLT-{F,P,V}-v0` | HalfCheetah with partial observations |

Suffixes:
- `F`: Full observation
- `P`: Position/angle only
- `V`: Velocity only

### Meta-RL Environments

| Environment | Description |
|-------------|-------------|
| `PointRobot-v0` | Point robot navigation |
| `Wind-v0` | Navigation with wind disturbance |
| `HalfCheetahVel-v0` | Velocity-following task |
| `AntDir-v0` | Direction-following task |
| `CheetahDir-v0` | HalfCheetah direction task |

### Credit Assignment

| Environment | Description |
|-------------|-------------|
| `Catch-{1,2,5,10,20,40}-v0` | Delayed reward catch game |
| `KeytoDoor-SR-v0` | Key-to-door credit assignment |

## Experiment Reproduction

### POMDP Benchmarks

Run experiments across multiple seeds:
```bash
for seed in 0 1 2 3 4; do
    python experiments/train.py \
        --env Pendulum-V-v0 \
        --model_type snn \
        --snn_type RecurrentLIF \
        --algo sac \
        --seed $seed
done
```

### Comparing Different Models

```bash
# RNN baseline
python experiments/train.py --env AntBLT-V-v0 --model_type rnn --encoder gru --algo sac --seed 0

# SNN variants
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type LIF --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type RecurrentLIF --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type GRSNwoTAP --algo sac --seed 0
```

## Project Structure

```
GRSN/
├── README.md                 # This file
├── README_CN.md             # Chinese documentation
├── requirements.txt         # Python dependencies
├── environments.yml         # Conda environment
│
├── grsn/                    # Main Python package
│   ├── models/             # Neural network models
│   ├── policies/           # RL policies (RNN, SNN, MLP)
│   ├── algorithms/         # RL algorithms (TD3, SAC, SACD)
│   ├── buffers/            # Experience replay buffers
│   ├── envs/               # Environment definitions
│   ├── utils/              # Utility functions
│   └── torchkit/           # PyTorch utilities
│
├── configs/                 # Configuration files
│   ├── pomdp/              # POMDP experiment configs
│   ├── meta/               # Meta-RL configs
│   ├── credit/             # Credit assignment configs
│   └── ...
│
├── experiments/             # Experiment scripts
│   └── train.py            # Unified training entry point
│
├── scripts/                 # Helper scripts
└── tests/                   # Unit tests
```

## Configuration

Configuration files are in YAML format. Example (`configs/pomdp/pendulum/v/rnn.yml`):

```yaml
train:
  num_updates_per_iter: 1.0
  buffer_size: 10000
  batch_size: 32
  num_iters: 1000
  num_init_rollouts_pool: 10
  num_rollouts_per_iter: 1
  sampled_seq_len: 50

policy:
  action_embedding_size: 8
  observ_embedding_size: 32
  reward_embedding_size: 8
  rnn_hidden_size: 128
  dqn_layers: [128, 128]
  policy_layers: [128, 128]
  lr: 3e-4
  gamma: 0.99
  tau: 5e-3
  sac:
    entropy_alpha: 0.1
    automatic_entropy_tuning: true
    alpha_lr: 3e-4
```

## Results

Results are saved in `./results/{env_name}/{experiment_name}.pth` as PyTorch dictionaries containing:
- `x`: Environment steps
- `y`: Average episode returns

Plot results using:
```python
import torch
import matplotlib.pyplot as plt

data = torch.load('results/Pendulum-V-v0/snn_RecurrentLIF_sac_seed0.pth')
plt.plot(data['x'], data['y'])
plt.xlabel('Environment Steps')
plt.ylabel('Average Return')
plt.show()
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'grsn'**
- Make sure you're running from the repository root
- Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**CUDA out of memory**
- Reduce `batch_size` in config
- Reduce `rnn_hidden_size` in config
- Use CPU: `--cuda -1`

**Environment not found**
- Check environment name spelling
- Ensure environment module is imported: `import grsn.envs.pomdp`

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{lang2025grsn,
  title     = {GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL},
  author    = {Lang, Qin and Ziming Wang and Runhao Jiang and Rui Yan and Huajin Tang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  series    = {AAAI'25},
  year      = {2025},
  publisher = {AAAI Press},
  location  = {Philadelphia, Pennsylvania, USA}
}
```

## Acknowledgments

This codebase is built upon:
- [Popular-RL-Algorithms](https://github.com/quantumiracle/Popular-RL-Algorithms) for RNN architecture inspiration
- [varibad](https://github.com/lmzintgraf/varibad) for hidden state update methods
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for spiking neuron implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

## 中文文档

See [README_CN.md](README_CN.md) for Chinese documentation.
