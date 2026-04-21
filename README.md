# GRSN: Gate Recurrent Spiking Neuron for POMDP Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository implements the POMDP portion of **GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL** (Qin et al., AAAI 2025, [arXiv:2404.15597](https://arxiv.org/abs/2404.15597)). The MARL portion (QMIX on SMAC) is not yet implemented — see [docs/MARL_EXTENSION.md](docs/MARL_EXTENSION.md) for the planned integration.

The architectural scaffolding is adapted from [pomdp-baselines](https://github.com/twni2016/pomdp-baselines) (Ni et al., ICML 2022).

## Overview

### Key Features

- **Multiple Model Types**: Supports RNN (GRU/LSTM), SNN (LIF/LIFwoTAP/GRSN/GRSNwoTAP), and MLP baselines
- **Unified Training Interface**: Single entry point for all experiments via `experiments/train.py`
- **Comprehensive Environments**: POMDP benchmarks, Meta-RL tasks, and Credit Assignment problems
- **Multiple RL Algorithms**: TD3, SAC, and SAC-Discrete

### SNN Neuron Types

| Neuron Type | Time steps | Rate coding | Description |
|-------------|------------|-------------|-------------|
| `LIF` | 1 | No | Baseline LIF with TAP: hard reset, fixed β=0.5, no gates |
| `LIFwoTAP` | 4 | Yes | Same LIF baseline but with T=4 rate coding (no TAP) |
| `GRSN` | 1 | No | **Paper's main model**: Eq.17 gated input current driven by o_{t-1}, learnable β, soft reset, TAP-aligned (T=1) |
| `GRSNwoTAP` | 4 | Yes | GRSN without TAP: T=4 rate coding ablation |

## Installation

**完整环境配置指南见 [`docs/SETUP.md`](docs/SETUP.md)**——包含 Python 版本要求、conda 创建、pip 兜底、SC2/SMAC 安装、GPU 注意事项、常见问题排查。

简版 TL;DR：

```bash
git clone https://github.com/StillWolf/GRSN-SNN.git
cd GRSN-SNN
conda env create -f environments.yml
conda activate grsn

# conda env create 在某些版本会跳过 pip 段，验证 + 兜底
python -c "import torch, gym, spikingjelly, pycolab" || pip install -r requirements.txt

# 验证：41 个测试应该全部 PASS
PYTHONPATH=. python -m pytest tests/ -v
```

实测组合：Python 3.10 + PyTorch 2.4 + gym 0.26.2 + numpy 1.26 + spikingjelly。

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

Train an SNN agent with GRSN:
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type snn \
    --snn_type GRSN \
    --algo td3 \
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
| `--snn_type` | SNN neuron type: `LIF/LIFwoTAP/GRSN/GRSNwoTAP` | `GRSN` |
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
        --snn_type GRSN \
        --algo sac \
        --seed $seed
done
```

### Comparing Different Models

```bash
# RNN baseline
python experiments/train.py --env AntBLT-V-v0 --model_type rnn --encoder gru --algo sac --seed 0

# SNN variants (all paper-aligned)
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type LIF --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type LIFwoTAP --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type GRSN --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type GRSNwoTAP --algo sac --seed 0
```

## Project Structure

```
GRSN-SNN/
├── README.md                # English docs
├── README_CN.md             # Chinese docs
├── requirements.txt         # pip deps
├── environments.yml         # conda env
│
├── grsn/                    # Main Python package
│   ├── policies/
│   │   ├── rlifs/           # SNN cells: LIF / LIFwoTAP / GRSN / GRSNwoTAP
│   │   ├── policy_mlp.py
│   │   ├── policy_rnn.py
│   │   ├── policy_snn.py
│   │   ├── spiking_actor.py
│   │   └── spiking_critic.py
│   ├── algorithms/          # RL algos (TD3 / SAC / SACD)
│   │   └── marl/            # MARL placeholder (not implemented — see docs/MARL_EXTENSION.md)
│   ├── buffers/             # Replay buffers
│   ├── envs/                # POMDP / Meta-RL / CreditAssign envs
│   ├── utils/
│   └── torchkit/
│
├── configs/                 # YAML configs (pomdp / meta / credit)
├── experiments/train.py     # Training entry point
├── scripts/                 # Batch runners + plotting
├── tests/                   # Unit tests (pytest)
└── docs/                    # MARL extension guide + implementation plans
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

data = torch.load('results/Pendulum-V-v0/GRSN_td3_seed0.pth')
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

## Key concepts (from the paper)

- **Temporal Alignment Paradigm (TAP)**: instead of running an SNN for `T>1` simulation
  steps to encode a single MDP state (the conventional rate-coding approach), TAP
  aligns one SNN step with one MDP step (`T=1`). Spiking neuron state then accumulates
  across MDP time naturally, and the spike becomes the only per-step output. This
  is how `LIF` and `GRSN` run in this repo.
- **Gated Recurrent Spiking Neuron (GRSN)**: adds a gated input-current mechanism
  where both forget gate `F` and input gate `I` are driven by the *previous spike*
  `o_{t-1}` (paper Eq. 17). Combined with a learnable leak `β` and soft reset, this
  equips spiking neurons with GRU-like long-term memory.
- **External state tensor**: `GRSN` stores gate state `[h, c, spike_prev]` in the
  caller-managed state tensor (shape `(num_layers, B, 3*hidden_size)`), so the gated
  recurrence is preserved across `act()` calls during inference. `LIF` only needs
  `h`, so its state is `(num_layers, B, hidden_size)`.

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
