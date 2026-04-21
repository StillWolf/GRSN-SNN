# GRSN: 门控循环脉冲神经网络用于POMDP强化学习

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本仓库实现论文 **GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL**（Qin 等, AAAI 2025, [arXiv:2404.15597](https://arxiv.org/abs/2404.15597)）的 POMDP 部分。MARL 部分（QMIX on SMAC）尚未实现，见 [docs/MARL_EXTENSION.md](docs/MARL_EXTENSION.md)。

整体架构参考 [pomdp-baselines](https://github.com/twni2016/pomdp-baselines)（Ni 等, ICML 2022）的 Separate Recurrent Actor-Critic。

## 项目概述

### 主要特点

- **多种模型类型**: 支持RNN (GRU/LSTM)、SNN (LIF/LIFwoTAP/GRSN/GRSNwoTAP) 和MLP基线
- **统一的训练接口**: 通过`experiments/train.py`单一入口点进行所有实验
- **全面的环境**: POMDP基准测试、元学习任务和信用分配问题
- **多种RL算法**: TD3、SAC 和 SAC-离散

### SNN神经元类型

| 神经元类型 | 仿真步 T | Rate coding | 说明 |
|-----------|---------|-------------|------|
| `LIF` | 1 | 否 | 基线 LIF，TAP 对齐：硬复位、β=0.5 常数、无门控 |
| `LIFwoTAP` | 4 | 是 | LIF 的 T=4 rate coding 消融变体 |
| `GRSN` | 1 | 否 | **论文主模型**：Eq.17 门控由 o_{t-1} 驱动、可学习 β、软复位、T=1 TAP 对齐 |
| `GRSNwoTAP` | 4 | 是 | GRSN 的 T=4 rate coding 消融变体 |

## 安装

**完整环境配置指南见 [`docs/SETUP.md`](docs/SETUP.md)**——包含 Python 版本要求、conda 创建、pip 兜底、SC2/SMAC 安装、GPU 注意事项、常见问题排查。

简版 TL;DR：

```bash
git clone https://github.com/StillWolf/GRSN-SNN.git
cd GRSN-SNN
conda env create -f environments.yml
conda activate grsn

# conda env create 在某些版本会跳过 pip 段，验证 + 兜底
python -c "import torch, gym, spikingjelly, pycolab" || pip install -r requirements.txt

# 验证：41 个测试应全部 PASS
PYTHONPATH=. python -m pytest tests/ -v
```

实测组合：Python 3.10 + PyTorch 2.4 + gym 0.26.2 + numpy 1.26 + spikingjelly。

## 快速开始

### 基本用法

在Pendulum-V上训练RNN智能体:
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type rnn \
    --encoder gru \
    --algo sac \
    --seed 0
```

使用GRSN训练SNN智能体:
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type snn \
    --snn_type GRSN \
    --algo td3 \
    --seed 0 \
    --save_model
```

训练MLP基线:
```bash
python experiments/train.py \
    --env Pendulum-F-v0 \
    --model_type mlp \
    --algo sac \
    --seed 0
```

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--env` | 环境名称 (必需) | - |
| `--model_type` | 模型类型: `mlp`, `rnn`, `snn` | `rnn` |
| `--snn_type` | SNN神经元类型: `LIF/LIFwoTAP/GRSN/GRSNwoTAP` | `GRSN` |
| `--encoder` | RNN编码器: `gru`, `lstm` (model_type=rnn时使用) | `gru` |
| `--algo` | RL算法: `td3`, `sac`, `sacd` | `sac` |
| `--seed` | 随机种子 | `0` |
| `--cuda` | CUDA设备ID (-1表示使用CPU) | `0` |
| `--config` | 自定义配置文件路径 | 自动检测 |
| `--save_model` | 保存训练好的模型 | False |

## 可用环境

### POMDP基准测试

| 环境 | 描述 |
|------|------|
| `Pendulum-{F,P,V}-v0` | 经典控制任务的部分观察版本 |
| `CartPole-{F,P,V}-v0` | 车杆平衡的部分观察版本 |
| `HopperBLT-{F,P,V}-v0` | Hopper的部分观察版本 |
| `WalkerBLT-{F,P,V}-v0` | Walker2D的部分观察版本 |
| `AntBLT-{F,P,V}-v0` | Ant的部分观察版本 |
| `HalfCheetahBLT-{F,P,V}-v0` | HalfCheetah的部分观察版本 |

后缀说明:
- `F`: 完全观察
- `P`: 仅位置/角度
- `V`: 仅速度

### 元学习 (Meta-RL) 环境

| 环境 | 描述 |
|------|------|
| `PointRobot-v0` | 点机器人导航 |
| `Wind-v0` | 带风力干扰的导航 |
| `HalfCheetahVel-v0` | 速度跟随任务 |
| `AntDir-v0` | 方向跟随任务 |
| `CheetahDir-v0` | HalfCheetah方向任务 |

### 信用分配

| 环境 | 描述 |
|------|------|
| `Catch-{1,2,5,10,20,40}-v0` | 延迟奖励的接球游戏 |
| `KeytoDoor-SR-v0` | 钥匙到门的信用分配任务 |

## 实验复现

### POMDP基准测试

在多个种子上运行实验:
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

### 比较不同模型

```bash
# RNN基线
python experiments/train.py --env AntBLT-V-v0 --model_type rnn --encoder gru --algo sac --seed 0

# SNN变体（论文对齐的全部四种）
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type LIF --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type LIFwoTAP --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type GRSN --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type GRSNwoTAP --algo sac --seed 0
```

## 项目结构

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

## 配置

配置文件采用YAML格式。示例 (`configs/pomdp/pendulum/v/rnn.yml`):

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

## 结果分析

结果保存在`./results/{env_name}/{experiment_name}.pth`，包含:
- `x`: 环境步数
- `y`: 平均回合回报

使用以下代码绘制结果:
```python
import torch
import matplotlib.pyplot as plt

data = torch.load('results/Pendulum-V-v0/GRSN_td3_seed0.pth')
plt.plot(data['x'], data['y'])
plt.xlabel('环境步数')
plt.ylabel('平均回报')
plt.show()
```

## 常见问题解决

### 常见问题

**ImportError: No module named 'grsn'**
- 确保从仓库根目录运行
- 添加到PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**CUDA out of memory (显存不足)**
- 减小配置中的`batch_size`
- 减小配置中的`rnn_hidden_size`
- 使用CPU: `--cuda -1`

**Environment not found (环境未找到)**
- 检查环境名称拼写
- 确保环境模块已导入: `import grsn.envs.pomdp`

## 关键概念（出自原论文）

- **时序对齐范式（Temporal Alignment Paradigm, TAP）**：传统脉冲 RL 里一个 MDP step 需要
  跑 T>1 个 SNN 仿真步（rate coding）；TAP 把 "1 个 SNN 步" 直接对齐到 "1 个 MDP 步"
  （T=1），让脉冲神经元的状态沿 MDP 时间自然累积，一个 MDP step 只输出一个 spike。
  本仓库的 `LIF` 和 `GRSN` 即 TAP 模式。
- **门控循环脉冲神经元（GRSN）**：在 LIF 之上加入门控输入电流，遗忘门 F 与输入门 I
  都由 *前一时刻脉冲* `o_{t-1}` 驱动（Eq.17），再配合可学习的衰减因子 β 与软复位，
  赋予脉冲神经元 GRU 级别的长短时记忆能力。
- **外部 state 张量**：`GRSN` 把门控状态 `[h, c, spike_prev]` 存在外部 state 张量里
  （形状 `(num_layers, B, 3*hidden_size)`），保证推理阶段 `act()` 调用之间门控递归
  不丢失。`LIF` 只需 `h`，state 形状为 `(num_layers, B, hidden_size)`。

## 引用

如果本代码对您的研究有帮助，请引用:

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

## 致谢

本代码库基于以下项目构建:
- [Popular-RL-Algorithms](https://github.com/quantumiracle/Popular-RL-Algorithms) - RNN架构灵感
- [varibad](https://github.com/lmzintgraf/varibad) - 隐藏状态更新方法
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - 脉冲神经元实现

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 联系方式

如有问题或建议，请提交GitHub issue或联系作者。
