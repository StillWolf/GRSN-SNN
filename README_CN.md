# GRSN: 门控循环脉冲神经网络用于POMDP强化学习

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本项目实现了**GRSN (Gate Recurrent Spiking Neuron)**，一种用于部分可观察马尔可夫决策过程(POMDP)强化学习的脉冲神经网络架构。本工作基于ICML 2022论文"Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs"，将循环无模型RL方法与生物启发的脉冲神经元相结合。

## 项目概述

### 主要特点

- **多种模型类型**: 支持RNN (GRU/LSTM)、SNN (LIF/RecurrentLIF/GRSNwoTAP/AdaptiveLIF) 和MLP基线
- **统一的训练接口**: 通过`experiments/train.py`单一入口点进行所有实验
- **全面的环境**: POMDP基准测试、元学习任务和信用分配问题
- **多种RL算法**: TD3、SAC 和 SAC-离散

### SNN神经元类型

| 神经元类型 | 描述 |
|-----------|------|
| LIF | 泄漏积分发放神经元 |
| RecurrentLIF | 带循环连接和门控的LIF |
| GRSNwoTAP | 无时间注意力机制的GRSN |
| AdaptiveLIF | 带自适应阈值的LIF |
| LIFwoTAP | 简化版LIF，无TAP |

## 安装

### 环境要求

- Python 3.8 或更高版本
- CUDA-capable GPU (推荐)

### 安装步骤

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/GRSN.git
cd GRSN
```

2. 创建conda环境:
```bash
conda env create -f environments.yml
conda activate grsn
```

或者使用pip安装:
```bash
pip install -r requirements.txt
```

3. 安装特定环境的额外依赖:
```bash
# 对于PyBullet环境 (Ant, HalfCheetah等)
pip install pybullet
```

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

使用RecurrentLIF训练SNN智能体:
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type snn \
    --snn_type RecurrentLIF \
    --algo sac \
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
| `--snn_type` | SNN神经元类型 (model_type=snn时使用) | `RecurrentLIF` |
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
        --snn_type RecurrentLIF \
        --algo sac \
        --seed $seed
done
```

### 比较不同模型

```bash
# RNN基线
python experiments/train.py --env AntBLT-V-v0 --model_type rnn --encoder gru --algo sac --seed 0

# SNN变体
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type LIF --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type RecurrentLIF --algo sac --seed 0
python experiments/train.py --env AntBLT-V-v0 --model_type snn --snn_type GRSNwoTAP --algo sac --seed 0
```

## 项目结构

```
GRSN/
├── README.md                 # 英文文档
├── README_CN.md             # 本文件
├── requirements.txt         # Python依赖
├── environments.yml         # Conda环境配置
│
├── grsn/                    # 主Python包
│   ├── models/             # 神经网络模型
│   ├── policies/           # RL策略 (RNN, SNN, MLP)
│   ├── algorithms/         # RL算法 (TD3, SAC, SACD)
│   ├── buffers/            # 经验回放缓冲区
│   ├── envs/               # 环境定义
│   ├── utils/              # 工具函数
│   └── torchkit/           # PyTorch工具
│
├── configs/                 # 配置文件
│   ├── pomdp/              # POMDP实验配置
│   ├── meta/               # 元学习配置
│   ├── credit/             # 信用分配配置
│   └── ...
│
├── experiments/             # 实验脚本
│   └── train.py            # 统一训练入口
│
├── scripts/                 # 辅助脚本
└── tests/                   # 单元测试
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

data = torch.load('results/Pendulum-V-v0/snn_RecurrentLIF_sac_seed0.pth')
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
