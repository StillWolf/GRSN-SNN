# MARL (QMIX + SMAC + GRSN) 使用指南

本仓库已实现论文 arXiv:2404.15597 (AAAI'25, Qin et al.) MARL 部分的**代码与 Mock
冒烟测试**。真实 SMAC 训练依赖 StarCraft II 游戏二进制，需要用户自行部署。

## 已实现的组件

| 组件 | 位置 | 说明 |
|---|---|---|
| QMIX 训练器 | `grsn/algorithms/marl/qmix.py` | TD loss + 单调 mixer + 硬 target update |
| 单调 Mixer | `grsn/algorithms/marl/mixer.py` | Hypernetwork + abs() 保证 ∂Q_tot/∂Q_i ≥ 0 |
| Agent 网络 | `grsn/algorithms/marl/agent_network.py` | 参数共享，RNN backbone 可插拔 |
| Episode Buffer | `grsn/buffers/episode_buffer.py` | CTDE 整 episode replay + padding mask |
| MARLEnv ABC | `grsn/envs/marl/base.py` | reset / step / get_obs / get_state / get_avail_actions |
| MockSMAC | `grsn/envs/marl/mock_smac.py` | 随机 env，冒烟测试用 |
| SMACWrapper | `grsn/envs/marl/smac_wrapper.py` | 真实 SMAC 的薄封装（延迟 import） |
| 训练入口 | `experiments/train_marl.py` | CLI + yaml config |
| 配置 | `configs/marl/smac/{8m,2s3z}/qmix_grsn.yml` | 论文对齐的超参默认 |
| 单元测试 | `tests/marl/` | 23 个单测 + 3 个端到端冒烟测试 |

## 快速开始（Mock env，不需要 SC2）

```bash
PYTHONPATH=. python experiments/train_marl.py \
    --env MockSMAC --map 8m --rnn_type GRSN \
    --seed 0 --cuda -1 --num_env_steps 2000
```

RNN backbone 可选 `gru` / `GRSN` / `LIF` / `GRSNwoTAP` / `LIFwoTAP`——这就是复现
论文 "GRSN vs GRU" 对比的开关。

## 真实 SMAC 训练

### 1. 安装 StarCraft II

**Linux（headless 训练推荐）：**

```bash
# 下载 Blizzard 的 SC2 for Linux（版本 ≥ 4.6.2）
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip -d ~/StarCraftII
export SC2PATH=~/StarCraftII
```

参考：https://github.com/Blizzard/s2client-proto#downloads

### 2. 安装 SMAC maps

```bash
git clone https://github.com/oxwhirl/smac.git
cp -r smac/smac/env/starcraft2/maps/SMAC_Maps $SC2PATH/Maps/
```

### 3. 安装 smac Python 包

```bash
pip install smac
```

### 4. 运行训练

```bash
export SC2PATH=~/StarCraftII  # 每次 shell 都要 export
PYTHONPATH=. python experiments/train_marl.py \
    --env SMAC --map 8m --rnn_type GRSN \
    --seed 0 --cuda 0 --num_env_steps 10000000
```

**论文对齐：**
- 5 seeds 同跑（`--seed 0..4`）
- 6 个 map：easy `8m` / `2s3z`，hard `8m_vs_9m` / `3s_vs_5z`，super hard `27m_vs_30m` / `MMM2`
- 每 map 每 seed 训 10M env steps
- RNN backbone 切换 `--rnn_type {GRSN, gru, GRSNwoTAP, LIF, LIFwoTAP}` 做对比

## 架构要点

### CTDE（centralized training distributed execution）

- 训练时：mixer 以全局 state 为输入，把 per-agent Q 混合成 Q_tot，监督信号来自 team reward
- 执行时：每个 agent 只用自己的 obs 过 `agent_net` 得到 Q，argmax 选动作
- 这样 mixer 只在训练用，部署时不需要 → 真正 decentralized execution

### 参数共享

所有 agent 共用一份 `AgentNetwork` 权重。实现上把 `n_agents` 折进 batch
维（`B × n_agents`），batched RNN 一次性前向。对同构 SMAC map（8m, 27m_vs_30m）
是标准做法。异构 map（2s3z, MMM2）也能跑，但若要 per-type sharing 需扩展
`AgentNetwork`。

### GRSN 状态管理

GRSN cell 的 `c`（门控电流）和 `spike_prev`（上一步脉冲）放在外部 state 张量里，
shape `(num_layers, B, 3*hidden_size)`。`AgentNetwork.init_hidden()` 按
`rnn.state_size_per_layer` 自动分配——切换 rnn_type 不用改训练代码。

### Mixer 单调性

Hypernetwork 生成的所有 mix 权重过 `abs()`，保证 `∂Q_tot / ∂Q_i ≥ 0`——这是
QMIX 值分解的数学基础。单测 `test_mixer_monotonicity_*` 直接验证该性质。

## 未实现 / 后续 TODO

- 其他 4 个 SMAC map 的 config（`8m_vs_9m`、`3s_vs_5z`、`27m_vs_30m`、`MMM2`）
- TensorBoard / wandb 日志接入（目前只 print 到 stdout）
- 评估曲线自动保存（仿照 `experiments/train.py` 把 `{x: env_steps, y: win_rate}` 存 pth）
- 多 seed / 多 map 批跑脚本（`scripts/run_marl_experiments.sh`）
- 异构 map 的 per-type parameter sharing（可选）

这些留给后续专门的实验复现任务。
