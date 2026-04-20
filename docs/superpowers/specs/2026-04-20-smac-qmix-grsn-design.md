# QMIX + GRSN on SMAC 接入设计

**Goal：** 为 `GRSN-SNN` 仓库加入论文 arXiv:2404.15597 (AAAI'25) 的 MARL 部分——QMIX 算法 + SMAC 环境接入 + GRSN 作为 agent RNN backbone——使之结构上能复现论文第二半的实验。本次会话的落地目标是**代码 + 单测 + Mock 冒烟测试**；真实 SC2 下载与训练曲线留给用户在 GPU 空闲时自行启动。

## 1. 设计原则

1. **骨架风格对齐现有仓库**：QMIX 放 `grsn/algorithms/marl/qmix.py`，与 `sac.py`/`td3.py` 并列；训练脚本 `experiments/train_marl.py` 独立于 `train.py`（单智能体 / 多智能体的 rollout + replay 机制差别太大，硬合效果差）
2. **RNN backbone 可插拔**：`agent_network` 从 `grsn.policies.rlifs.REGISTRY` 取神经元（`GRSN` / `LIF` / `GRSNwoTAP` / `LIFwoTAP`），也支持 `nn.GRU`（用于复现论文 "GRU vs GRSN" 对比基线）
3. **环境抽象 → Mock/真实可热插**：训练 loop 只依赖 `MARLEnv` ABC，`mock_smac` 和 `smac_wrapper` 遵守同一接口
4. **复现论文规格，不追求 pymarl2 完全等价**：只实现 vanilla QMIX（无 WQMIX / QPLEX / double-Q-value etc.）——论文对照的 baseline 就是这个版本

## 2. 目录结构

```
grsn/
├── algorithms/marl/
│   ├── __init__.py
│   ├── qmix.py              # QMIX 训练器（计算 TD 损失 + 梯度更新）
│   ├── mixer.py             # 单调混合网络
│   └── agent_network.py     # 参数共享的 agent 网络（RNN backbone）
├── buffers/
│   └── episode_buffer.py    # CTDE 整条 episode replay buffer
├── envs/marl/
│   ├── __init__.py
│   ├── base.py              # MARLEnv 抽象基类
│   ├── mock_smac.py         # 冒烟测试 mock 环境
│   └── smac_wrapper.py      # 真实 SMAC 包装（延迟 import）
├── experiments/
│   └── train_marl.py        # MARL 训练入口
└── configs/marl/smac/
    ├── 8m/qmix_grsn.yml     # 默认超参（本次只填 8m 一个 map，其余后续补）
    └── 2s3z/qmix_grsn.yml
```

`docs/MARL_EXTENSION.md` 同步更新，从 "roadmap" 改为 "implementation guide"。

## 3. 组件接口

### 3.1 `MARLEnv`（base.py）

抽象类，遵循 PyMARL / SMAC 约定：

```python
class MARLEnv(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> None: ...
    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> tuple[float, bool, dict]:
        """Returns (reward, terminated, info). reward 是团队共享标量。"""
    @abc.abstractmethod
    def get_obs(self) -> np.ndarray:   # (n_agents, obs_shape)
        ...
    @abc.abstractmethod
    def get_state(self) -> np.ndarray: # (state_shape,)
        ...
    @abc.abstractmethod
    def get_avail_actions(self) -> np.ndarray:  # (n_agents, n_actions) binary
        ...
    @abc.abstractmethod
    def get_env_info(self) -> dict:
        """至少包含：n_agents, n_actions, obs_shape, state_shape, episode_limit。"""
    def close(self) -> None:
        pass
```

### 3.2 `MockSMAC`（mock_smac.py）

- 构造参数：`n_agents=8, n_actions=14, obs_dim=80, state_dim=168, episode_limit=60`（8m 的常见维度）
- `reset()`：episode 计数清零
- `step()`：episode 长度 = `random.randint(20, episode_limit)`；reward ~ `N(0, 1)`；行动受 `get_avail_actions` 限制（全 1 开放）
- **仅用于冒烟/单测，训练无意义**

### 3.3 `SMACWrapper`（smac_wrapper.py）

```python
try:
    from smac.env import StarCraft2Env
except ImportError as e:
    raise ImportError(
        "smac not installed. See docs/MARL_EXTENSION.md for SC2 + smac install."
    ) from e
```

延迟到真正 `__init__` 时触发；只是 `StarCraft2Env` 的薄封装，把 `get_obs` 等方法转成 `MARLEnv` 签名。**本次只写代码不运行真实 env。**

### 3.4 `AgentNetwork`（agent_network.py）

```python
class AgentNetwork(nn.Module):
    """参数共享的 agent 网络。所有 agent 共用一份权重；agent 间差异只通过各自的 obs + hidden state 区分。"""
    def __init__(self, obs_dim, n_actions, rnn_type, rnn_hidden_size, obs_embed_size=64):
        ...
        self.obs_embed = nn.Linear(obs_dim, obs_embed_size)
        # rnn_type ∈ {"gru", "GRSN", "LIF", "GRSNwoTAP", "LIFwoTAP"}
        self.rnn = _build_rnn(rnn_type, obs_embed_size, rnn_hidden_size)
        self.q_head = nn.Linear(rnn_hidden_size, n_actions)

    def forward(self, obs, state):
        """
        obs: (T, B*n_agents, obs_dim)
        state: (num_layers, B*n_agents, state_size_per_layer)  # 或 None
        returns: q_vals (T, B*n_agents, n_actions), new_state
        """
```

- 训练时 `T = episode_limit`，一次性前向；推理时 `T=1`，把 state 传回
- 参数共享用 "fold n_agents into batch" 实现——这是 pymarl2 的经典做法，避免写双层 for

### 3.5 `Mixer`（mixer.py）

QMIX 单调混合网络（Rashid et al. 2018）：

```python
class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32, hypernet_layers=2):
        ...
    def forward(self, agent_qs, states):
        """
        agent_qs: (B, T, n_agents) — 每个 agent 选择动作的 Q 值
        states:   (B, T, state_dim)
        returns:  (B, T, 1) — joint Q
        """
```

- Hypernetworks：`W1, b1, W2, b2` 都由 state 生成
- `W1, W2` 过 `abs()` 保证 `∂Q_tot/∂Q_i ≥ 0`（单调性）
- 两层 mix（hypernet_layers=2）

### 3.6 `EpisodeBuffer`（episode_buffer.py）

```python
class EpisodeBuffer:
    def __init__(self, buffer_size, episode_limit, n_agents, obs_dim, state_dim, n_actions):
        """按 episode 存储，每个 episode 定长 padded to episode_limit + 1 step。"""
    def insert(self, episode: dict) -> None: ...
    def sample(self, batch_size: int) -> dict:
        """返回 padded batch，带 filled_mask 区分真实 step 和 padding。"""
    def __len__(self) -> int: ...
```

Fields per episode：`obs`, `state`, `actions`, `avail_actions`, `rewards`, `terminated`, `filled`.

### 3.7 `QMIX`（qmix.py）

```python
class QMIX:
    def __init__(self, agent_net, target_agent_net, mixer, target_mixer, lr, gamma, grad_clip):
        ...
    def train_step(self, batch) -> dict:
        """
        1. 对 batch 里每个 episode 前向 agent_net 得到 Q_eval
        2. 对同一 batch 前向 target_agent_net 得到 Q_target（next obs）
        3. 选动作：eval 用 batch['actions']；target 用 argmax over avail_actions
        4. mixer(Q_eval_chosen, state) -> Q_tot_eval
        5. target_mixer(Q_target_max, next_state) -> Q_tot_target
        6. y = reward + gamma * (1 - done) * Q_tot_target
        7. loss = mean((Q_tot_eval - y.detach())^2 * filled_mask) / filled_mask.sum()
        8. backward + grad clip + step
        返回 {'loss': ..., 'grad_norm': ..., 'q_tot_mean': ...}
        """
    def target_update(self) -> None:
        """硬更新 target 网络（复制参数）。"""
```

### 3.8 `train_marl.py`

CLI：
```
python experiments/train_marl.py \
    --env MockSMAC  \       # 或 SMAC
    --map 8m \
    --rnn_type GRSN \       # GRSN/LIF/GRSNwoTAP/LIFwoTAP/gru
    --seed 0 \
    --cuda -1 \
    --num_env_steps 2050000 \  # 论文 10M；冒烟传小值
    --config configs/marl/smac/8m/qmix_grsn.yml
```

**顶层 loop：**
1. 从 env 收 rollout，填进 EpisodeBuffer（ε-greedy 动作）
2. 当 buffer 有 ≥ batch_size 个 episode 时，开始 train：每 episode 训 1 次 `QMIX.train_step`
3. 每 target_update_interval 次训练后 hard copy target 网络
4. 每 eval_interval 个 env step 跑一次 greedy eval

## 4. 论文对齐的默认超参

写入 `configs/marl/smac/8m/qmix_grsn.yml`，直接用 pymarl2 vanilla-QMIX 的默认（论文未专门重写）：

| 项 | 值 | 说明 |
|---|---|---|
| `agent.rnn_hidden_size` | 64 | pymarl2 默认 |
| `agent.obs_embed_size` | 64 | 稍大于 rnn_hidden 也可，这里取 64 |
| `mixer.embed_dim` | 32 | pymarl2 默认 |
| `mixer.hypernet_layers` | 2 | 两层 hypernet |
| `train.lr` | 5e-4 | Adam |
| `train.gamma` | 0.99 | |
| `train.grad_clip` | 10.0 | |
| `train.batch_size` | 32 | 单位是 episode |
| `train.buffer_size` | 5000 | 单位是 episode |
| `train.target_update_interval` | 200 | 训练 iter 数 |
| `train.num_env_steps` | 10_000_000 | 论文值 |
| `explore.epsilon_start` | 1.0 | |
| `explore.epsilon_end` | 0.05 | |
| `explore.epsilon_anneal_time` | 50_000 | env steps |
| `eval.interval_env_steps` | 20_000 | |
| `eval.num_episodes` | 32 | |
| `seed` | 0 | |

论文特定要求：
- **RNN 类型默认 GRSN，T=1 TAP 对齐**（外部 state 张量 `(L, B*n_agents, 3*H)`）
- Agent parameter sharing：**开启**（同构 map 上是标准做法；若以后扩展到异构 map 可加开关）

## 5. 测试策略

`tests/marl/`：

1. `test_mixer_monotonicity.py`：固定 state，逐个递增 agent_q，验证 Q_tot 非降
2. `test_mixer_shapes.py`：(B=4, T=5, n_agents=8) → Q_tot (4,5,1)
3. `test_agent_network_param_sharing.py`：所有 agent 跑相同 obs 时得到相同 Q（权重被共享）
4. `test_agent_network_grsn_state.py`：rnn_type="GRSN" 时 state shape 正确（3*H），连续调用传回 state 结果与一次性前向一致
5. `test_episode_buffer_roundtrip.py`：insert N 个随机 episode，sample 回来维度和值对得上
6. `test_mock_smac_api.py`：`reset()/step()/get_obs()/get_state()/get_avail_actions()/get_env_info()` 全部返回预期 dtype/shape
7. `test_qmix_smoke.py`：端到端跑 `train_marl.py --env MockSMAC --rnn_type GRSN --num_env_steps 500 --cuda -1` 不报错、loss 有下降趋势（弱断言 loss[-1] < loss[0]，不强求）

## 6. 本次会话范围（Scope Ⅰ）

**In scope：**
- 所有 Section 2 里的文件
- 所有 Section 5 的测试
- `MockSMAC` 端到端跑通
- `SMACWrapper` 代码写完但不导入真实 smac（延迟 import）
- `docs/MARL_EXTENSION.md` 更新为 "已实现，用法 + SC2 安装步骤"
- 8m 和 2s3z 两份 config（其余 4 个 map 后续补）

**Out of scope（后续任务）：**
- 下载 StarCraft II 二进制
- 安装 `smac` pip 包并验证真实 env
- 任何 >1000 env steps 的训练
- GPU 上的任何执行
- 其余 4 个 SMAC map 的 config

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| `AgentNetwork` 参数共享用 batch fold 实现可能和 GRSN 的外部 state 对齐出错 | 单测 `test_agent_network_grsn_state.py` 直接比"T=1 连续 2 步" vs "T=2 一次" 的输出 |
| Mixer 单调性容易写漏 `abs()` | 单测 `test_mixer_monotonicity.py` 随机扰动验证 |
| EpisodeBuffer 的 padding/mask 逻辑是经典 off-by-one 重灾区 | 单测专门 cover "episode 长度 < episode_limit" 的情况 |
| 真实 SMAC env API 未来可能微调 | `MARLEnv` ABC 做适配层；`SMACWrapper` 变动集中在一个文件 |

## 8. 与现有代码的边界

- 不改 `grsn/policies/rlifs/*`（已修好）
- 不改 `grsn/algorithms/{base,sac,sacd,td3}.py`（单智能体代码）
- `grsn/buffers/` 新增 `episode_buffer.py`，不动现有两个 buffer
- `experiments/train.py` 保持单智能体入口；MARL 入口 `train_marl.py` 完全独立
- `grsn/algorithms/marl/__init__.py` 从 placeholder 改为真实 export
