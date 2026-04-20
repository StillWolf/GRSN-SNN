# QMIX + GRSN on SMAC 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/superpowers/specs/2026-04-20-smac-qmix-grsn-design.md` 落地 QMIX + GRSN + SMAC 接入——代码、单测、Mock 冒烟测试全部通过，真实 SC2/训练留给后续。

**Architecture:** `grsn/algorithms/marl/` 放 QMIX / Mixer / AgentNetwork，`grsn/envs/marl/` 放 MARLEnv 抽象 + MockSMAC + SMACWrapper（延迟 import），`grsn/buffers/episode_buffer.py` 放 CTDE 整 episode buffer，`experiments/train_marl.py` 独立训练入口。RNN backbone 通过字符串参数从 `grsn.policies.rlifs.REGISTRY` 或 `nn.GRU` 挑选——可热插拔。

**Tech Stack:** PyTorch ≥1.9, spikingjelly, gym (已装), ruamel.yaml, pytest。**No GPU** — 本次全程 CPU（`--cuda -1`）。

---

## HARD Constraints（全部 Task 共享）

1. **不要运行 `git config`**。Git 身份已配置为 StillWolf（local）。
2. **不要用 GPU**。所有脚本传 `--cuda -1` 或等价。运行前不必 `nvidia-smi`（我们已知道当前 GPU 全忙）。
3. **Commit message 用中文**，不加 `Co-Authored-By`。
4. **不下载 StarCraft II**；不 `pip install smac`（要做 SC2 才装）。`SMACWrapper` 用延迟 import 即可。
5. **Working directory 始终在** `/home/edge/RoboRL/GRSN/GRSN-SNN`。

---

## 文件结构

新建/修改的全部文件：

```
grsn/
├── algorithms/marl/
│   ├── __init__.py          # [修改] 从 raise NotImplementedError 改为 export QMIX
│   ├── qmix.py              # [新] 训练器
│   ├── mixer.py             # [新] 单调混合网络
│   └── agent_network.py     # [新] 参数共享 agent 网络
├── buffers/
│   └── episode_buffer.py    # [新] CTDE episode buffer
└── envs/marl/
    ├── __init__.py          # [新]
    ├── base.py              # [新] MARLEnv ABC
    ├── mock_smac.py         # [新]
    └── smac_wrapper.py      # [新] 延迟 import

experiments/
└── train_marl.py            # [新] MARL 训练入口

configs/marl/smac/
├── 8m/qmix_grsn.yml         # [新]
└── 2s3z/qmix_grsn.yml       # [新]

tests/marl/
├── __init__.py              # [新]
├── test_mock_smac_api.py
├── test_mixer.py
├── test_agent_network.py
├── test_episode_buffer.py
└── test_qmix_smoke.py

docs/
└── MARL_EXTENSION.md        # [修改] 从 roadmap 改为 implementation guide
```

---

## Task 1：MARL 环境抽象 + MockSMAC + 单测

**Files:**
- Create: `grsn/envs/marl/__init__.py`
- Create: `grsn/envs/marl/base.py`
- Create: `grsn/envs/marl/mock_smac.py`
- Create: `tests/marl/__init__.py`
- Create: `tests/marl/test_mock_smac_api.py`

- [ ] **Step 1：创建 `grsn/envs/marl/base.py`**

```python
"""MARL 环境抽象基类。

接口遵循 PyMARL / SMAC 约定：obs 和 state 通过各自 getter 访问；reward 是团队共享标量。
"""
import abc
from typing import Dict, Tuple

import numpy as np


class MARLEnv(abc.ABC):
    """多智能体环境基类。MockSMAC 和 SMACWrapper 都继承此类。"""

    @abc.abstractmethod
    def reset(self) -> None:
        """开始新 episode。"""

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        """执行一步，返回 (team_reward, terminated, info)。actions: (n_agents,) int。"""

    @abc.abstractmethod
    def get_obs(self) -> np.ndarray:
        """每个 agent 的局部 obs，shape (n_agents, obs_dim)。"""

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """全局 state（仅训练时 mixer 用），shape (state_dim,)。"""

    @abc.abstractmethod
    def get_avail_actions(self) -> np.ndarray:
        """每 agent 的可执行动作 mask，shape (n_agents, n_actions)，0/1 binary。"""

    @abc.abstractmethod
    def get_env_info(self) -> Dict:
        """返回 {n_agents, n_actions, obs_shape, state_shape, episode_limit}。"""

    def close(self) -> None:
        """子类可覆盖。默认 no-op。"""
```

- [ ] **Step 2：创建 `grsn/envs/marl/mock_smac.py`**

```python
"""冒烟测试用 mock SMAC 环境。

默认维度对应 SMAC 的 8m map：n_agents=8, n_actions=14, obs_dim=80, state_dim=168。
obs/state/reward 均为随机，只用于验证训练 loop 可跑通，不具训练意义。
"""
from typing import Dict, Tuple

import numpy as np

from grsn.envs.marl.base import MARLEnv


class MockSMAC(MARLEnv):
    def __init__(
        self,
        n_agents: int = 8,
        n_actions: int = 14,
        obs_dim: int = 80,
        state_dim: int = 168,
        episode_limit: int = 60,
        seed: int = 0,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.episode_limit = episode_limit
        self._rng = np.random.RandomState(seed)
        self._step = 0
        self._horizon = 0  # 本 episode 的实际长度

    def reset(self) -> None:
        self._step = 0
        self._horizon = int(self._rng.randint(20, self.episode_limit + 1))

    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        assert actions.shape == (self.n_agents,)
        self._step += 1
        reward = float(self._rng.randn())
        terminated = self._step >= self._horizon
        return reward, terminated, {}

    def get_obs(self) -> np.ndarray:
        return self._rng.randn(self.n_agents, self.obs_dim).astype(np.float32)

    def get_state(self) -> np.ndarray:
        return self._rng.randn(self.state_dim).astype(np.float32)

    def get_avail_actions(self) -> np.ndarray:
        # 全部开放；某些场景可加禁用第 0 个动作等，但 mock 保持全开
        return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

    def get_env_info(self) -> Dict:
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_dim,
            "state_shape": self.state_dim,
            "episode_limit": self.episode_limit,
        }
```

- [ ] **Step 3：创建 `grsn/envs/marl/__init__.py`**

```python
"""MARL 环境包。

- MARLEnv：抽象基类
- MockSMAC：冒烟测试用的随机 env
- SMACWrapper：真实 SMAC 的薄封装（仅在 import 时才加载 smac/SC2 依赖）
"""
from grsn.envs.marl.base import MARLEnv
from grsn.envs.marl.mock_smac import MockSMAC

__all__ = ["MARLEnv", "MockSMAC"]
```

- [ ] **Step 4：创建 `tests/marl/__init__.py`**

空文件。

- [ ] **Step 5：创建 `tests/marl/test_mock_smac_api.py`**

```python
"""验证 MockSMAC 遵守 MARLEnv 接口。"""
import numpy as np
import pytest

from grsn.envs.marl import MARLEnv, MockSMAC


def test_mock_smac_is_marl_env():
    env = MockSMAC()
    assert isinstance(env, MARLEnv)


def test_env_info_contains_required_keys():
    env = MockSMAC()
    info = env.get_env_info()
    for k in ["n_agents", "n_actions", "obs_shape", "state_shape", "episode_limit"]:
        assert k in info


def test_reset_then_getters_return_correct_shapes():
    env = MockSMAC(n_agents=5, n_actions=7, obs_dim=11, state_dim=23)
    env.reset()
    assert env.get_obs().shape == (5, 11)
    assert env.get_state().shape == (23,)
    assert env.get_avail_actions().shape == (5, 7)
    assert env.get_obs().dtype == np.float32


def test_step_returns_triple_and_terminates_within_horizon():
    env = MockSMAC(episode_limit=10, seed=42)
    env.reset()
    steps = 0
    terminated = False
    while not terminated:
        actions = np.zeros(env.n_agents, dtype=np.int64)
        reward, terminated, info = env.step(actions)
        steps += 1
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)
    assert 20 <= steps + 0 or steps <= 10  # 兜底：horizon 在 [20, episode_limit] 之间
    # MockSMAC 的 _horizon 最小是 20，但 episode_limit=10 时随机会出错？看实现：
    # self._rng.randint(20, self.episode_limit + 1) — episode_limit=10 时范围 [20,11) 会抛
    # 所以此测试用 episode_limit >= 20


def test_step_terminates_respecting_horizon():
    env = MockSMAC(episode_limit=30, seed=7)
    env.reset()
    terminated = False
    count = 0
    while not terminated and count < 100:
        reward, terminated, info = env.step(np.zeros(env.n_agents, dtype=np.int64))
        count += 1
    assert terminated is True
    assert 20 <= count <= 30


def test_step_rejects_wrong_action_shape():
    env = MockSMAC(n_agents=4)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(np.zeros(3, dtype=np.int64))
```

- [ ] **Step 6：修掉 Step 5 里的自我矛盾测试**

`test_step_returns_triple_and_terminates_within_horizon` 的最后 assert 不对——`MockSMAC` 的 horizon 在 `[20, episode_limit]` 里，如果 `episode_limit=10` 会抛错。删掉这个测试（只留 `test_step_terminates_respecting_horizon`）。

重写 `tests/marl/test_mock_smac_api.py`，完整内容如下（替换 Step 5 的版本）：

```python
"""验证 MockSMAC 遵守 MARLEnv 接口。"""
import numpy as np
import pytest

from grsn.envs.marl import MARLEnv, MockSMAC


def test_mock_smac_is_marl_env():
    env = MockSMAC()
    assert isinstance(env, MARLEnv)


def test_env_info_contains_required_keys():
    env = MockSMAC()
    info = env.get_env_info()
    for k in ["n_agents", "n_actions", "obs_shape", "state_shape", "episode_limit"]:
        assert k in info


def test_reset_then_getters_return_correct_shapes():
    env = MockSMAC(n_agents=5, n_actions=7, obs_dim=11, state_dim=23)
    env.reset()
    assert env.get_obs().shape == (5, 11)
    assert env.get_state().shape == (23,)
    assert env.get_avail_actions().shape == (5, 7)
    assert env.get_obs().dtype == np.float32


def test_step_terminates_within_horizon_window():
    """MockSMAC 的 horizon 在 [20, episode_limit] 间采样。"""
    env = MockSMAC(episode_limit=30, seed=7)
    env.reset()
    terminated = False
    count = 0
    while not terminated and count < 100:
        reward, terminated, _ = env.step(np.zeros(env.n_agents, dtype=np.int64))
        count += 1
    assert terminated is True
    assert 20 <= count <= 30


def test_step_rejects_wrong_action_shape():
    env = MockSMAC(n_agents=4)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(np.zeros(3, dtype=np.int64))


def test_seed_makes_episode_deterministic():
    a = MockSMAC(seed=123)
    a.reset()
    obs_a = a.get_obs()
    b = MockSMAC(seed=123)
    b.reset()
    obs_b = b.get_obs()
    np.testing.assert_array_equal(obs_a, obs_b)
```

- [ ] **Step 7：运行测试**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -m pytest tests/marl/test_mock_smac_api.py -v
```

Expected: 全部 6 个测试 PASS。

- [ ] **Step 8：Commit**

```bash
git add grsn/envs/marl/ tests/marl/__init__.py tests/marl/test_mock_smac_api.py
git commit -m "新增 MARL 环境抽象：MARLEnv ABC + MockSMAC + 单测"
```

---

## Task 2：Mixer 单调混合网络 + 单测

**Files:**
- Create: `grsn/algorithms/marl/mixer.py`
- Create: `tests/marl/test_mixer.py`

- [ ] **Step 1：创建 `grsn/algorithms/marl/mixer.py`**

```python
"""QMIX 单调混合网络。

参考 Rashid et al. ICML 2018 "QMIX"。通过 hypernetwork 从 global state 生成
mixer 权重，再用 abs() 保证 ∂Q_tot / ∂Q_i ≥ 0（值分解的单调性约束）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """把 n_agents 个 per-agent Q 值混合成 joint Q_tot。

    Args:
        n_agents: agent 数（对同构 SMAC 地图是 units 数）
        state_dim: 全局 state 的维度
        embed_dim: mixer 中间层宽度（pymarl2 默认 32）
        hypernet_layers: hypernet 的层数（2 代表一个隐藏层，pymarl2 默认）
        hypernet_embed: hypernet 隐藏层宽度（pymarl2 默认 64）
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        def _build_hypernet(out_dim: int) -> nn.Module:
            if hypernet_layers == 1:
                return nn.Linear(state_dim, out_dim)
            elif hypernet_layers == 2:
                return nn.Sequential(
                    nn.Linear(state_dim, hypernet_embed),
                    nn.ReLU(inplace=True),
                    nn.Linear(hypernet_embed, out_dim),
                )
            else:
                raise ValueError(f"hypernet_layers must be 1 or 2, got {hypernet_layers}")

        # 第 1 层 mix：W1 形状 (n_agents, embed_dim)，b1 形状 (embed_dim,)
        self.hyper_w1 = _build_hypernet(n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # 第 2 层 mix：W2 形状 (embed_dim, 1)，b2 是 state 过 MLP 的标量偏置
        self.hyper_w2 = _build_hypernet(embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: (B, T, n_agents) — 每个 agent 选择动作的 Q 值
            states:   (B, T, state_dim)

        Returns:
            q_tot: (B, T, 1)
        """
        B, T, N = agent_qs.shape
        assert N == self.n_agents
        # 把 B 和 T 展平成一维以便通过 Linear
        agent_qs = agent_qs.reshape(B * T, 1, N)          # (BT, 1, N)
        states = states.reshape(B * T, self.state_dim)    # (BT, S)

        # 第一层
        w1 = torch.abs(self.hyper_w1(states))             # (BT, N*embed)
        w1 = w1.view(B * T, N, self.embed_dim)            # (BT, N, embed)
        b1 = self.hyper_b1(states).view(B * T, 1, self.embed_dim)  # (BT, 1, embed)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)      # (BT, 1, embed)

        # 第二层
        w2 = torch.abs(self.hyper_w2(states))             # (BT, embed)
        w2 = w2.view(B * T, self.embed_dim, 1)            # (BT, embed, 1)
        b2 = self.hyper_b2(states).view(B * T, 1, 1)      # (BT, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2                # (BT, 1, 1)
        return q_tot.view(B, T, 1)
```

- [ ] **Step 2：创建 `tests/marl/test_mixer.py`**

```python
"""Mixer 形状与单调性测试。"""
import torch

from grsn.algorithms.marl.mixer import QMixer


def test_mixer_output_shape():
    mixer = QMixer(n_agents=4, state_dim=32, embed_dim=16)
    agent_qs = torch.randn(2, 5, 4)   # (B, T, n_agents)
    states = torch.randn(2, 5, 32)
    q_tot = mixer(agent_qs, states)
    assert q_tot.shape == (2, 5, 1)


def test_mixer_monotonicity_increasing_any_agent_q_never_decreases_q_tot():
    """论文核心约束：∂Q_tot / ∂Q_i ≥ 0。"""
    torch.manual_seed(0)
    mixer = QMixer(n_agents=4, state_dim=8, embed_dim=16)
    base_q = torch.randn(1, 1, 4)
    state = torch.randn(1, 1, 8)
    q_base = mixer(base_q, state).item()

    # 逐个 agent 把 Q 调高一点，看 Q_tot 不降
    for i in range(4):
        perturbed = base_q.clone()
        perturbed[0, 0, i] += 0.5
        q_new = mixer(perturbed, state).item()
        assert q_new >= q_base - 1e-5, (
            f"agent {i} Q 增加后 Q_tot 反而下降：{q_base} -> {q_new}"
        )


def test_mixer_monotonicity_random_perturbations():
    """多种随机扰动下单调性都成立。"""
    torch.manual_seed(42)
    mixer = QMixer(n_agents=3, state_dim=5, embed_dim=8)
    for _ in range(30):
        base_q = torch.randn(1, 1, 3)
        state = torch.randn(1, 1, 5)
        delta = torch.relu(torch.randn(1, 1, 3))  # 非负扰动
        q_base = mixer(base_q, state).item()
        q_up = mixer(base_q + delta, state).item()
        assert q_up >= q_base - 1e-5


def test_mixer_gradient_flows_to_hypernet():
    mixer = QMixer(n_agents=4, state_dim=8)
    agent_qs = torch.randn(1, 1, 4, requires_grad=True)
    states = torch.randn(1, 1, 8)
    q_tot = mixer(agent_qs, states)
    q_tot.sum().backward()
    # 所有 hypernet 参数都要有梯度
    for name, p in mixer.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
```

- [ ] **Step 3：运行测试**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -m pytest tests/marl/test_mixer.py -v
```

Expected: 4 个测试 PASS。

- [ ] **Step 4：Commit**

```bash
git add grsn/algorithms/marl/mixer.py tests/marl/test_mixer.py
git commit -m "新增 QMIX 单调混合网络 + 单测（单调性 + 形状 + 梯度）"
```

---

## Task 3：AgentNetwork（参数共享 agent 网络）+ 单测

**Files:**
- Create: `grsn/algorithms/marl/agent_network.py`
- Create: `tests/marl/test_agent_network.py`

- [ ] **Step 1：创建 `grsn/algorithms/marl/agent_network.py`**

```python
"""参数共享的 MARL agent 网络。

所有 agent 共用一份权重；把 n_agents 折进 batch 维（B×n_agents）作为 batched RNN 输入。
RNN backbone 通过 rnn_type 字符串挑选：
- "gru"：标准 nn.GRU
- "GRSN" / "LIF" / "GRSNwoTAP" / "LIFwoTAP"：grsn.policies.rlifs.REGISTRY
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from grsn.policies.rlifs import REGISTRY as _SNN_REGISTRY


def _build_rnn(rnn_type: str, input_size: int, hidden_size: int, num_layers: int):
    """构造 RNN 并返回 (rnn, state_size_per_layer)。"""
    if rnn_type == "gru":
        rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=False)
        state_size_per_layer = hidden_size
    elif rnn_type in _SNN_REGISTRY:
        rnn = _SNN_REGISTRY[rnn_type](input_size, hidden_size, num_layers)
        state_size_per_layer = rnn.state_size_per_layer
    else:
        raise ValueError(
            f"unknown rnn_type: {rnn_type!r}. "
            f"expected 'gru' or one of {list(_SNN_REGISTRY.keys())}"
        )
    return rnn, state_size_per_layer


class AgentNetwork(nn.Module):
    """obs → embed → RNN → Q-head。参数在所有 agent 间共享。

    Args:
        obs_dim: 每个 agent 的 obs 维度
        n_actions: 动作空间大小（离散）
        rnn_type: "gru" / "GRSN" / "LIF" / "GRSNwoTAP" / "LIFwoTAP"
        rnn_hidden_size: RNN 隐层宽度
        obs_embed_size: obs embedding 输出维度（ReLU 激活）
        num_layers: RNN 堆叠层数（通常 1）
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        rnn_type: str,
        rnn_hidden_size: int = 64,
        obs_embed_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers

        self.obs_embed = nn.Linear(obs_dim, obs_embed_size)
        self.rnn, self.state_size_per_layer = _build_rnn(
            rnn_type, obs_embed_size, rnn_hidden_size, num_layers
        )
        self.q_head = nn.Linear(rnn_hidden_size, n_actions)

    def init_hidden(self, batch_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
        """分配 (num_layers, B, state_size_per_layer) 的零初始 state。"""
        return torch.zeros(
            self.num_layers, batch_size, self.state_size_per_layer,
            device=device, dtype=dtype,
        )

    def forward(
        self, obs: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:   (T, B, obs_dim) — B 一般是 batch_size × n_agents
            state: (num_layers, B, state_size_per_layer)

        Returns:
            q_values: (T, B, n_actions)
            new_state: (num_layers, B, state_size_per_layer)
        """
        x = F.relu(self.obs_embed(obs))
        rnn_out, new_state = self.rnn(x, state)
        # rnn_out 形状对 nn.GRU 是 (T, B, H)；对 rlifs RNN 也是 (T, B, H)
        q_values = self.q_head(rnn_out)
        return q_values, new_state
```

- [ ] **Step 2：创建 `tests/marl/test_agent_network.py`**

```python
"""AgentNetwork 单测：参数共享、RNN 切换、GRSN state 连续性。"""
import pytest
import torch

from grsn.algorithms.marl.agent_network import AgentNetwork


@pytest.mark.parametrize("rnn_type,expected_state_size", [
    ("gru", 64),
    ("LIF", 64),
    ("LIFwoTAP", 64),
    ("GRSN", 3 * 64),
    ("GRSNwoTAP", 3 * 64),
])
def test_output_shapes_per_rnn_type(rnn_type, expected_state_size):
    net = AgentNetwork(obs_dim=10, n_actions=5, rnn_type=rnn_type, rnn_hidden_size=64)
    B = 8
    obs = torch.randn(3, B, 10)
    state = net.init_hidden(B)
    q, new_state = net(obs, state)
    assert q.shape == (3, B, 5)
    assert new_state.shape == (1, B, expected_state_size)


def test_parameter_sharing_identical_obs_gives_identical_q():
    """把 n_agents 折进 batch 维：相同 obs → 相同 Q（因为权重共享）。"""
    torch.manual_seed(0)
    net = AgentNetwork(obs_dim=6, n_actions=4, rnn_type="gru", rnn_hidden_size=16)
    n_agents = 5
    B_actual = 2  # 2 个 env，每个 5 agent
    B_flat = B_actual * n_agents
    # 同一个 obs 重复 n_agents 次
    one_obs = torch.randn(3, 1, 6)
    obs = one_obs.expand(3, B_flat, 6).contiguous()
    state = net.init_hidden(B_flat)
    q, _ = net(obs, state)
    # 所有 batch 行的 Q 应该相同
    q0 = q[:, 0, :]
    for i in range(1, B_flat):
        torch.testing.assert_close(q[:, i, :], q0)


def test_grsn_state_propagation_split_equals_full():
    """GRSN 外部 state：两次 T=1 带 state 传递应该和一次 T=2 一次性前向等价。"""
    torch.manual_seed(0)
    net = AgentNetwork(obs_dim=4, n_actions=3, rnn_type="GRSN", rnn_hidden_size=8)
    B = 2
    obs_full = torch.randn(2, B, 4)
    state0 = net.init_hidden(B)

    # 一次性 T=2
    q_full, _ = net(obs_full, state0)

    # 分两步 T=1
    q_step0, state1 = net(obs_full[0:1], state0)
    q_step1, _ = net(obs_full[1:2], state1)
    q_split = torch.cat([q_step0, q_step1], dim=0)

    torch.testing.assert_close(q_full, q_split, atol=1e-5, rtol=1e-5)


def test_unknown_rnn_type_raises():
    with pytest.raises(ValueError, match="unknown rnn_type"):
        AgentNetwork(obs_dim=4, n_actions=3, rnn_type="lstm")
```

- [ ] **Step 3：运行测试**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -m pytest tests/marl/test_agent_network.py -v
```

Expected: 4 组 + 其他 = 7 个测试 PASS（`test_output_shapes_per_rnn_type` parametrize 了 5 个 rnn_type）。总共 5 + 3 = 8 个。

- [ ] **Step 4：Commit**

```bash
git add grsn/algorithms/marl/agent_network.py tests/marl/test_agent_network.py
git commit -m "新增 MARL AgentNetwork（参数共享，RNN backbone 可插拔）+ 单测"
```

---

## Task 4：EpisodeBuffer（CTDE 整 episode replay）+ 单测

**Files:**
- Create: `grsn/buffers/episode_buffer.py`
- Create: `tests/marl/test_episode_buffer.py`

- [ ] **Step 1：创建 `grsn/buffers/episode_buffer.py`**

```python
"""CTDE 整 episode replay buffer。

存储格式：每个 slot 存一整条 episode，所有字段 pad 到 episode_limit + 1 长度
（+1 是因为要存 next_obs/next_state）。用 filled_mask 区分真实 step 与 padding。

Sample 回的 batch 维度：(B, T, ...)，T = episode_limit + 1。
"""
from typing import Dict

import numpy as np
import torch


class EpisodeBuffer:
    """固定大小循环 buffer。每个 slot = 一整条 episode。"""

    def __init__(
        self,
        buffer_size: int,
        episode_limit: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
    ):
        self.buffer_size = buffer_size
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions

        # T = episode_limit + 1 （最后一格存 terminal 后的 next_obs）
        T = episode_limit + 1
        self.T = T

        # 预分配 numpy buffer
        self.obs = np.zeros((buffer_size, T, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((buffer_size, T, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, T, n_agents), dtype=np.int64)
        self.avail_actions = np.zeros((buffer_size, T, n_agents, n_actions), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, T, 1), dtype=np.float32)
        self.terminated = np.zeros((buffer_size, T, 1), dtype=np.float32)
        self.filled = np.zeros((buffer_size, T, 1), dtype=np.float32)

        self._idx = 0
        self._n_stored = 0

    def insert(self, episode: Dict[str, np.ndarray]) -> None:
        """插入一条 episode。

        episode dict 必须包含以下 key，每个 array 第一维 L ≤ episode_limit：
            obs:           (L+1, n_agents, obs_dim)
            state:         (L+1, state_dim)
            actions:       (L,   n_agents)
            avail_actions: (L+1, n_agents, n_actions)
            rewards:       (L,   1)
            terminated:    (L,   1)
        """
        L = episode["rewards"].shape[0]
        assert L <= self.episode_limit, f"episode 长度 {L} 超过上限 {self.episode_limit}"
        slot = self._idx

        self.obs[slot, : L + 1] = episode["obs"]
        self.state[slot, : L + 1] = episode["state"]
        self.actions[slot, :L] = episode["actions"]
        self.avail_actions[slot, : L + 1] = episode["avail_actions"]
        self.rewards[slot, :L] = episode["rewards"]
        self.terminated[slot, :L] = episode["terminated"]
        # filled：前 L 个 transition（即 L 个 reward-terminated 对）是 valid
        self.filled[slot].fill(0.0)
        self.filled[slot, :L] = 1.0

        # 超出 L 的 slot 清零（避免上一次 episode 残留）
        self.obs[slot, L + 1 :] = 0.0
        self.state[slot, L + 1 :] = 0.0
        self.actions[slot, L:] = 0
        self.avail_actions[slot, L + 1 :] = 0
        self.rewards[slot, L:] = 0.0
        self.terminated[slot, L:] = 0.0

        self._idx = (self._idx + 1) % self.buffer_size
        self._n_stored = min(self._n_stored + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机不放回采样 batch_size 个 episode，返回 torch tensor dict。"""
        assert batch_size <= self._n_stored, (
            f"请求 {batch_size} 条，但 buffer 只有 {self._n_stored} 条"
        )
        idx = np.random.choice(self._n_stored, size=batch_size, replace=False)

        def _to_torch(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[idx].copy())

        return {
            "obs": _to_torch(self.obs),
            "state": _to_torch(self.state),
            "actions": _to_torch(self.actions),
            "avail_actions": _to_torch(self.avail_actions),
            "rewards": _to_torch(self.rewards),
            "terminated": _to_torch(self.terminated),
            "filled": _to_torch(self.filled),
        }

    def __len__(self) -> int:
        return self._n_stored
```

- [ ] **Step 2：创建 `tests/marl/test_episode_buffer.py`**

```python
"""EpisodeBuffer 单测：roundtrip、padding、容量循环。"""
import numpy as np
import pytest
import torch

from grsn.buffers.episode_buffer import EpisodeBuffer


def _make_episode(L, n_agents, obs_dim, state_dim, n_actions, rng):
    return {
        "obs": rng.randn(L + 1, n_agents, obs_dim).astype(np.float32),
        "state": rng.randn(L + 1, state_dim).astype(np.float32),
        "actions": rng.randint(0, n_actions, size=(L, n_agents)).astype(np.int64),
        "avail_actions": np.ones((L + 1, n_agents, n_actions), dtype=np.int64),
        "rewards": rng.randn(L, 1).astype(np.float32),
        "terminated": np.concatenate(
            [np.zeros((L - 1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)]
        ),
    }


def test_insert_then_sample_roundtrip():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=10,
        n_agents=3, obs_dim=5, state_dim=7, n_actions=4,
    )
    rng = np.random.RandomState(0)
    ep = _make_episode(8, 3, 5, 7, 4, rng)
    buf.insert(ep)
    assert len(buf) == 1
    batch = buf.sample(1)
    assert batch["obs"].shape == (1, 11, 3, 5)  # T = episode_limit + 1 = 11
    assert batch["state"].shape == (1, 11, 7)
    assert batch["actions"].shape == (1, 11, 3)
    # 前 L 个 reward 应该和插入的一致
    np.testing.assert_allclose(batch["rewards"][0, :8].numpy(), ep["rewards"], atol=0)


def test_padding_mask_marks_only_real_transitions():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=10,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    L = 5
    ep = _make_episode(L, 2, 3, 4, 5, rng)
    buf.insert(ep)
    batch = buf.sample(1)
    filled = batch["filled"][0, :, 0].numpy()  # shape (T,) = (11,)
    assert filled[:L].sum() == L  # 前 L 个是 1
    assert filled[L:].sum() == 0  # 之后全 0


def test_buffer_overwrites_oldest_when_full():
    buf = EpisodeBuffer(
        buffer_size=2, episode_limit=5,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    ep1 = _make_episode(3, 2, 3, 4, 5, rng)
    ep2 = _make_episode(3, 2, 3, 4, 5, rng)
    ep3 = _make_episode(3, 2, 3, 4, 5, rng)
    buf.insert(ep1)
    buf.insert(ep2)
    buf.insert(ep3)  # 覆盖 ep1
    assert len(buf) == 2


def test_sample_rejects_too_many():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=5,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    buf.insert(_make_episode(3, 2, 3, 4, 5, rng))
    with pytest.raises(AssertionError):
        buf.sample(2)


def test_insert_episode_longer_than_limit_fails():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=5,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    with pytest.raises(AssertionError):
        buf.insert(_make_episode(6, 2, 3, 4, 5, rng))
```

- [ ] **Step 3：运行测试**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -m pytest tests/marl/test_episode_buffer.py -v
```

Expected: 5 个测试 PASS。

- [ ] **Step 4：Commit**

```bash
git add grsn/buffers/episode_buffer.py tests/marl/test_episode_buffer.py
git commit -m "新增 CTDE EpisodeBuffer + 单测（roundtrip / padding / 容量循环）"
```

---

## Task 5：QMIX 训练器

**Files:**
- Create: `grsn/algorithms/marl/qmix.py`
- Modify: `grsn/algorithms/marl/__init__.py`（从占位替换为真实 export）

- [ ] **Step 1：创建 `grsn/algorithms/marl/qmix.py`**

```python
"""QMIX 训练器。

Q-learning + 单调 mixer 的联合训练：
1. 对 batch 里每条 episode 前向 agent_net → Q_eval（per-agent per-action）
2. 对同一 batch 前向 target_agent_net → Q_target（next obs）
3. 选取 eval 动作 = batch['actions']；target 动作 = argmax over avail_actions
4. mixer 把 per-agent Q 组合成 joint Q_tot
5. TD 目标 y = r + γ(1-done)·target_mixer(Q_target_max, next_state)
6. loss = mean((Q_tot_eval - y.detach())^2 * filled_mask) / filled_mask.sum()
"""
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn


class QMIX:
    def __init__(
        self,
        agent_net: nn.Module,
        mixer: nn.Module,
        lr: float = 5e-4,
        gamma: float = 0.99,
        grad_clip: float = 10.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_net = agent_net.to(device)
        self.mixer = mixer.to(device)
        self.target_agent_net = deepcopy(self.agent_net).to(device)
        self.target_mixer = deepcopy(self.mixer).to(device)
        for p in self.target_agent_net.parameters():
            p.requires_grad = False
        for p in self.target_mixer.parameters():
            p.requires_grad = False

        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = device

        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _rollout_agent(self, agent_net, obs, batch_size, n_agents):
        """对整条 episode 前向 agent_net。

        obs: (B, T, n_agents, obs_dim) → 折 n_agents 进 batch → (T, B*n_agents, obs_dim)
        返回 Q: (B, T, n_agents, n_actions)
        """
        B, T, N, D = obs.shape
        obs_flat = obs.permute(1, 0, 2, 3).reshape(T, B * N, D)
        init_state = agent_net.init_hidden(B * N, device=self.device, dtype=obs.dtype)
        q_flat, _ = agent_net(obs_flat, init_state)  # (T, B*N, n_actions)
        A = q_flat.shape[-1]
        q = q_flat.reshape(T, B, N, A).permute(1, 0, 2, 3).contiguous()
        return q  # (B, T, n_agents, n_actions)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # 移到 device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        obs = batch["obs"]                   # (B, T, N, obs_dim)
        state = batch["state"]               # (B, T, state_dim)
        actions = batch["actions"]           # (B, T, N)
        avail = batch["avail_actions"]       # (B, T, N, n_actions)
        rewards = batch["rewards"]           # (B, T, 1)
        terminated = batch["terminated"]     # (B, T, 1)
        filled = batch["filled"]             # (B, T, 1)

        B, T, N, _ = obs.shape

        # Q_eval over all timesteps
        q_eval_all = self._rollout_agent(self.agent_net, obs, B, N)  # (B, T, N, A)

        # Q_target over all timesteps (no grad)
        with torch.no_grad():
            q_target_all = self._rollout_agent(self.target_agent_net, obs, B, N)

        # Q_eval 取已执行动作
        actions_unsq = actions.unsqueeze(-1)  # (B, T, N, 1)
        q_eval_chosen = q_eval_all.gather(dim=-1, index=actions_unsq).squeeze(-1)  # (B, T, N)

        # Q_target 取 argmax over avail actions（从 next step 开始）
        # 对 padding / terminal 后的 step，avail 全 0 → 需保护
        # 把被屏蔽的动作 Q 置为 -inf
        with torch.no_grad():
            q_target_masked = q_target_all.clone()
            q_target_masked[avail == 0] = -1e9
            q_target_max = q_target_masked.max(dim=-1).values  # (B, T, N)

        # 注意 TD：y_t = r_t + γ(1-done_t) * Q_tot_target(next_state_{t+1})
        # 所以 "target 输入" 应该是 t+1 时刻的 per-agent Q，但我们每步都算过了
        # 这里采用 pymarl 的做法：对 t 时刻的 loss 用 q_tot_eval[t] 与 r[t] + γ*q_tot_target[t+1]
        # 为实现这一点，只在 0..T-2 上计算损失

        q_tot_eval = self.mixer(q_eval_chosen, state).squeeze(-1)  # (B, T)
        with torch.no_grad():
            q_tot_target = self.target_mixer(q_target_max, state).squeeze(-1)  # (B, T)

        # TD target：y[:, :-1] = r[:, :-1] + γ (1-done[:, :-1]) * q_tot_target[:, 1:]
        rewards_s = rewards.squeeze(-1)       # (B, T)
        terminated_s = terminated.squeeze(-1) # (B, T)
        filled_s = filled.squeeze(-1)         # (B, T)
        y = rewards_s[:, :-1] + self.gamma * (1.0 - terminated_s[:, :-1]) * q_tot_target[:, 1:]
        td = q_tot_eval[:, :-1] - y.detach()

        mask = filled_s[:, :-1]
        num_valid = mask.sum().clamp(min=1.0)
        loss = ((td ** 2) * mask).sum() / num_valid

        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "q_tot_mean": q_tot_eval[:, :-1].mul(mask).sum().div(num_valid).item(),
            "y_mean": y.mul(mask).sum().div(num_valid).item(),
        }

    def target_update(self) -> None:
        """硬复制 eval → target。"""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
```

- [ ] **Step 2：替换 `grsn/algorithms/marl/__init__.py`**

```python
"""MARL 算法包（QMIX + SMAC 接入）。

复现论文 arXiv:2404.15597 (AAAI'25) 的 MARL 部分——在 QMIX agent 网络位置
插入 GRSN 作为 RNN backbone。

使用方法见 docs/MARL_EXTENSION.md 和 experiments/train_marl.py。
"""
from grsn.algorithms.marl.qmix import QMIX
from grsn.algorithms.marl.mixer import QMixer
from grsn.algorithms.marl.agent_network import AgentNetwork

__all__ = ["QMIX", "QMixer", "AgentNetwork"]
```

- [ ] **Step 3：冒烟测试 QMIX.train_step 能跑通**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -c "
import numpy as np
import torch
from grsn.algorithms.marl import QMIX, QMixer, AgentNetwork
from grsn.buffers.episode_buffer import EpisodeBuffer

torch.manual_seed(0); np.random.seed(0)
n_agents, n_actions, obs_dim, state_dim, episode_limit = 3, 5, 6, 8, 10

agent = AgentNetwork(obs_dim, n_actions, rnn_type='GRSN', rnn_hidden_size=16)
mixer = QMixer(n_agents, state_dim, embed_dim=8)
algo = QMIX(agent, mixer, lr=5e-4, gamma=0.99)

buf = EpisodeBuffer(buffer_size=8, episode_limit=episode_limit,
                    n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim, n_actions=n_actions)
rng = np.random.RandomState(0)
for _ in range(4):
    L = 7
    buf.insert({
        'obs': rng.randn(L+1, n_agents, obs_dim).astype(np.float32),
        'state': rng.randn(L+1, state_dim).astype(np.float32),
        'actions': rng.randint(0, n_actions, (L, n_agents)).astype(np.int64),
        'avail_actions': np.ones((L+1, n_agents, n_actions), dtype=np.int64),
        'rewards': rng.randn(L, 1).astype(np.float32),
        'terminated': np.concatenate([np.zeros((L-1,1)), np.ones((1,1))]).astype(np.float32),
    })

batch = buf.sample(2)
info = algo.train_step(batch)
print('train_step OK:', info)
algo.target_update()
print('target_update OK')
"
```

Expected: 最后打印 `train_step OK: {'loss': ..., 'grad_norm': ..., 'q_tot_mean': ..., 'y_mean': ...}` 和 `target_update OK`。

- [ ] **Step 4：Commit**

```bash
git add grsn/algorithms/marl/qmix.py grsn/algorithms/marl/__init__.py
git commit -m "新增 QMIX 训练器：TD loss + 单调 mixer + 硬 target update

- 支持任意 rnn_type（gru / GRSN / LIF / GRSNwoTAP / LIFwoTAP）
- batch 前向按 pymarl 习惯把 n_agents 折进 batch 维
- 对 padding 和 terminal 用 filled_mask 屏蔽
- target 动作 argmax 时把不可用动作置 -1e9"
```

---

## Task 6：SMACWrapper（延迟 import，本次不真实运行）

**Files:**
- Create: `grsn/envs/marl/smac_wrapper.py`
- Modify: `grsn/envs/marl/__init__.py`

- [ ] **Step 1：创建 `grsn/envs/marl/smac_wrapper.py`**

```python
"""真实 SMAC 环境的薄封装。

**运行时依赖：** 需要 StarCraft II 游戏二进制 + SMAC map pack + smac Python 包。
安装步骤见 docs/MARL_EXTENSION.md。本仓库默认不安装这些依赖——测试用
`MockSMAC` 替代。
"""
from typing import Dict, Tuple

import numpy as np

from grsn.envs.marl.base import MARLEnv


class SMACWrapper(MARLEnv):
    """把 smac.env.StarCraft2Env 适配到 MARLEnv 接口。

    Args:
        map_name: SMAC map 名，例如 "8m" / "2s3z" / "3s_vs_5z"
        seed: 随机种子
        **smac_kwargs: 额外 kwargs 传给 StarCraft2Env
    """

    def __init__(self, map_name: str, seed: int = 0, **smac_kwargs):
        try:
            from smac.env import StarCraft2Env
        except ImportError as e:
            raise ImportError(
                "smac 未安装。安装步骤：\n"
                "  1. pip install smac\n"
                "  2. 下载 StarCraft II（见 docs/MARL_EXTENSION.md）\n"
                "  3. 安装 SMAC map pack 到 $SC2PATH/Maps/SMAC_Maps/"
            ) from e

        self._env = StarCraft2Env(map_name=map_name, seed=seed, **smac_kwargs)
        env_info = self._env.get_env_info()
        self._env_info = env_info
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_dim = env_info["obs_shape"]
        self.state_dim = env_info["state_shape"]
        self.episode_limit = env_info["episode_limit"]

    def reset(self) -> None:
        self._env.reset()

    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        reward, terminated, info = self._env.step(actions)
        return float(reward), bool(terminated), dict(info)

    def get_obs(self) -> np.ndarray:
        return np.asarray(self._env.get_obs(), dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return np.asarray(self._env.get_state(), dtype=np.float32)

    def get_avail_actions(self) -> np.ndarray:
        return np.asarray(self._env.get_avail_actions(), dtype=np.int64)

    def get_env_info(self) -> Dict:
        return dict(self._env_info)

    def close(self) -> None:
        self._env.close()
```

- [ ] **Step 2：更新 `grsn/envs/marl/__init__.py`**

把原先的 2 个 export 扩展为 3 个（但 `SMACWrapper` 不在默认 `__all__` 里，避免顶层 import 就触发 smac 依赖——用户需要时手动导入）：

```python
"""MARL 环境包。

- MARLEnv：抽象基类
- MockSMAC：冒烟测试用的随机 env
- SMACWrapper：真实 SMAC 的薄封装（import 时才会 raise ImportError）

注意 SMACWrapper 没放进默认 __all__，以免 `from grsn.envs.marl import *`
触发 smac 依赖。需要时显式写 `from grsn.envs.marl.smac_wrapper import SMACWrapper`。
"""
from grsn.envs.marl.base import MARLEnv
from grsn.envs.marl.mock_smac import MockSMAC

__all__ = ["MARLEnv", "MockSMAC"]
```

- [ ] **Step 3：验证延迟 import 生效**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -c "
from grsn.envs.marl import MARLEnv, MockSMAC
print('OK: default import 不触发 smac 依赖')

# 显式导入 SMACWrapper 类本身也 OK（只在 __init__ 时才 import smac）
from grsn.envs.marl.smac_wrapper import SMACWrapper
print('OK: SMACWrapper 类可 import')

# 实例化才会触发缺失依赖报错
try:
    SMACWrapper('8m')
except ImportError as e:
    print('OK: 实例化时按预期 ImportError：', str(e).split(chr(10))[0])
"
```

Expected: 三行 `OK:` 都打印出来。

- [ ] **Step 4：Commit**

```bash
git add grsn/envs/marl/smac_wrapper.py grsn/envs/marl/__init__.py
git commit -m "新增 SMACWrapper：真实 SMAC env 的薄封装，延迟 import 避免默认触发依赖"
```

---

## Task 7：训练入口 `train_marl.py` + 两个 map 的 config

**Files:**
- Create: `experiments/train_marl.py`
- Create: `configs/marl/smac/8m/qmix_grsn.yml`
- Create: `configs/marl/smac/2s3z/qmix_grsn.yml`

- [ ] **Step 1：创建 `configs/marl/smac/8m/qmix_grsn.yml`**

```yaml
# 对应 SMAC map "8m" 的 QMIX + GRSN 配置
# 默认值来自 pymarl2 的 vanilla QMIX（论文 arXiv:2404.15597 未单独重写）

env:
  map_name: 8m
  n_agents: 8        # MockSMAC 用；真实 SMAC 会被 env.get_env_info() 覆盖
  n_actions: 14
  obs_dim: 80
  state_dim: 168
  episode_limit: 60

agent:
  rnn_type: GRSN     # gru / GRSN / LIF / GRSNwoTAP / LIFwoTAP
  rnn_hidden_size: 64
  obs_embed_size: 64
  num_layers: 1

mixer:
  embed_dim: 32
  hypernet_layers: 2
  hypernet_embed: 64

train:
  lr: 5.0e-4
  gamma: 0.99
  grad_clip: 10.0
  batch_size: 32            # 单位：episode
  buffer_size: 5000         # 单位：episode
  target_update_interval: 200  # 单位：训练迭代次数
  num_env_steps: 10000000   # 论文 10M；冒烟测试请用 --num_env_steps 覆盖

explore:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_anneal_time: 50000  # env steps

eval:
  interval_env_steps: 20000
  num_episodes: 32
```

- [ ] **Step 2：创建 `configs/marl/smac/2s3z/qmix_grsn.yml`**

```yaml
# SMAC map "2s3z"（5 个异构单位）的 QMIX + GRSN 配置
# 注意：2s3z 的 units 不同构（2 个 stalker + 3 个 zealot），但 pymarl2 的默认
# QMIX 仍对所有 agent 共享权重，性能够用；若要 per-type 共享需扩展 AgentNetwork。

env:
  map_name: 2s3z
  n_agents: 5
  n_actions: 11
  obs_dim: 80        # 真实值需 smac env.get_obs_size() 确认
  state_dim: 120     # 真实值需 smac env.get_state_size() 确认
  episode_limit: 120

agent:
  rnn_type: GRSN
  rnn_hidden_size: 64
  obs_embed_size: 64
  num_layers: 1

mixer:
  embed_dim: 32
  hypernet_layers: 2
  hypernet_embed: 64

train:
  lr: 5.0e-4
  gamma: 0.99
  grad_clip: 10.0
  batch_size: 32
  buffer_size: 5000
  target_update_interval: 200
  num_env_steps: 10000000

explore:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_anneal_time: 50000

eval:
  interval_env_steps: 20000
  num_episodes: 32
```

- [ ] **Step 3：创建 `experiments/train_marl.py`**

```python
"""MARL 训练入口：QMIX + GRSN on SMAC / MockSMAC。

示例：
    # 冒烟（Mock env, CPU, 1000 env steps）
    python experiments/train_marl.py --env MockSMAC --map 8m \
        --rnn_type GRSN --seed 0 --cuda -1 --num_env_steps 1000

    # 真实 SMAC（需先装 SC2 + smac，见 docs/MARL_EXTENSION.md）
    python experiments/train_marl.py --env SMAC --map 8m \
        --rnn_type GRSN --seed 0 --cuda 0 --num_env_steps 10000000
"""
import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
from ruamel.yaml import YAML

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grsn.algorithms.marl import QMIX, QMixer, AgentNetwork  # noqa: E402
from grsn.buffers.episode_buffer import EpisodeBuffer  # noqa: E402
from grsn.envs.marl import MARLEnv, MockSMAC  # noqa: E402


def make_env(name: str, map_name: str, seed: int, cfg_env: Dict) -> MARLEnv:
    if name == "MockSMAC":
        return MockSMAC(
            n_agents=cfg_env["n_agents"],
            n_actions=cfg_env["n_actions"],
            obs_dim=cfg_env["obs_dim"],
            state_dim=cfg_env["state_dim"],
            episode_limit=cfg_env["episode_limit"],
            seed=seed,
        )
    if name == "SMAC":
        from grsn.envs.marl.smac_wrapper import SMACWrapper
        return SMACWrapper(map_name=map_name, seed=seed)
    raise ValueError(f"unknown env: {name!r}")


def epsilon_schedule(step: int, eps_start: float, eps_end: float, anneal: int) -> float:
    frac = min(step / max(anneal, 1), 1.0)
    return eps_start + frac * (eps_end - eps_start)


@torch.no_grad()
def rollout_one_episode(env: MARLEnv, agent_net: AgentNetwork, epsilon: float,
                        device: torch.device, rng: np.random.RandomState) -> Dict:
    env.reset()
    info = env.get_env_info()
    n_agents = info["n_agents"]
    n_actions = info["n_actions"]
    episode_limit = info["episode_limit"]

    obs_list, state_list, actions_list = [], [], []
    avail_list, reward_list, term_list = [], [], []

    # per-agent state：(num_layers, n_agents, state_size_per_layer)
    state_t = agent_net.init_hidden(n_agents, device=device)

    terminated = False
    for _ in range(episode_limit):
        obs = env.get_obs()           # (n_agents, obs_dim)
        state = env.get_state()       # (state_dim,)
        avail = env.get_avail_actions()  # (n_agents, n_actions)

        obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)  # (1, n_agents, obs_dim)
        q, state_t = agent_net(obs_tensor, state_t)
        q = q.squeeze(0)  # (n_agents, n_actions)
        q_masked = q.clone()
        q_masked[torch.from_numpy(avail).to(device) == 0] = -1e9
        greedy = q_masked.argmax(dim=-1).cpu().numpy()

        # ε-greedy
        random_actions = np.array([
            rng.choice(np.flatnonzero(avail[a])) for a in range(n_agents)
        ])
        use_random = rng.rand(n_agents) < epsilon
        actions = np.where(use_random, random_actions, greedy).astype(np.int64)

        reward, terminated, _ = env.step(actions)

        obs_list.append(obs)
        state_list.append(state)
        actions_list.append(actions)
        avail_list.append(avail)
        reward_list.append([reward])
        term_list.append([1.0 if terminated else 0.0])

        if terminated:
            break

    # 最后再收一次 obs/state/avail 作为 next state
    obs_list.append(env.get_obs())
    state_list.append(env.get_state())
    avail_list.append(env.get_avail_actions())

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "state": np.asarray(state_list, dtype=np.float32),
        "actions": np.asarray(actions_list, dtype=np.int64),
        "avail_actions": np.asarray(avail_list, dtype=np.int64),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "terminated": np.asarray(term_list, dtype=np.float32),
        "length": len(reward_list),
    }


def main():
    parser = argparse.ArgumentParser(description="QMIX + GRSN on SMAC/MockSMAC")
    parser.add_argument("--env", type=str, default="MockSMAC", choices=["MockSMAC", "SMAC"])
    parser.add_argument("--map", type=str, default="8m")
    parser.add_argument("--rnn_type", type=str, default="GRSN",
                        choices=["gru", "LIF", "LIFwoTAP", "GRSN", "GRSNwoTAP"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=-1, help="-1 for CPU")
    parser.add_argument("--config", type=str, default=None,
                        help="path to yaml config; default = configs/marl/smac/<map>/qmix_grsn.yml")
    parser.add_argument("--num_env_steps", type=int, default=None,
                        help="override config's train.num_env_steps (for smoke tests)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_path = args.config or f"configs/marl/smac/{args.map}/qmix_grsn.yml"
    yaml = YAML()
    with open(cfg_path) as f:
        cfg = yaml.load(f)

    if args.num_env_steps is not None:
        cfg["train"]["num_env_steps"] = int(args.num_env_steps)

    device = torch.device("cpu") if args.cuda < 0 else torch.device(f"cuda:{args.cuda}")

    env = make_env(args.env, args.map, args.seed, cfg["env"])
    env_info = env.get_env_info()
    # 真实 SMAC 的维度由 env 决定，会覆盖 config
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]
    episode_limit = env_info["episode_limit"]

    agent_net = AgentNetwork(
        obs_dim=obs_dim, n_actions=n_actions,
        rnn_type=cfg["agent"]["rnn_type"] if args.rnn_type is None else args.rnn_type,
        rnn_hidden_size=cfg["agent"]["rnn_hidden_size"],
        obs_embed_size=cfg["agent"]["obs_embed_size"],
        num_layers=cfg["agent"]["num_layers"],
    )
    mixer = QMixer(
        n_agents=n_agents, state_dim=state_dim,
        embed_dim=cfg["mixer"]["embed_dim"],
        hypernet_layers=cfg["mixer"]["hypernet_layers"],
        hypernet_embed=cfg["mixer"]["hypernet_embed"],
    )
    algo = QMIX(
        agent_net, mixer,
        lr=cfg["train"]["lr"],
        gamma=cfg["train"]["gamma"],
        grad_clip=cfg["train"]["grad_clip"],
        device=device,
    )
    buffer = EpisodeBuffer(
        buffer_size=cfg["train"]["buffer_size"],
        episode_limit=episode_limit,
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
    )

    rng = np.random.RandomState(args.seed)
    env_steps = 0
    train_iter = 0
    target_update_interval = cfg["train"]["target_update_interval"]
    batch_size = cfg["train"]["batch_size"]
    eps_cfg = cfg["explore"]
    num_env_steps = cfg["train"]["num_env_steps"]

    print(f"[train_marl] env={args.env} map={args.map} rnn={args.rnn_type} "
          f"device={device} num_env_steps={num_env_steps}")
    t0 = time.time()

    while env_steps < num_env_steps:
        epsilon = epsilon_schedule(
            env_steps,
            eps_cfg["epsilon_start"], eps_cfg["epsilon_end"], eps_cfg["epsilon_anneal_time"],
        )
        episode = rollout_one_episode(env, algo.agent_net, epsilon, device, rng)
        ep_len = episode.pop("length")
        env_steps += ep_len
        buffer.insert(episode)

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            info = algo.train_step(batch)
            train_iter += 1
            if train_iter % target_update_interval == 0:
                algo.target_update()
            if train_iter % 10 == 0 or env_steps >= num_env_steps:
                elapsed = time.time() - t0
                print(f"[train_marl] env_steps={env_steps} iter={train_iter} "
                      f"eps={epsilon:.3f} loss={info['loss']:.4f} "
                      f"q_tot_mean={info['q_tot_mean']:.3f} elapsed={elapsed:.1f}s")

    env.close()
    print(f"[train_marl] done. total env_steps={env_steps} train_iters={train_iter}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：Commit**

```bash
mkdir -p configs/marl/smac/8m configs/marl/smac/2s3z
git add configs/marl/experiments/train_marl.py 2>/dev/null || true
git add configs/marl/ experiments/train_marl.py
git commit -m "新增 MARL 训练入口 train_marl.py + 8m / 2s3z 两份 QMIX+GRSN 配置"
```

---

## Task 8：端到端冒烟测试（MockSMAC）

**Files:**
- Create: `tests/marl/test_qmix_smoke.py`

- [ ] **Step 1：创建 `tests/marl/test_qmix_smoke.py`**

```python
"""QMIX 端到端冒烟测试：在 MockSMAC 上跑 train_marl 若干 env steps，验证无异常。"""
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _run_train(rnn_type: str, num_env_steps: int = 500) -> str:
    """运行 train_marl.py 若干步，返回 stdout。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    result = subprocess.run(
        [
            sys.executable, str(REPO / "experiments" / "train_marl.py"),
            "--env", "MockSMAC",
            "--map", "8m",
            "--rnn_type", rnn_type,
            "--seed", "0",
            "--cuda", "-1",
            "--num_env_steps", str(num_env_steps),
        ],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"train_marl failed (rnn={rnn_type}):\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result.stdout


def test_smoke_grsn():
    out = _run_train("GRSN", num_env_steps=500)
    assert "[train_marl] done" in out
    assert "loss=" in out, f"no training step occurred; output:\n{out}"


def test_smoke_gru():
    out = _run_train("gru", num_env_steps=500)
    assert "[train_marl] done" in out


def test_smoke_grsn_wo_tap():
    out = _run_train("GRSNwoTAP", num_env_steps=500)
    assert "[train_marl] done" in out
```

- [ ] **Step 2：运行冒烟测试**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -m pytest tests/marl/test_qmix_smoke.py -v
```

Expected: 3 个测试 PASS（每个约 30-60s）。

如果超时或挂，检查：
- `buffer.sample(batch_size)` 是否在 buffer 未满时被调用（train_loop 写法问题）
- GRSN state shape 是否正确
- `rollout_one_episode` 里动作 shape 是否对

如果测试失败，定位根因并修复；修复也要 commit（单独 commit `修复 <具体问题>`）。

- [ ] **Step 3：手工再跑一次确认输出格式**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python experiments/train_marl.py \
    --env MockSMAC --map 8m --rnn_type GRSN --seed 0 --cuda -1 --num_env_steps 800 2>&1 | tail -20
```

Expected：最后一行 `[train_marl] done. total env_steps=... train_iters=...`，中间有若干 `loss=` 行。

- [ ] **Step 4：Commit**

```bash
git add tests/marl/test_qmix_smoke.py
git commit -m "新增 MARL 端到端冒烟测试：GRSN / gru / GRSNwoTAP 各跑 500 env steps"
```

---

## Task 9：更新 `docs/MARL_EXTENSION.md` 从 roadmap 改为 guide

**Files:**
- Overwrite: `docs/MARL_EXTENSION.md`

- [ ] **Step 1：覆写 `docs/MARL_EXTENSION.md`**

```markdown
# MARL (QMIX + SMAC + GRSN) 使用指南

本仓库已实现论文 arXiv:2404.15597 (AAAI'25, Qin et al.) MARL 部分的**代码
与 Mock 冒烟测试**。真实 SMAC 训练依赖 StarCraft II 游戏二进制，需要用户自行部署。

## 已实现的组件

| 组件 | 位置 | 说明 |
|---|---|---|
| QMIX 训练器 | `grsn/algorithms/marl/qmix.py` | TD loss + 单调 mixer + 硬 target update |
| 单调 Mixer | `grsn/algorithms/marl/mixer.py` | Hypernetwork + abs() |
| Agent 网络 | `grsn/algorithms/marl/agent_network.py` | 参数共享，RNN backbone 可插拔 |
| Episode Buffer | `grsn/buffers/episode_buffer.py` | CTDE 整 episode replay + padding mask |
| MARLEnv ABC | `grsn/envs/marl/base.py` | 接口：reset/step/get_obs/get_state/get_avail_actions |
| MockSMAC | `grsn/envs/marl/mock_smac.py` | 随机 env，冒烟测试用 |
| SMACWrapper | `grsn/envs/marl/smac_wrapper.py` | 真实 SMAC 的薄封装（延迟 import） |
| 训练入口 | `experiments/train_marl.py` | CLI + yaml config |
| 配置 | `configs/marl/smac/{8m,2s3z}/qmix_grsn.yml` | 论文对齐的超参默认 |

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

**Linux**（headless 训练推荐）：

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
```

- [ ] **Step 2：Commit**

```bash
git add docs/MARL_EXTENSION.md
git commit -m "更新 MARL 使用指南：从 roadmap 改为已实现组件 + SC2 安装步骤 + 快速开始"
```

---

## Task 10：最终复核

**Files:** 只执行，不修改。

- [ ] **Step 1：跑全部测试**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: 之前 15 个神经元测试 + 新增的 MARL 测试全部 PASS（预计 30+ 测试）。

- [ ] **Step 2：git log 复查**

```bash
git log --oneline -15
```

Expected: 看到 7-10 个中文 commit，顺序覆盖 env base → mixer → agent_net → buffer → qmix → smac wrapper → train_marl → smoke test → docs。

- [ ] **Step 3：最终 smoke 跑一次**

```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. timeout 120 python experiments/train_marl.py \
    --env MockSMAC --map 8m --rnn_type GRSN --seed 0 --cuda -1 --num_env_steps 1000 2>&1 | tail -10
```

Expected: 最后打印 `[train_marl] done. total env_steps=... train_iters=...`，中间多次 `loss=`。

- [ ] **Step 4：无新的磁盘副作用**

```bash
ls data.pth 2>/dev/null && echo "BUG" || echo "OK no data.pth"
git status
```

Expected: 无 `data.pth`；`git status` 显示 `nothing to commit, working tree clean`。

- [ ] **Step 5：产出验收报告**

写简短总结回给用户：新增文件、测试数量、是否所有 PASS、smoke 是否通过、下一步（真实 SC2 训练的入口命令）。

---

## Self-Review

**1. Spec coverage：**
- ✅ 目录结构（Section 2）→ Task 1-7
- ✅ MARLEnv / MockSMAC / SMACWrapper（Section 3.1-3.3）→ Task 1, 6
- ✅ AgentNetwork（3.4）→ Task 3
- ✅ Mixer（3.5）→ Task 2
- ✅ EpisodeBuffer（3.6）→ Task 4
- ✅ QMIX（3.7）→ Task 5
- ✅ train_marl.py（3.8）→ Task 7
- ✅ 论文对齐的默认超参（Section 4）→ Task 7 的 yaml
- ✅ 7 种单测（Section 5）→ Task 1, 2, 3, 4, 8
- ✅ docs/MARL_EXTENSION.md（Section 6）→ Task 9

**2. Placeholder 扫描：** 无 TODO/TBD；所有 step 都有完整代码或具体命令。

**3. Type/命名一致性：**
- `AgentNetwork(obs_dim, n_actions, rnn_type, rnn_hidden_size, obs_embed_size, num_layers)` — Task 3 定义 + Task 7 使用，签名一致 ✓
- `QMixer(n_agents, state_dim, embed_dim, hypernet_layers, hypernet_embed)` — Task 2 定义 + Task 5/7 使用 ✓
- `QMIX(agent_net, mixer, lr, gamma, grad_clip, device)` — Task 5 定义 + Task 7 使用 ✓
- `EpisodeBuffer(buffer_size, episode_limit, n_agents, obs_dim, state_dim, n_actions)` — Task 4 定义 + Task 5/7 使用 ✓
- `MARLEnv` 接口（reset/step/get_obs/get_state/get_avail_actions/get_env_info）— Task 1 定义 + Task 6/7 遵守 ✓
- Episode dict keys（obs/state/actions/avail_actions/rewards/terminated）— Task 4 定义 + Task 7 的 `rollout_one_episode` 生成，一致 ✓
- `state_size_per_layer` 属性 — 与之前 Task 2-5（GRSN 神经元 refactor）一致 ✓
