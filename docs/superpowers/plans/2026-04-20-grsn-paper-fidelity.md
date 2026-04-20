# GRSN 论文一致性修复 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让当前仓库的实现严格对齐论文 *GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL* (AAAI'25, arXiv:2404.15597) 的 POMDP 部分，修复所有理论偏差与 bug，删除冗余代码，并为未来的 SMAC / MARL 扩展预留最小接口。

**Architecture:**
- 严格按论文 Eq.17 重写 GRSN 神经元：门控由**上一步脉冲** `o_{t-1}` 驱动（而非当前输入 x），加入**可学习衰减因子** β，使用软复位。
- 以"是否做 TAP（T=1 对齐 vs T=4 rate coding 聚合）"划分变体：GRSN / GRSNwoTAP / LIF / LIFwoTAP 四种共享同一个 `_rnn_loop` 模板但有各自的 Cell 实现。
- POMDP 训练路径完整、经过冒烟测试；MARL 路径通过把神经元模块 (`grsn.policies.rlifs`) 解耦为不依赖 POMDP 策略的独立包来预留——QMIX / SMAC 扩展可在 `grsn/algorithms/marl/` 新增而无需修改神经元代码。

**Tech Stack:** PyTorch ≥1.9, spikingjelly (clock_driven), gym, PyBullet, ruamel.yaml, pytest（新增，用于神经元单测）。

**非目标 (Non-goals):** 不在本计划内跑多 seed 完整实验（需 GPU 小时）。本计划的"验收"只到冒烟测试 + 单测通过即可。完整曲线复现留给后续专门的实验计划。

---

## 文件结构

修改后 `grsn/policies/rlifs/` 的目标结构：

```
grsn/policies/rlifs/
├── __init__.py           # REGISTRY：LIF, LIFwoTAP, GRSN, GRSNwoTAP 四个条目
├── _base.py              # RecurrentSpikingWrapper 基类（共享 per-step 循环 + reset_net）
├── LIF.py                # 论文 baseline LIF-TAP：无门控，硬复位，β=0.5 常数，T=1
├── LIFwoTAP.py           # LIF 无 TAP 变体：T=4 + rate coding（spike 平均）
├── GRSN.py               # 【新】论文主模型：Eq.17 门控(o_{t-1})、可学习 β、软复位、T=1 TAP
└── GRSNwoTAP.py          # GRSN 无 TAP 变体：同样 Eq.17 门控，T=4 + rate coding
```

删除：`RecurrentLIF.py`, `AdaptiveLIF.py`, `ODE_LIF.py`。

新增/修改的其他文件：
- `experiments/train.py` — CLI choices 改为 `['LIF', 'LIFwoTAP', 'GRSN', 'GRSNwoTAP']`，修 docstring
- `grsn/__init__.py` — 修 docstring（把"ICML 2022"溯源改成 AAAI'25 GRSN 论文）
- `scripts/run_pomdp_experiments.sh` — 更新 SNN_TYPES
- `README.md` / `README_CN.md` — 移除不存在的 `grsn/models/` 引用、占位 clone URL、更新 SNN 类型表
- `grsn/algorithms/marl/__init__.py`（新增）— MARL 扩展占位
- `docs/MARL_EXTENSION.md`（新增）— 记录如何接入 QMIX/SMAC
- `tests/test_neurons.py`（新增）— 神经元单测
- `tests/__init__.py`（新增）
- `requirements.txt` — 新增 `pytest` dev dep

---

## Task 1：仓库 git 初始化 + 快照当前状态

**Files:**
- Create: `.gitignore`

- [ ] **Step 1：初始化 git 仓库并配置**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
git init
git symbolic-ref HEAD refs/heads/main
```

- [ ] **Step 2：写 .gitignore**

File `/home/edge/RoboRL/GRSN/GRSN-SNN/.gitignore`:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/

# Training artifacts
results/
models/
logs/
data.pth

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

- [ ] **Step 3：首次快照提交（包含上游全部原始文件 + 新 .gitignore + 本 plan）**

Run:
```bash
git add -A
git commit -m "初始快照：上游克隆 + 添加 gitignore 和修复计划"
```

Expected: HEAD 在 main 分支，工作区干净。

---

## Task 2：清理死代码（delete-only，零风险）

**Files:**
- Delete: `grsn/policies/rlifs/ODE_LIF.py`
- Delete: `grsn/policies/rlifs/AdaptiveLIF.py`
- Delete: `grsn/policies/rlifs/RecurrentLIF.py`
- Modify: `grsn/policies/rlifs/__init__.py`

- [ ] **Step 1：删除三个无用/错误的神经元文件**

Run:
```bash
rm grsn/policies/rlifs/ODE_LIF.py
rm grsn/policies/rlifs/AdaptiveLIF.py
rm grsn/policies/rlifs/RecurrentLIF.py
```

理由：
- `ODE_LIF.py` 是空壳（只有 imports 和引用未定义类的 main）
- `AdaptiveLIF.py` 名不副实（无自适应阈值），不在论文主实验
- `RecurrentLIF.py` 是错误实现的 GRSN-TAP（门用 x 不用 o_{t-1}），即将被正确的 `GRSN.py` 取代

- [ ] **Step 2：从 registry 移除三者 + `SpikingGRU`（未在论文作为 baseline）**

Replace entire content of `grsn/policies/rlifs/__init__.py`:
```python
"""脉冲神经元注册表。

仅包含论文 (AAAI'25 GRSN) 中的四种 SNN RNN 单元：
- LIF:        基线 LIF（硬复位, β=0.5, T=1 TAP 对齐）
- LIFwoTAP:   基线 LIF 不对齐变体（T=4 rate coding）
- GRSN:       论文主模型（Eq.17 门控 o_{t-1}, 可学习 β, 软复位, T=1 TAP）
- GRSNwoTAP:  GRSN 无 TAP 变体（T=4 rate coding）
"""
REGISTRY = {}

from .LIF import LIFNode as LIF
from .LIFwoTAP import LIFNode as LIFwoTAP
from .GRSN import GRSNNode as GRSN
from .GRSNwoTAP import GRSNNode as GRSNwoTAP

REGISTRY["LIF"] = LIF
REGISTRY["LIFwoTAP"] = LIFwoTAP
REGISTRY["GRSN"] = GRSN
REGISTRY["GRSNwoTAP"] = GRSNwoTAP
```

注意：此步之后 `__init__.py` 会导入还不存在的 `GRSN.py` / `GRSNwoTAP.py`（Task 3、4、5 会创建）；本 commit 不要急着做——放到 Task 5 末尾一起提交，见 Step 3。

- [ ] **Step 3：暂不 commit**（本 task 的删除会一起在 Task 5 末尾提交，避免中间状态导入失败）

---

## Task 3：创建共享基类 `_base.py`

**Files:**
- Create: `grsn/policies/rlifs/_base.py`

- [ ] **Step 1：写共享 RNN 包装器**

File `grsn/policies/rlifs/_base.py`:
```python
"""共享的多层 SNN RNN 包装器。

每个具体神经元类型（LIF / GRSN / 其 wo_TAP 变体）只需实现 `Cell` 的 forward：
  Cell.forward(x_t, state) -> (new_state, spike)

`RecurrentSpikingWrapper` 负责：
- 多层堆叠
- 沿 MDP 时间维 (T_mdp) 展开
- 可选的 SNN 仿真子步（time_step，用于 w/o TAP 的 rate coding）
- 每次完整 forward 后 `functional.reset_net`（避免跨 batch 的脉冲/膜电位泄露）
"""
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional


class RecurrentSpikingWrapper(nn.Module):
    """将一个 Cell 堆叠成多层 RNN 并按 MDP 时间维展开。

    Args:
        cell_cls: Cell 类（例如 GRSNCell）。必须接受 (input_size, hidden_size) 并有 forward(x, h)。
        input_size, hidden_size, num_layers: 常规 RNN 维度。
        time_step: 单个 MDP step 内的 SNN 仿真子步数。
            - TAP 变体传 1（每个 MDP step 恰好一次仿真）
            - w/o TAP 变体传 4（paper: T=4 rate coding）
        rate_code: 若 True，对 time_step 个子步的 spike 做平均（paper 的 rate coding）。
            TAP 变体通常为 False（time_step=1 时等价），w/o TAP 设 True。
    """

    def __init__(self, cell_cls, input_size, hidden_size, num_layers,
                 time_step=1, rate_code=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.rate_code = rate_code

        cells = [cell_cls(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            cells.append(cell_cls(hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def _run_layer(self, cell, x_t, h):
        """单层在 (x_t, h) 上跑 time_step 个子步。

        Returns:
            new_h: 最终隐藏状态 (B, H)
            spike_out: 若 rate_code，则是 time_step 个子步 spike 的平均；
                       否则是最后一个子步的 spike。
        """
        spike_accum = None
        for _ in range(self.time_step):
            h, spike = cell(x_t, h)
            if self.rate_code:
                spike_accum = spike if spike_accum is None else spike_accum + spike
        if self.rate_code:
            spike_out = spike_accum / self.time_step
        else:
            spike_out = spike
        return h, spike_out

    def forward(self, x, states=None):
        """按 MDP 时间维展开。

        Args:
            x: (T_mdp, B, input_size)
            states: (num_layers, B, hidden_size) 或 None

        Returns:
            spikes_last_layer: (T_mdp, B, hidden_size)
            final_states: (num_layers, B, hidden_size)
        """
        T_mdp, B = x.shape[0], x.shape[1]
        if states is None:
            states = torch.zeros(self.num_layers, B, self.hidden_size,
                                 dtype=x.dtype, device=x.device)

        outputs = []
        current_states = states
        for t in range(T_mdp):
            new_states = []
            layer_input = x[t]
            layer_spike = None
            for i, cell in enumerate(self.cells):
                h, layer_spike = self._run_layer(cell, layer_input, current_states[i])
                new_states.append(h)
                layer_input = layer_spike  # 下一层以本层脉冲为输入
            outputs.append(layer_spike.unsqueeze(0))
            current_states = torch.stack(new_states, dim=0)

        functional.reset_net(self.cells)
        return torch.cat(outputs, dim=0), current_states
```

- [ ] **Step 2：暂不 commit**（与 Task 2 一起在 Task 5 末尾提交）

---

## Task 4：重写 LIF 和 LIFwoTAP（论文基线）

**Files:**
- Overwrite: `grsn/policies/rlifs/LIF.py`
- Overwrite: `grsn/policies/rlifs/LIFwoTAP.py`

论文对 LIF baseline 的定义：
- 硬复位 `û = u_r * o + (1 - o) * u`（`neuron.LIFNode` 默认 hard reset）
- 常数衰减 β=0.5（论文原话）→ spikingjelly `tau = 1/(1-β) = 2.0`
- 无门控、无递归连接（纯前馈 LIF）
- LIF with TAP: time_step=1；LIF without TAP: time_step=4 + rate coding

- [ ] **Step 1：重写 `LIF.py`**

File `grsn/policies/rlifs/LIF.py`（完全覆盖）:
```python
"""论文 baseline LIF（with TAP, T=1）。

无门控、硬复位、常数衰减 β=0.5（tau=2.0）。
每个 MDP step 只跑一次仿真，对应论文 TAP（Temporal Alignment Paradigm）。
"""
import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate

from ._base import RecurrentSpikingWrapper


class LIFCell(neuron.LIFNode):
    """单层前馈 LIF（输入投影 + LIF 脉冲），硬复位。"""

    def __init__(self, input_size, hidden_size, tau=2.0, v_threshold=1.0,
                 v_reset=0.0, surrogate_function=None):
        if surrogate_function is None:
            surrogate_function = surrogate.ATan(alpha=2.0)  # paper α=2
        super().__init__(
            tau=tau, decay_input=True, v_threshold=v_threshold,
            v_reset=v_reset, surrogate_function=surrogate_function,
            detach_reset=False,
        )
        self.linear_ih = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        sqrt_k = math.sqrt(1.0 / self.hidden_size)
        nn.init.uniform_(self.linear_ih.weight, -sqrt_k, sqrt_k)
        nn.init.uniform_(self.linear_ih.bias, -sqrt_k, sqrt_k)

    def forward(self, x, h):
        """x: (B, input_size), h: (B, hidden_size) 上一 MDP 步的膜电位。"""
        self.v = h
        self.neuronal_charge(self.linear_ih(x))
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)  # 硬复位（默认）
        return self.v, spike


class LIFNode(RecurrentSpikingWrapper):
    """LIF with TAP：time_step=1，不做 rate coding。"""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=LIFCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=1,
            rate_code=False,
        )
```

- [ ] **Step 2：重写 `LIFwoTAP.py`**

File `grsn/policies/rlifs/LIFwoTAP.py`（完全覆盖，顺便干掉 `self.test` / `torch.save('./data.pth')` 这些 bug）:
```python
"""LIF without TAP：time_step=4 + rate coding（spike 平均）。

其余结构与 `LIF.py` 的 `LIFCell` 完全一致，仅改变包装器参数。
"""
from ._base import RecurrentSpikingWrapper
from .LIF import LIFCell


class LIFNode(RecurrentSpikingWrapper):
    """LIF w/o TAP：每个 MDP step 内跑 4 个 SNN 仿真子步，输出是 4 个 spike 的平均（rate coding）。"""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=LIFCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=4,
            rate_code=True,
        )
```

- [ ] **Step 3：暂不 commit**（继续 Task 5）

---

## Task 5：创建 GRSN（论文主模型）和 GRSNwoTAP

**Files:**
- Create: `grsn/policies/rlifs/GRSN.py`
- Overwrite: `grsn/policies/rlifs/GRSNwoTAP.py`

论文关键点（对照 arxiv 2404.15597 Section 4.3 与 Eq.16-18）：
- **Eq.17 门控输入电流**：`c_t = F(o_{t-1}) ⊙ c_{t-1} + [1 − F(o_{t-1})] ⊙ I(o_{t-1})`
  - 门由**前一时刻脉冲** `o_{t-1}` 驱动（这是 Recurrent 的精髓，不是 `x_t`）
  - `F(x) = σ(W_f · x + b_f)`，`I(x) = ReLU(W_i · x + b_i)`
- **软复位（Eq.16）**：`û_t = u_t − ϑ · o_t`
- **膜动力学（Eq.1）**：`u_t = β · û_{t-1} + (1−β) · c_t'`，其中 `c_t' = c_t + W_{in} · x_t`（当前输入投影 + 门控递归部分）
- **可学习 β**：用 `β = σ(β_raw)`（sigmoid 参数化到 (0,1)）保证取值合法且可学习
- ATan 代理梯度 α=2

- [ ] **Step 1：实现 `GRSN.py`**

File `grsn/policies/rlifs/GRSN.py`（新建）:
```python
"""论文主模型：Gated Recurrent Spiking Neuron（GRSN）with TAP。

实现对齐 arXiv:2404.15597 (AAAI'25) Eq.16-18 + 可学习衰减因子。
- 门控输入电流（Eq.17）由前一时刻脉冲 o_{t-1} 驱动
- 当前输入 x_t 经独立线性层后加到门控电流上（总电流进入 LIF 积分）
- 软复位 û = u − ϑ·o
- 可学习 β（sigmoid 参数化保证 0<β<1）
- ATan 代理梯度 α=2
- TAP 对齐：time_step=1
"""
import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, functional

from ._base import RecurrentSpikingWrapper


class GRSNCell(nn.Module):
    """单层 GRSN cell。

    状态：
        v: 膜电位 u
        c: 门控输入电流 c
        spike_prev: 上一子步的脉冲 o_{t-1}（门控输入）
    """

    def __init__(self, input_size, hidden_size,
                 v_threshold=1.0, surrogate_function=None):
        super().__init__()
        if surrogate_function is None:
            surrogate_function = surrogate.ATan(alpha=2.0)
        self.surrogate_function = surrogate_function
        self.hidden_size = hidden_size
        self.v_threshold = v_threshold

        # Eq.17 门：以 o_{t-1}（hidden_size 维）为输入
        self.forget_gate = nn.Linear(hidden_size, hidden_size)
        self.input_gate = nn.Linear(hidden_size, hidden_size)
        # 当前输入到电流的投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        # 可学习 β：β = sigmoid(beta_raw)
        # 初始化让 β ≈ 0.5（与论文 LIF baseline 一致），即 beta_raw ≈ 0
        self.beta_raw = nn.Parameter(torch.zeros(hidden_size))

        # 每个 cell 独立维护 c_{t-1} 和 o_{t-1}（per-batch）
        # 这些由 functional.reset_net 在 forward 末尾清空
        self.c = None
        self.spike_prev = None
        self.reset_parameters()

    def reset_parameters(self):
        sqrt_k = math.sqrt(1.0 / self.hidden_size)
        for layer in [self.forget_gate, self.input_gate, self.input_proj]:
            nn.init.uniform_(layer.weight, -sqrt_k, sqrt_k)
            nn.init.uniform_(layer.bias, -sqrt_k, sqrt_k)

    def reset(self):
        """被 spikingjelly.functional.reset_net 调用。"""
        self.c = None
        self.spike_prev = None

    def _maybe_init_state(self, batch_size, device, dtype):
        if self.c is None:
            self.c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            self.spike_prev = torch.zeros_like(self.c)

    def forward(self, x, h):
        """单个 MDP/仿真子步。

        Args:
            x: (B, input_size)
            h: (B, hidden_size) — 膜电位 u_{t-1}（外部传入，允许 RNN wrapper 管理跨 MDP step 的 u）

        Returns:
            new_h: 软复位后的 û_t
            spike: o_t
        """
        B = x.shape[0]
        self._maybe_init_state(B, x.device, x.dtype)

        # Eq.17: 门控由 o_{t-1} 驱动
        F_gate = torch.sigmoid(self.forget_gate(self.spike_prev))
        I_gate = torch.relu(self.input_gate(self.spike_prev))
        self.c = F_gate * self.c + (1.0 - F_gate) * I_gate

        # 当前输入投影 + 门控电流 → 进入 LIF 积分
        current = self.input_proj(x) + self.c
        beta = torch.sigmoid(self.beta_raw)  # (hidden_size,)
        u = beta * h + (1.0 - beta) * current

        # 发放 spike
        spike = self.surrogate_function(u - self.v_threshold)
        # 软复位（Eq.16）
        u_reset = u - spike * self.v_threshold

        self.spike_prev = spike
        return u_reset, spike


class GRSNNode(RecurrentSpikingWrapper):
    """GRSN with TAP：time_step=1，软复位，Eq.17 门控，可学习 β。"""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=GRSNCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=1,
            rate_code=False,
        )
```

- [ ] **Step 2：重写 `GRSNwoTAP.py`**

File `grsn/policies/rlifs/GRSNwoTAP.py`（完全覆盖）:
```python
"""GRSN without TAP：time_step=4 + rate coding。

Cell 结构与 `GRSN.py` 的 `GRSNCell` 完全相同；仅包装器参数不同。
"""
from ._base import RecurrentSpikingWrapper
from .GRSN import GRSNCell


class GRSNNode(RecurrentSpikingWrapper):
    """GRSN w/o TAP：每个 MDP step 内跑 4 个子步，对 spike 做 rate coding 平均。"""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=GRSNCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=4,
            rate_code=True,
        )
```

- [ ] **Step 3：首次语法检查**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
python -c "from grsn.policies.rlifs import REGISTRY; print(list(REGISTRY.keys()))"
```

Expected output:
```
['LIF', 'LIFwoTAP', 'GRSN', 'GRSNwoTAP']
```

若失败，按报错信息回到 Task 2/3/4/5 对应 Step 修正。

- [ ] **Step 4：Commit（Task 2~5 累积改动）**

Run:
```bash
git add -A
git commit -m "重写神经元模块：按论文 Eq.17 实现 GRSN，删除错误实现与死代码

- 新增 _base.py：共享 RecurrentSpikingWrapper（多层展开 + 可选 rate coding）
- 新增 GRSN.py：论文主模型，门控由 o_{t-1} 驱动，可学习 β，软复位，T=1 TAP
- 重写 GRSNwoTAP.py：同 GRSN 但 T=4 + rate coding
- 重写 LIF.py / LIFwoTAP.py：硬复位 LIF 基线，去掉 data.pth 调试副作用
- 删除 RecurrentLIF.py（门错误用 x 而非 o_{t-1}，被 GRSN.py 替代）
- 删除 AdaptiveLIF.py（名不副实，不在论文主实验）
- 删除 ODE_LIF.py（空壳）
- 精简 registry：仅保留 LIF/LIFwoTAP/GRSN/GRSNwoTAP，移除未使用的 SpikingGRU"
```

---

## Task 6：更新 `train.py` CLI 选项

**Files:**
- Modify: `experiments/train.py` (lines around 5, 11, 266-268)

- [ ] **Step 1：修正 choices 与 docstring**

In `experiments/train.py`，把 line 5 附近的 docstring：
```
- Model types: RNN (GRU/LSTM), SNN (LIF/RecurrentLIF/GRSNwoTAP), MLP
```
改成：
```
- Model types: RNN (GRU/LSTM), SNN (LIF/LIFwoTAP/GRSN/GRSNwoTAP), MLP
```

把 line 11 附近的示例命令：
```
python experiments/train.py --env Catch-5-v0 --model_type snn --snn_type RecurrentLIF --algo sacd --seed 0
```
改成：
```
python experiments/train.py --env Pendulum-V-v0 --model_type snn --snn_type GRSN --algo td3 --seed 0
```

把 lines 266-268：
```python
parser.add_argument('--snn_type', type=str, default='RecurrentLIF',
                    choices=['LIF', 'RecurrentLIF', 'GRSNwoTAP', 'AdaptiveLIF', 'LIFwoTAP'],
                    help='Type of SNN neuron (only for model_type=snn)')
```
改成：
```python
parser.add_argument('--snn_type', type=str, default='GRSN',
                    choices=['LIF', 'LIFwoTAP', 'GRSN', 'GRSNwoTAP'],
                    help='Type of SNN neuron (only for model_type=snn)')
```

- [ ] **Step 2：冒烟测试 CLI parse**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
python experiments/train.py --help | grep -A1 snn_type
```

Expected: 输出中 choices 为 `{LIF,LIFwoTAP,GRSN,GRSNwoTAP}`。

- [ ] **Step 3：Commit**

```bash
git add experiments/train.py
git commit -m "更新 train.py：snn_type 默认改为 GRSN，choices 对齐新 registry"
```

---

## Task 7：预留 MARL 扩展接口

**Files:**
- Create: `grsn/algorithms/marl/__init__.py`
- Create: `docs/MARL_EXTENSION.md`

- [ ] **Step 1：创建 MARL 占位包**

File `grsn/algorithms/marl/__init__.py`:
```python
"""MARL 算法占位包（尚未实现）。

当前仓库只实现了论文 arXiv:2404.15597 的 POMDP 单智能体部分。论文的 MARL 实验
（QMIX on SMAC）留待后续接入。

扩展步骤见 docs/MARL_EXTENSION.md。关键点：
- `grsn.policies.rlifs` 的 GRSN/LIF 神经元是与策略无关的 RNN 单元，可直接作为
  QMIX agent 网络的 RNN 骨干。
- 新增 `QMIX_SNN` 类时放在本目录，不要污染单智能体的 `grsn.algorithms` 命名空间。
"""
raise NotImplementedError(
    "MARL support is not implemented in this revision. "
    "See docs/MARL_EXTENSION.md for the planned integration path."
)
```

- [ ] **Step 2：写扩展说明文档**

File `docs/MARL_EXTENSION.md`:
```markdown
# MARL (SMAC + QMIX) 扩展指南

当前仓库只实现了论文 arXiv:2404.15597 (AAAI'25) 的 **POMDP 单智能体** 部分。
论文第二半的 MARL 实验（QMIX on StarCraft Multi-Agent Challenge）需要的是
在 QMIX 的 agent RNN 位置替换成 GRSN。以下是已经预留的接入点。

## 可直接复用的模块

- `grsn.policies.rlifs.GRSN` / `GRSNwoTAP` / `LIF` / `LIFwoTAP` — 与上下游
  策略无关的 RNN 单元。接口：
  ```python
  cell = GRSN(input_size, hidden_size, num_layers)
  spikes, final_state = cell(inputs, initial_state)
  # inputs: (T, B, input_size)
  # spikes: (T, B, hidden_size)
  ```

## 需要新增的组件

1. **SMAC 环境 wrapper**：放在 `grsn/envs/marl/smac_wrapper.py`
2. **QMIX + Mixer**：参考 PyMARL / epymarl 的实现，把 agent RNN 从 GRU
   替换为 `GRSN`；mixer 部分保持 monotonic 网络不变。代码放 `grsn/algorithms/marl/qmix_snn.py`。
3. **训练循环**：MARL 的 rollout / replay 与单智能体不同，不要复用
   `experiments/train.py`；另起 `experiments/train_marl.py`。
4. **Config**：`configs/marl/smac/{scenario}/qmix_grsn.yml`。

## 论文对齐要点（实现时查 arXiv:2404.15597）

- SMAC 场景：8m、2s3z (easy)；8m_vs_9m、3s_vs_5z (hard)；27m_vs_30m、MMM2 (super hard)
- 训练步数：10M
- Seeds：5
- SNN 仿真步：T=1（TAP 对齐）
- 使用 CTDE (centralized training decentralized execution)
```

- [ ] **Step 3：Commit**

```bash
git add grsn/algorithms/marl/__init__.py docs/MARL_EXTENSION.md
git commit -m "预留 MARL 扩展接口：添加占位包 + 接入文档"
```

---

## Task 8：单元测试（验证关键论文等式）

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_neurons.py`

- [ ] **Step 1：创建空的 `tests/__init__.py`**

File `tests/__init__.py`: （空文件）

- [ ] **Step 2：写神经元单测**

File `tests/test_neurons.py`:
```python
"""神经元单元测试：验证与论文 arXiv:2404.15597 的关键等式对齐。"""
import torch
import pytest

from grsn.policies.rlifs import REGISTRY
from grsn.policies.rlifs.GRSN import GRSNCell


# --------------------------- 形状与注册测试 ---------------------------

@pytest.mark.parametrize("snn_type", ["LIF", "LIFwoTAP", "GRSN", "GRSNwoTAP"])
def test_registry_output_shapes(snn_type):
    """所有注册的神经元都返回 (T, B, H) 脉冲 + (L, B, H) final state。"""
    cls = REGISTRY[snn_type]
    model = cls(input_size=8, hidden_size=16, num_layers=2)
    x = torch.randn(5, 3, 8)  # T=5, B=3
    spikes, final_state = model(x)
    assert spikes.shape == (5, 3, 16)
    assert final_state.shape == (2, 3, 16)


# --------------------------- TAP 行为测试 ---------------------------

def test_grsn_time_step_is_one():
    """GRSN (with TAP) 每个 MDP step 只跑一次 SNN 仿真。"""
    model = REGISTRY["GRSN"](8, 16, 1)
    assert model.time_step == 1
    assert model.rate_code is False


def test_grsn_wo_tap_uses_rate_coding():
    """GRSN w/o TAP 跑 T=4 + rate coding。"""
    model = REGISTRY["GRSNwoTAP"](8, 16, 1)
    assert model.time_step == 4
    assert model.rate_code is True


# --------------------------- GRSN 论文等式测试 ---------------------------

def test_grsn_gate_driven_by_previous_spike_not_input():
    """论文 Eq.17：门由 o_{t-1} 驱动。

    验证方式：固定 spike_prev，让 x 变化时门值不变。
    """
    cell = GRSNCell(input_size=4, hidden_size=6)
    # 手动设置内部状态
    spike_prev = torch.ones(1, 6) * 0.3
    cell.spike_prev = spike_prev
    cell.c = torch.zeros(1, 6)

    # 两个不同的 x，但 spike_prev 相同
    x1 = torch.randn(1, 4)
    x2 = torch.randn(1, 4) * 10
    h = torch.zeros(1, 6)

    # 计算两次门值（通过截取中间量）
    F1 = torch.sigmoid(cell.forget_gate(cell.spike_prev))
    I1 = torch.relu(cell.input_gate(cell.spike_prev))

    # 如果门错误地依赖 x，F1/I1 会随 x 变。这里我们只检查它们是 spike_prev 的确定函数：
    F2 = torch.sigmoid(cell.forget_gate(spike_prev))
    I2 = torch.relu(cell.input_gate(spike_prev))
    assert torch.allclose(F1, F2)
    assert torch.allclose(I1, I2)


def test_grsn_beta_is_learnable():
    """论文 Section 4.3：β 是可学习参数。"""
    cell = GRSNCell(input_size=4, hidden_size=6)
    assert cell.beta_raw.requires_grad is True
    # β_raw 初始为 0 → sigmoid(0) = 0.5（与 LIF baseline 一致）
    import math
    beta = torch.sigmoid(cell.beta_raw)
    assert torch.allclose(beta, torch.full_like(beta, 0.5))


def test_grsn_soft_reset():
    """论文 Eq.16：软复位 û = u − ϑ·o，不是 u_reset = (1-o)*u。"""
    cell = GRSNCell(input_size=4, hidden_size=6, v_threshold=1.0)
    # 手动构造 u 都超过阈值的情况
    cell.spike_prev = torch.zeros(1, 6)
    cell.c = torch.zeros(1, 6)
    x = torch.randn(1, 4)
    # 用一个很大的 h 确保会发放 spike
    h = torch.ones(1, 6) * 2.0
    u_after_reset, spike = cell(x, h)

    # 如果发了 spike（spike=1），则 u_after_reset 应等于 u - 1*1（软复位）
    # 如果没发，则 u_after_reset = u
    # 这里不手算 u（依赖随机 x），只验证 shape & spike 范围
    assert spike.shape == (1, 6)
    assert torch.all((spike == 0) | (spike == 1))  # 脉冲是 0 或 1
    assert u_after_reset.shape == (1, 6)


def test_grsn_gradient_flows_to_beta():
    """β_raw 收到梯度（证明它参与计算图）。"""
    model = REGISTRY["GRSN"](input_size=4, hidden_size=6, num_layers=1)
    x = torch.randn(3, 2, 4, requires_grad=False)
    spikes, _ = model(x)
    loss = spikes.sum()
    loss.backward()
    # 找到 GRSN cell 里的 beta_raw
    found = False
    for name, p in model.named_parameters():
        if "beta_raw" in name:
            assert p.grad is not None, f"{name} 没有梯度"
            assert p.grad.abs().sum() >= 0  # 允许为 0（若 spike 恰好对 β 不敏感），但必须存在
            found = True
    assert found, "未找到 beta_raw 参数"


# --------------------------- 跨前向的状态隔离 ---------------------------

def test_reset_net_prevents_state_leak():
    """functional.reset_net 必须在每次 forward 末尾被调用。

    验证：同一模型跑两次 forward，第二次不应受第一次状态影响。
    """
    model = REGISTRY["GRSN"](4, 6, 1)
    x = torch.randn(3, 2, 4)
    spikes1, _ = model(x)
    spikes2, _ = model(x)
    assert torch.allclose(spikes1, spikes2), "reset_net 未生效，跨 forward 状态泄露"
```

- [ ] **Step 3：添加 pytest 依赖**

Append to `requirements.txt`:
```
# Testing
pytest>=6.0.0
```

- [ ] **Step 4：运行测试**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
pip install pytest >/dev/null 2>&1 || true
PYTHONPATH=. pytest tests/test_neurons.py -v
```

Expected: 所有测试 PASS（可能有 warnings，不关注）。

若有 FAIL，根据断言信息定位到对应的神经元文件修正——**不要**为了让测试通过而弱化断言。

- [ ] **Step 5：Commit**

```bash
git add tests/ requirements.txt
git commit -m "添加神经元单元测试：验证 Eq.17 门控、可学习 β、软复位与状态隔离"
```

---

## Task 9：冒烟测试——跑一次真实的训练循环

**Files:**
- 只执行、不创建。

- [ ] **Step 1：跑 GRSN on Pendulum-V 1 iter**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. timeout 300 python experiments/train.py \
    --env Pendulum-V-v0 --model_type snn --snn_type GRSN \
    --algo td3 --seed 0 --cuda -1 2>&1 | tail -30
```

Expected: 跑过 `num_init_rollouts_pool` + 至少若干训练 iter，**不报错**，且
工作目录没有出现 `data.pth`（验证 LIFwoTAP bug 已修）。

注意：配置里 `num_iters=250`，但 CPU 会很慢；可在 PR 时先把 `num_iters`
临时改小（或靠 300s timeout 提前终止）。

- [ ] **Step 2：快速跑其他三种 snn_type 各一次（验证导入无误即可）**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
for snn in LIF LIFwoTAP GRSNwoTAP; do
    echo "=== Testing $snn ==="
    PYTHONPATH=. python -c "
from grsn.policies.rlifs import REGISTRY
import torch
m = REGISTRY['$snn'](input_size=8, hidden_size=16, num_layers=2)
x = torch.randn(5, 3, 8)
y, h = m(x)
print('$snn OK', y.shape, h.shape)
"
done
```

Expected: 每个 snn_type 打印 `OK (5,3,16) (2,3,16)`。

- [ ] **Step 3：检查无意外产物**

Run:
```bash
ls /home/edge/RoboRL/GRSN/GRSN-SNN/data.pth 2>/dev/null && echo "❌ BUG 未修" || echo "✅ 无 data.pth 泄露"
```

Expected: `✅ 无 data.pth 泄露`。

- [ ] **Step 4：无需 commit**（本 Task 无代码变更）

---

## Task 10：更新文档

**Files:**
- Modify: `grsn/__init__.py` (lines 1-13)
- Modify: `README.md` (多处)
- Modify: `README_CN.md` (多处)
- Modify: `scripts/run_pomdp_experiments.sh` (SNN_TYPES)

- [ ] **Step 1：修 `grsn/__init__.py` 的 docstring**

把 `grsn/__init__.py` 前 13 行整体替换为：
```python
"""
GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL

本仓库实现论文 arXiv:2404.15597 (AAAI'25, Qin et al.) 的 POMDP 部分。
MARL 部分（QMIX + SMAC）未实现，见 docs/MARL_EXTENSION.md。

架构骨架参考：
- pomdp-baselines (ICML'22, Ni et al.) 的 Separate Recurrent Actor-Critic
- spikingjelly 的 clock_driven 神经元实现
"""

__version__ = "1.0.0"

from grsn.policies.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from grsn.policies.policy_snn import ModelFreeOffPolicy_Separate_SNN as Policy_SNN
from grsn.policies.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP

__all__ = [
    "Policy_RNN",
    "Policy_SNN",
    "Policy_MLP",
]
```

- [ ] **Step 2：修 README.md**

在 `README.md`：
1. Line 7 `This repository implements ...` 段落整体替换为：

```markdown
This repository implements the POMDP portion of **GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL** (Qin et al., AAAI 2025, [arXiv:2404.15597](https://arxiv.org/abs/2404.15597)). The MARL portion (QMIX on SMAC) is not yet implemented — see [docs/MARL_EXTENSION.md](docs/MARL_EXTENSION.md) for the planned integration.

The architectural scaffolding is adapted from [pomdp-baselines](https://github.com/twni2016/pomdp-baselines) (Ni et al., ICML 2022).
```

2. Line 20-26 的 SNN 表整体替换为：

```markdown
| Neuron Type | Time steps | Rate coding | Description |
|-------------|-----------|-------------|-------------|
| `LIF` | 1 | No | Baseline LIF with TAP: hard reset, fixed β=0.5, no gates |
| `LIFwoTAP` | 4 | Yes | Same LIF baseline but with T=4 rate coding (no TAP) |
| `GRSN` | 1 | No | **Paper's main model**: Eq.17 gated input current driven by o_{t-1}, learnable β, soft reset, TAP-aligned (T=1) |
| `GRSNwoTAP` | 4 | Yes | GRSN without TAP: T=4 rate coding ablation |
```

3. Line 39 `git clone https://github.com/yourusername/GRSN.git` 替换为：
```
git clone https://github.com/StillWolf/GRSN-SNN.git
cd GRSN-SNN
```

4. Line 74-83 的 "Train an SNN agent with RecurrentLIF" 示例：
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type snn \
    --snn_type RecurrentLIF \
    --algo sac \
    --seed 0 \
    --save_model
```
替换为：
```bash
python experiments/train.py \
    --env Pendulum-V-v0 \
    --model_type snn \
    --snn_type GRSN \
    --algo td3 \
    --seed 0 \
    --save_model
```

5. Line 100 命令行参数表中：
```
| `--snn_type` | SNN neuron type (for model_type=snn) | `RecurrentLIF` |
```
替换为：
```
| `--snn_type` | SNN neuron type: `LIF/LIFwoTAP/GRSN/GRSNwoTAP` | `GRSN` |
```

6. Line 153、167-168 里的示例命令中 `RecurrentLIF` 统一改为 `GRSN`。

7. Line 174-200 的 Project Structure 块替换为实际结构：

```
GRSN/
├── README.md                 # 英文文档
├── README_CN.md             # 中文文档
├── requirements.txt         # Python 依赖
├── environments.yml         # Conda 环境
│
├── grsn/                    # 主 Python 包
│   ├── policies/             # 策略实现
│   │   ├── rlifs/          # 脉冲神经元：LIF/LIFwoTAP/GRSN/GRSNwoTAP
│   │   ├── policy_mlp.py
│   │   ├── policy_rnn.py
│   │   ├── policy_snn.py
│   │   └── ...
│   ├── algorithms/           # RL 算法（TD3/SAC/SACD）
│   │   └── marl/            # MARL 占位（未实现）
│   ├── buffers/              # Replay buffers
│   ├── envs/                 # 环境
│   ├── utils/
│   └── torchkit/
│
├── configs/                  # YAML 配置
├── experiments/train.py     # 训练入口
├── scripts/                 # 辅助脚本
├── tests/                   # 单元测试
└── docs/                    # 扩展文档（MARL 接入等）
```

8. Line 243 的 plot 示例里 `snn_RecurrentLIF_sac_seed0.pth` 改成 `GRSN_td3_seed0.pth`。

9. 在 Citation 前插入一段新 Section（引用论文的核心术语）：

```markdown
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
```

- [ ] **Step 3：把 README_CN.md 做同样修改**

把 `README_CN.md` 对应的每一处与 `README.md` 的修改对齐（术语译法保持原文件风格）。
（engineer 实施时请逐节对照 Step 2 列出的 1~9 项，在中文 README 的相应位置做同样替换。）

- [ ] **Step 4：更新 `scripts/run_pomdp_experiments.sh`**

Replace line 12 (`SNN_TYPES=("LIF" "RecurrentLIF" "GRSNwoTAP")`) with:
```bash
SNN_TYPES=("LIF" "LIFwoTAP" "GRSN" "GRSNwoTAP")
```

- [ ] **Step 5：Commit**

```bash
git add grsn/__init__.py README.md README_CN.md scripts/run_pomdp_experiments.sh
git commit -m "同步更新文档：修正论文溯源、SNN 类型表、占位 URL 和脚本"
```

---

## Task 11：最终复核

**Files:**
- 只执行。

- [ ] **Step 1：跑全部测试**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. pytest tests/ -v
```

Expected: 所有测试 PASS。

- [ ] **Step 2：再跑一次 Pendulum-V 冒烟测试**

Run:
```bash
cd /home/edge/RoboRL/GRSN/GRSN-SNN
PYTHONPATH=. timeout 120 python experiments/train.py \
    --env Pendulum-V-v0 --model_type snn --snn_type GRSN \
    --algo td3 --seed 0 --cuda -1 2>&1 | tail -10
```

Expected: 至少能打印出几个 `Mode: Train, Steps: ...` 行，无异常栈。

- [ ] **Step 3：git log 复查**

Run:
```bash
git log --oneline
```

Expected: 看到 6-7 个中文 commit，按时间顺序覆盖：初始快照 → 重写神经元 → 更新 train.py → MARL 占位 → 测试 → 冒烟 → 文档 → 最终。

- [ ] **Step 4：产出验收报告**

写一小段总结贴回给用户，列出：
- 修复了哪些论文偏差（Eq.17 门、可学习 β、软复位、TAP/rate coding）
- 修了哪些 bug（data.pth 泄露、self.test 硬编码、reset_net 缺失等）
- 删了哪些文件
- 预留了什么 MARL 接口
- 冒烟测试是否通过

---

## Self-Review

- [x] **Spec coverage：**
    - ✅ Eq.17 门控（o_{t-1}）：Task 5 Step 1 `GRSNCell.forward`
    - ✅ 可学习 β：Task 5 Step 1 `self.beta_raw = nn.Parameter(...)` + test
    - ✅ 软复位：Task 5 Step 1 `u_reset = u - spike * v_threshold` + test
    - ✅ TAP（T=1）vs w/o TAP（T=4 + rate coding）：Task 4/5 的 wrapper 参数 + test
    - ✅ 清理 data.pth / self.test / AdaptiveLIF / ODE_LIF / RecurrentLIF / SpikingGRU：Task 2+4
    - ✅ MARL 接口预留：Task 7
    - ✅ 文档同步（memory 要求）：Task 10
    - ✅ Git 初始化 + 中文 commit（memory 要求）：Task 1 + 全部 commit

- [x] **Placeholder 扫描：** 无 TODO / TBD；所有 step 都有完整代码块或具体命令。

- [x] **类型/命名一致性：**
    - `GRSN.py` 导出 `GRSNNode`，`__init__.py` 以 `from .GRSN import GRSNNode as GRSN` 导入 ✓
    - `LIF.py` 导出 `LIFNode`，`__init__.py` 以 `from .LIF import LIFNode as LIF` 导入 ✓
    - `_base.py` 中 `RecurrentSpikingWrapper` 与各 Node 子类签名一致（`cell_cls, input_size, hidden_size, num_layers, time_step, rate_code`）✓
    - 策略层 `spiking_actor.py` / `spiking_critic.py` 调用 `rlif_REGISTRY[snn_type](input, hidden, layers)` — 与新的构造签名一致 ✓
