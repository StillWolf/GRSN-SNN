"""论文主模型：Gated Recurrent Spiking Neuron（GRSN）with TAP。

实现对齐 arXiv:2404.15597 (AAAI'25) Eq.16-18 + 可学习衰减因子。
- 门控输入电流（Eq.17）由前一时刻脉冲 o_{t-1} 驱动
- 当前输入 x_t 经独立线性层后加到门控电流上（总电流进入 LIF 积分）
- 软复位 û = u − ϑ·o
- 可学习 β（sigmoid 参数化保证 0<β<1）
- ATan 代理梯度 α=2
- TAP 对齐：time_step=1

状态张量布局：
    state: (B, 3*hidden_size) = concat([h, c, spike_prev], dim=-1)
    其中 h 是膜电位，c 是门控电流，spike_prev 是上一步脉冲 o_{t-1}。
    使用外部 state 是为了让门控递归在推理阶段（每次 act() 调用 T_mdp=1）
    也能跨 MDP step 保留——否则 spike_prev 总是 0，GRSN 的门控失效。
"""
import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate

from ._base import RecurrentSpikingWrapper


class GRSNCell(nn.Module):
    """单层 GRSN cell。状态完全外部化（h, c, spike_prev 拼接在 state 张量里）。"""

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
        # 可学习 β：β = sigmoid(beta_raw)，beta_raw=0 → β=0.5（与 LIF baseline 对齐）
        self.beta_raw = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    @property
    def state_size(self):
        """wrapper 用来分配 state 张量的第三维大小。"""
        return 3 * self.hidden_size

    def reset_parameters(self):
        sqrt_k = math.sqrt(1.0 / self.hidden_size)
        for layer in [self.forget_gate, self.input_gate, self.input_proj]:
            nn.init.uniform_(layer.weight, -sqrt_k, sqrt_k)
            nn.init.uniform_(layer.bias, -sqrt_k, sqrt_k)

    def forward(self, x, state):
        """单步前向。

        Args:
            x: (B, input_size)
            state: (B, 3*hidden_size) = concat([h, c, spike_prev])

        Returns:
            new_state: (B, 3*hidden_size) 下一步状态
            spike: (B, hidden_size) 本步脉冲 o_t
        """
        H = self.hidden_size
        h = state[:, 0:H]
        c = state[:, H:2 * H]
        spike_prev = state[:, 2 * H:3 * H]

        # Eq.17：门控由 o_{t-1} 驱动
        F_gate = torch.sigmoid(self.forget_gate(spike_prev))
        I_gate = torch.relu(self.input_gate(spike_prev))
        c_new = F_gate * c + (1.0 - F_gate) * I_gate

        # 当前输入投影 + 门控电流
        current = self.input_proj(x) + c_new
        beta = torch.sigmoid(self.beta_raw)
        u = beta * h + (1.0 - beta) * current

        # 发放 + 软复位（Eq.16）
        spike = self.surrogate_function(u - self.v_threshold)
        h_new = u - spike * self.v_threshold

        new_state = torch.cat([h_new, c_new, spike], dim=-1)
        return new_state, spike


class GRSNNode(RecurrentSpikingWrapper):
    """GRSN with TAP：time_step=1，软复位，Eq.17 门控 o_{t-1}，可学习 β。"""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=GRSNCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=1,
            rate_code=False,
        )
