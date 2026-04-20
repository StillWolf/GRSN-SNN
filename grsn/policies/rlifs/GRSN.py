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
from spikingjelly.clock_driven import surrogate

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
        B = x.shape[0]
        self._maybe_init_state(B, x.device, x.dtype)

        # Eq.17: 门控由 o_{t-1} 驱动
        F_gate = torch.sigmoid(self.forget_gate(self.spike_prev))
        I_gate = torch.relu(self.input_gate(self.spike_prev))
        self.c = F_gate * self.c + (1.0 - F_gate) * I_gate

        # 当前输入投影 + 门控电流 → 进入 LIF 积分
        current = self.input_proj(x) + self.c
        beta = torch.sigmoid(self.beta_raw)
        u = beta * h + (1.0 - beta) * current

        spike = self.surrogate_function(u - self.v_threshold)
        u_reset = u - spike * self.v_threshold

        self.spike_prev = spike
        return u_reset, spike


class GRSNNode(RecurrentSpikingWrapper):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=GRSNCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=1,
            rate_code=False,
        )
