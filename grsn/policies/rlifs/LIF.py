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
            surrogate_function = surrogate.ATan(alpha=2.0)
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
        self.v = h
        self.neuronal_charge(self.linear_ih(x))
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return self.v, spike


class LIFNode(RecurrentSpikingWrapper):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=LIFCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=1,
            rate_code=False,
        )
