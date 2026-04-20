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
        q_values = self.q_head(rnn_out)
        return q_values, new_state
