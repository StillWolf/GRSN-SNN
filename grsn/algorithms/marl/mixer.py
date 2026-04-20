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

        self.hyper_w1 = _build_hypernet(n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
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
        agent_qs = agent_qs.reshape(B * T, 1, N)
        states = states.reshape(B * T, self.state_dim)

        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(B * T, N, self.embed_dim)
        b1 = self.hyper_b1(states).view(B * T, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(B * T, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B * T, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B, T, 1)
