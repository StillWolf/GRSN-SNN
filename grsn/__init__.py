"""
GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL

本仓库实现论文 arXiv:2404.15597 (AAAI'25, Qin et al.) 的 POMDP 部分。
MARL 部分（QMIX + SMAC）未实现，见 docs/MARL_EXTENSION.md。

架构骨架参考：
- pomdp-baselines (ICML'22, Ni et al.) 的 Separate Recurrent Actor-Critic
- spikingjelly 的 clock_driven 神经元实现
"""

__version__ = "1.0.0"

__all__ = [
    "Policy_RNN",
    "Policy_SNN",
    "Policy_MLP",
]


def __getattr__(name):
    if name == "Policy_RNN":
        from grsn.policies.policy_rnn import ModelFreeOffPolicy_Separate_RNN
        return ModelFreeOffPolicy_Separate_RNN
    if name == "Policy_SNN":
        from grsn.policies.policy_snn import ModelFreeOffPolicy_Separate_SNN
        return ModelFreeOffPolicy_Separate_SNN
    if name == "Policy_MLP":
        from grsn.policies.policy_mlp import ModelFreeOffPolicy_MLP
        return ModelFreeOffPolicy_MLP
    raise AttributeError(f"module 'grsn' has no attribute {name!r}")
