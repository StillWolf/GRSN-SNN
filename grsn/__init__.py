"""
GRSN: Gate Recurrent Spiking Neuron for POMDP Reinforcement Learning

This package implements SNN-based RL agents for partially observable environments.
Based on the ICML 2022 paper on Recurrent Model-Free RL for POMDPs.
"""

__version__ = "1.0.0"
__author__ = "GRSN Team"

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
