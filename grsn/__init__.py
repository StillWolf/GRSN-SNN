"""
GRSN: Gate Recurrent Spiking Neuron for POMDP Reinforcement Learning

This package implements SNN-based RL agents for partially observable environments.
Based on the ICML 2022 paper on Recurrent Model-Free RL for POMDPs.
"""

__version__ = "1.0.0"
__author__ = "GRSN Team"

from grsn.policies.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from grsn.policies.policy_snn import ModelFreeOffPolicy_Separate_SNN as Policy_SNN
from grsn.policies.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP

__all__ = [
    "Policy_RNN",
    "Policy_SNN",
    "Policy_MLP",
]
