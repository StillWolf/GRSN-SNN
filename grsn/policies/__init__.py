from .policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from .policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from .policy_snn import ModelFreeOffPolicy_Separate_SNN as Policy_SNN

AGENT_CLASSES = {
    "Policy_MLP": Policy_MLP,
    "Policy_RNN": Policy_RNN,
    "Policy_SNN": Policy_SNN,
}

from enum import Enum


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Markov = Policy_MLP.ARCH
    Memory = Policy_RNN.ARCH
