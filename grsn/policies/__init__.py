from enum import Enum

__all__ = [
    "Policy_MLP",
    "Policy_RNN",
    "Policy_SNN",
    "AGENT_CLASSES",
    "AGENT_ARCHS",
]


def __getattr__(name):
    if name in ("Policy_MLP", "Policy_RNN", "Policy_SNN", "AGENT_CLASSES", "AGENT_ARCHS"):
        from .policy_mlp import ModelFreeOffPolicy_MLP as _Policy_MLP
        from .policy_rnn import ModelFreeOffPolicy_Separate_RNN as _Policy_RNN
        from .policy_snn import ModelFreeOffPolicy_Separate_SNN as _Policy_SNN
        global Policy_MLP, Policy_RNN, Policy_SNN, AGENT_CLASSES, AGENT_ARCHS
        Policy_MLP = _Policy_MLP
        Policy_RNN = _Policy_RNN
        Policy_SNN = _Policy_SNN
        AGENT_CLASSES = {
            "Policy_MLP": _Policy_MLP,
            "Policy_RNN": _Policy_RNN,
            "Policy_SNN": _Policy_SNN,
        }

        class AGENT_ARCHS(str, Enum):
            Markov = _Policy_MLP.ARCH
            Memory = _Policy_RNN.ARCH

        if name == "Policy_MLP":
            return _Policy_MLP
        if name == "Policy_RNN":
            return _Policy_RNN
        if name == "Policy_SNN":
            return _Policy_SNN
        if name == "AGENT_CLASSES":
            return AGENT_CLASSES
        if name == "AGENT_ARCHS":
            return AGENT_ARCHS
    raise AttributeError(f"module 'grsn.policies' has no attribute {name!r}")
