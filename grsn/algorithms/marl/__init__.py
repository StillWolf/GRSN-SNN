"""MARL 算法包（QMIX + SMAC 接入）。

复现论文 arXiv:2404.15597 (AAAI'25) 的 MARL 部分——在 QMIX agent 网络位置
插入 GRSN 作为 RNN backbone。

使用方法见 docs/MARL_EXTENSION.md 和 experiments/train_marl.py。
"""
from grsn.algorithms.marl.qmix import QMIX
from grsn.algorithms.marl.mixer import QMixer
from grsn.algorithms.marl.agent_network import AgentNetwork

__all__ = ["QMIX", "QMixer", "AgentNetwork"]
