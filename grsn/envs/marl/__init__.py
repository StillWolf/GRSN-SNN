"""MARL 环境包。

- MARLEnv：抽象基类
- MockSMAC：冒烟测试用的随机 env
- SMACWrapper：真实 SMAC 的薄封装（仅在 import 时才加载 smac/SC2 依赖，本 Task 不做）
"""
from grsn.envs.marl.base import MARLEnv
from grsn.envs.marl.mock_smac import MockSMAC

__all__ = ["MARLEnv", "MockSMAC"]
