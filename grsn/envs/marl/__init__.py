"""MARL 环境包。

- MARLEnv：抽象基类
- MockSMAC：冒烟测试用的随机 env
- SMACWrapper：真实 SMAC 的薄封装（import 时才会 raise ImportError）

注意 SMACWrapper 没放进默认 __all__，以免 `from grsn.envs.marl import *`
触发 smac 依赖。需要时显式写 `from grsn.envs.marl.smac_wrapper import SMACWrapper`。
"""
from grsn.envs.marl.base import MARLEnv
from grsn.envs.marl.mock_smac import MockSMAC

__all__ = ["MARLEnv", "MockSMAC"]
