"""MARL 环境抽象基类。

接口遵循 PyMARL / SMAC 约定：obs 和 state 通过各自 getter 访问；reward 是团队共享标量。
"""
import abc
from typing import Dict, Tuple

import numpy as np


class MARLEnv(abc.ABC):
    """多智能体环境基类。MockSMAC 和 SMACWrapper 都继承此类。"""

    @abc.abstractmethod
    def reset(self) -> None:
        """开始新 episode。"""

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        """执行一步，返回 (team_reward, terminated, info)。actions: (n_agents,) int。"""

    @abc.abstractmethod
    def get_obs(self) -> np.ndarray:
        """每个 agent 的局部 obs，shape (n_agents, obs_dim)。"""

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """全局 state（仅训练时 mixer 用），shape (state_dim,)。"""

    @abc.abstractmethod
    def get_avail_actions(self) -> np.ndarray:
        """每 agent 的可执行动作 mask，shape (n_agents, n_actions)，0/1 binary。"""

    @abc.abstractmethod
    def get_env_info(self) -> Dict:
        """返回 {n_agents, n_actions, obs_shape, state_shape, episode_limit}。"""

    def close(self) -> None:
        """子类可覆盖。默认 no-op。"""
