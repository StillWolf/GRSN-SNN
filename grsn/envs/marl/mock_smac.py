"""冒烟测试用 mock SMAC 环境。

默认维度对应 SMAC 的 8m map：n_agents=8, n_actions=14, obs_dim=80, state_dim=168。
obs/state/reward 均为随机，只用于验证训练 loop 可跑通，不具训练意义。
"""
from typing import Dict, Tuple

import numpy as np

from grsn.envs.marl.base import MARLEnv


class MockSMAC(MARLEnv):
    def __init__(
        self,
        n_agents: int = 8,
        n_actions: int = 14,
        obs_dim: int = 80,
        state_dim: int = 168,
        episode_limit: int = 60,
        seed: int = 0,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.episode_limit = episode_limit
        self._rng = np.random.RandomState(seed)
        self._step = 0
        self._horizon = 0  # 本 episode 的实际长度

    def reset(self) -> None:
        self._step = 0
        self._horizon = int(self._rng.randint(20, self.episode_limit + 1))

    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        assert actions.shape == (self.n_agents,)
        self._step += 1
        reward = float(self._rng.randn())
        terminated = self._step >= self._horizon
        return reward, terminated, {}

    def get_obs(self) -> np.ndarray:
        return self._rng.randn(self.n_agents, self.obs_dim).astype(np.float32)

    def get_state(self) -> np.ndarray:
        return self._rng.randn(self.state_dim).astype(np.float32)

    def get_avail_actions(self) -> np.ndarray:
        return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

    def get_env_info(self) -> Dict:
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_dim,
            "state_shape": self.state_dim,
            "episode_limit": self.episode_limit,
        }
