"""CTDE 整 episode replay buffer。

存储格式：每个 slot 存一整条 episode，所有字段 pad 到 episode_limit + 1 长度
（+1 是因为要存 next_obs/next_state）。用 filled_mask 区分真实 step 与 padding。

Sample 回的 batch 维度：(B, T, ...)，T = episode_limit + 1。
"""
from typing import Dict

import numpy as np
import torch


class EpisodeBuffer:
    """固定大小循环 buffer。每个 slot = 一整条 episode。"""

    def __init__(
        self,
        buffer_size: int,
        episode_limit: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
    ):
        self.buffer_size = buffer_size
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions

        T = episode_limit + 1
        self.T = T

        self.obs = np.zeros((buffer_size, T, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((buffer_size, T, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, T, n_agents), dtype=np.int64)
        self.avail_actions = np.zeros((buffer_size, T, n_agents, n_actions), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, T, 1), dtype=np.float32)
        self.terminated = np.zeros((buffer_size, T, 1), dtype=np.float32)
        self.filled = np.zeros((buffer_size, T, 1), dtype=np.float32)

        self._idx = 0
        self._n_stored = 0

    def insert(self, episode: Dict[str, np.ndarray]) -> None:
        """插入一条 episode。

        episode dict 必须包含以下 key，每个 array 第一维 L ≤ episode_limit：
            obs:           (L+1, n_agents, obs_dim)
            state:         (L+1, state_dim)
            actions:       (L,   n_agents)
            avail_actions: (L+1, n_agents, n_actions)
            rewards:       (L,   1)
            terminated:    (L,   1)
        """
        L = episode["rewards"].shape[0]
        assert L <= self.episode_limit, f"episode 长度 {L} 超过上限 {self.episode_limit}"
        slot = self._idx

        self.obs[slot, : L + 1] = episode["obs"]
        self.state[slot, : L + 1] = episode["state"]
        self.actions[slot, :L] = episode["actions"]
        self.avail_actions[slot, : L + 1] = episode["avail_actions"]
        self.rewards[slot, :L] = episode["rewards"]
        self.terminated[slot, :L] = episode["terminated"]
        self.filled[slot].fill(0.0)
        self.filled[slot, :L] = 1.0

        self.obs[slot, L + 1 :] = 0.0
        self.state[slot, L + 1 :] = 0.0
        self.actions[slot, L:] = 0
        self.avail_actions[slot, L + 1 :] = 0
        self.rewards[slot, L:] = 0.0
        self.terminated[slot, L:] = 0.0

        self._idx = (self._idx + 1) % self.buffer_size
        self._n_stored = min(self._n_stored + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机不放回采样 batch_size 个 episode，返回 torch tensor dict。"""
        assert batch_size <= self._n_stored, (
            f"请求 {batch_size} 条，但 buffer 只有 {self._n_stored} 条"
        )
        idx = np.random.choice(self._n_stored, size=batch_size, replace=False)

        def _to_torch(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[idx].copy())

        return {
            "obs": _to_torch(self.obs),
            "state": _to_torch(self.state),
            "actions": _to_torch(self.actions),
            "avail_actions": _to_torch(self.avail_actions),
            "rewards": _to_torch(self.rewards),
            "terminated": _to_torch(self.terminated),
            "filled": _to_torch(self.filled),
        }

    def __len__(self) -> int:
        return self._n_stored
