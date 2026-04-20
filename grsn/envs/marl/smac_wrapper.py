"""真实 SMAC 环境的薄封装。

**运行时依赖：** 需要 StarCraft II 游戏二进制 + SMAC map pack + smac Python 包。
安装步骤见 docs/MARL_EXTENSION.md。本仓库默认不安装这些依赖——测试用
`MockSMAC` 替代。
"""
from typing import Dict, Tuple

import numpy as np

from grsn.envs.marl.base import MARLEnv


class SMACWrapper(MARLEnv):
    """把 smac.env.StarCraft2Env 适配到 MARLEnv 接口。

    Args:
        map_name: SMAC map 名，例如 "8m" / "2s3z" / "3s_vs_5z"
        seed: 随机种子
        **smac_kwargs: 额外 kwargs 传给 StarCraft2Env
    """

    def __init__(self, map_name: str, seed: int = 0, **smac_kwargs):
        try:
            from smac.env import StarCraft2Env
        except ImportError as e:
            raise ImportError(
                "smac 未安装。安装步骤：\n"
                "  1. pip install smac\n"
                "  2. 下载 StarCraft II（见 docs/MARL_EXTENSION.md）\n"
                "  3. 安装 SMAC map pack 到 $SC2PATH/Maps/SMAC_Maps/"
            ) from e

        self._env = StarCraft2Env(map_name=map_name, seed=seed, **smac_kwargs)
        env_info = self._env.get_env_info()
        self._env_info = env_info
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_dim = env_info["obs_shape"]
        self.state_dim = env_info["state_shape"]
        self.episode_limit = env_info["episode_limit"]

    def reset(self) -> None:
        self._env.reset()

    def step(self, actions: np.ndarray) -> Tuple[float, bool, Dict]:
        reward, terminated, info = self._env.step(actions)
        return float(reward), bool(terminated), dict(info)

    def get_obs(self) -> np.ndarray:
        return np.asarray(self._env.get_obs(), dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return np.asarray(self._env.get_state(), dtype=np.float32)

    def get_avail_actions(self) -> np.ndarray:
        return np.asarray(self._env.get_avail_actions(), dtype=np.int64)

    def get_env_info(self) -> Dict:
        return dict(self._env_info)

    def close(self) -> None:
        self._env.close()
