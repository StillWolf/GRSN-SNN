"""验证 MockSMAC 遵守 MARLEnv 接口。"""
import numpy as np
import pytest

from grsn.envs.marl import MARLEnv, MockSMAC


def test_mock_smac_is_marl_env():
    env = MockSMAC()
    assert isinstance(env, MARLEnv)


def test_env_info_contains_required_keys():
    env = MockSMAC()
    info = env.get_env_info()
    for k in ["n_agents", "n_actions", "obs_shape", "state_shape", "episode_limit"]:
        assert k in info


def test_reset_then_getters_return_correct_shapes():
    env = MockSMAC(n_agents=5, n_actions=7, obs_dim=11, state_dim=23)
    env.reset()
    assert env.get_obs().shape == (5, 11)
    assert env.get_state().shape == (23,)
    assert env.get_avail_actions().shape == (5, 7)
    assert env.get_obs().dtype == np.float32


def test_step_terminates_within_horizon_window():
    """MockSMAC 的 horizon 在 [20, episode_limit] 间采样。"""
    env = MockSMAC(episode_limit=30, seed=7)
    env.reset()
    terminated = False
    count = 0
    while not terminated and count < 100:
        reward, terminated, _ = env.step(np.zeros(env.n_agents, dtype=np.int64))
        count += 1
    assert terminated is True
    assert 20 <= count <= 30


def test_step_rejects_wrong_action_shape():
    env = MockSMAC(n_agents=4)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(np.zeros(3, dtype=np.int64))


def test_seed_makes_episode_deterministic():
    a = MockSMAC(seed=123)
    a.reset()
    obs_a = a.get_obs()
    b = MockSMAC(seed=123)
    b.reset()
    obs_b = b.get_obs()
    np.testing.assert_array_equal(obs_a, obs_b)
