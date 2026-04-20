"""EpisodeBuffer 单测：roundtrip、padding、容量循环。"""
import numpy as np
import pytest
import torch

from grsn.buffers.episode_buffer import EpisodeBuffer


def _make_episode(L, n_agents, obs_dim, state_dim, n_actions, rng):
    return {
        "obs": rng.randn(L + 1, n_agents, obs_dim).astype(np.float32),
        "state": rng.randn(L + 1, state_dim).astype(np.float32),
        "actions": rng.randint(0, n_actions, size=(L, n_agents)).astype(np.int64),
        "avail_actions": np.ones((L + 1, n_agents, n_actions), dtype=np.int64),
        "rewards": rng.randn(L, 1).astype(np.float32),
        "terminated": np.concatenate(
            [np.zeros((L - 1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)]
        ),
    }


def test_insert_then_sample_roundtrip():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=10,
        n_agents=3, obs_dim=5, state_dim=7, n_actions=4,
    )
    rng = np.random.RandomState(0)
    ep = _make_episode(8, 3, 5, 7, 4, rng)
    buf.insert(ep)
    assert len(buf) == 1
    batch = buf.sample(1)
    assert batch["obs"].shape == (1, 11, 3, 5)
    assert batch["state"].shape == (1, 11, 7)
    assert batch["actions"].shape == (1, 11, 3)
    np.testing.assert_allclose(batch["rewards"][0, :8].numpy(), ep["rewards"], atol=0)


def test_padding_mask_marks_only_real_transitions():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=10,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    L = 5
    ep = _make_episode(L, 2, 3, 4, 5, rng)
    buf.insert(ep)
    batch = buf.sample(1)
    filled = batch["filled"][0, :, 0].numpy()
    assert filled[:L].sum() == L
    assert filled[L:].sum() == 0


def test_buffer_overwrites_oldest_when_full():
    buf = EpisodeBuffer(
        buffer_size=2, episode_limit=5,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    ep1 = _make_episode(3, 2, 3, 4, 5, rng)
    ep2 = _make_episode(3, 2, 3, 4, 5, rng)
    ep3 = _make_episode(3, 2, 3, 4, 5, rng)
    buf.insert(ep1)
    buf.insert(ep2)
    buf.insert(ep3)
    assert len(buf) == 2


def test_sample_rejects_too_many():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=5,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    buf.insert(_make_episode(3, 2, 3, 4, 5, rng))
    with pytest.raises(AssertionError):
        buf.sample(2)


def test_insert_episode_longer_than_limit_fails():
    buf = EpisodeBuffer(
        buffer_size=4, episode_limit=5,
        n_agents=2, obs_dim=3, state_dim=4, n_actions=5,
    )
    rng = np.random.RandomState(0)
    with pytest.raises(AssertionError):
        buf.insert(_make_episode(6, 2, 3, 4, 5, rng))
