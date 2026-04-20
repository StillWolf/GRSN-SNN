"""AgentNetwork 单测：参数共享、RNN 切换、GRSN state 连续性。"""
import pytest
import torch

from grsn.algorithms.marl.agent_network import AgentNetwork


@pytest.mark.parametrize("rnn_type,expected_state_size", [
    ("gru", 64),
    ("LIF", 64),
    ("LIFwoTAP", 64),
    ("GRSN", 3 * 64),
    ("GRSNwoTAP", 3 * 64),
])
def test_output_shapes_per_rnn_type(rnn_type, expected_state_size):
    net = AgentNetwork(obs_dim=10, n_actions=5, rnn_type=rnn_type, rnn_hidden_size=64)
    B = 8
    obs = torch.randn(3, B, 10)
    state = net.init_hidden(B)
    q, new_state = net(obs, state)
    assert q.shape == (3, B, 5)
    assert new_state.shape == (1, B, expected_state_size)


def test_parameter_sharing_identical_obs_gives_identical_q():
    """把 n_agents 折进 batch 维：相同 obs → 相同 Q（因为权重共享）。"""
    torch.manual_seed(0)
    net = AgentNetwork(obs_dim=6, n_actions=4, rnn_type="gru", rnn_hidden_size=16)
    n_agents = 5
    B_actual = 2
    B_flat = B_actual * n_agents
    one_obs = torch.randn(3, 1, 6)
    obs = one_obs.expand(3, B_flat, 6).contiguous()
    state = net.init_hidden(B_flat)
    q, _ = net(obs, state)
    q0 = q[:, 0, :]
    for i in range(1, B_flat):
        torch.testing.assert_close(q[:, i, :], q0)


def test_grsn_state_propagation_split_equals_full():
    """GRSN 外部 state：两次 T=1 带 state 传递应该和一次 T=2 一次性前向等价。"""
    torch.manual_seed(0)
    net = AgentNetwork(obs_dim=4, n_actions=3, rnn_type="GRSN", rnn_hidden_size=8)
    B = 2
    obs_full = torch.randn(2, B, 4)
    state0 = net.init_hidden(B)

    q_full, _ = net(obs_full, state0)
    q_step0, state1 = net(obs_full[0:1], state0)
    q_step1, _ = net(obs_full[1:2], state1)
    q_split = torch.cat([q_step0, q_step1], dim=0)

    torch.testing.assert_close(q_full, q_split, atol=1e-5, rtol=1e-5)


def test_unknown_rnn_type_raises():
    with pytest.raises(ValueError, match="unknown rnn_type"):
        AgentNetwork(obs_dim=4, n_actions=3, rnn_type="lstm")
