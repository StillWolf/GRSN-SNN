"""Mixer 形状与单调性测试。"""
import torch

from grsn.algorithms.marl.mixer import QMixer


def test_mixer_output_shape():
    mixer = QMixer(n_agents=4, state_dim=32, embed_dim=16)
    agent_qs = torch.randn(2, 5, 4)
    states = torch.randn(2, 5, 32)
    q_tot = mixer(agent_qs, states)
    assert q_tot.shape == (2, 5, 1)


def test_mixer_monotonicity_increasing_any_agent_q_never_decreases_q_tot():
    """论文核心约束：∂Q_tot / ∂Q_i ≥ 0。"""
    torch.manual_seed(0)
    mixer = QMixer(n_agents=4, state_dim=8, embed_dim=16)
    base_q = torch.randn(1, 1, 4)
    state = torch.randn(1, 1, 8)
    q_base = mixer(base_q, state).item()

    for i in range(4):
        perturbed = base_q.clone()
        perturbed[0, 0, i] += 0.5
        q_new = mixer(perturbed, state).item()
        assert q_new >= q_base - 1e-5, (
            f"agent {i} Q 增加后 Q_tot 反而下降：{q_base} -> {q_new}"
        )


def test_mixer_monotonicity_random_perturbations():
    torch.manual_seed(42)
    mixer = QMixer(n_agents=3, state_dim=5, embed_dim=8)
    for _ in range(30):
        base_q = torch.randn(1, 1, 3)
        state = torch.randn(1, 1, 5)
        delta = torch.relu(torch.randn(1, 1, 3))
        q_base = mixer(base_q, state).item()
        q_up = mixer(base_q + delta, state).item()
        assert q_up >= q_base - 1e-5


def test_mixer_gradient_flows_to_hypernet():
    mixer = QMixer(n_agents=4, state_dim=8)
    agent_qs = torch.randn(1, 1, 4, requires_grad=True)
    states = torch.randn(1, 1, 8)
    q_tot = mixer(agent_qs, states)
    q_tot.sum().backward()
    for name, p in mixer.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
