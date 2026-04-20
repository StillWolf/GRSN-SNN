"""神经元单元测试：验证与论文 arXiv:2404.15597 的关键等式对齐。

状态张量约定（Critical fix 后）：
- LIFCell.state_size = hidden_size
- GRSNCell.state_size = 3*hidden_size，前后依次是 [h, c, spike_prev]
"""
import pytest
import torch

from grsn.policies.rlifs import REGISTRY
from grsn.policies.rlifs.GRSN import GRSNCell


# --------------------------- 形状与注册 ---------------------------

@pytest.mark.parametrize("snn_type,expected_state_mult", [
    ("LIF", 1),
    ("LIFwoTAP", 1),
    ("GRSN", 3),
    ("GRSNwoTAP", 3),
])
def test_registry_and_shapes(snn_type, expected_state_mult):
    """所有注册的神经元都返回 (T, B, H) 脉冲 + (L, B, state_size*H) final state。"""
    hidden = 8
    cls = REGISTRY[snn_type]
    model = cls(input_size=4, hidden_size=hidden, num_layers=2)
    x = torch.randn(5, 3, 4)
    spikes, final_state = model(x)
    assert spikes.shape == (5, 3, hidden)
    assert final_state.shape == (2, 3, expected_state_mult * hidden)


# --------------------------- TAP / rate coding 行为 ---------------------------

def test_grsn_time_step_is_one():
    model = REGISTRY["GRSN"](4, 8, 1)
    assert model.time_step == 1
    assert model.rate_code is False


def test_grsn_wo_tap_uses_rate_coding():
    model = REGISTRY["GRSNwoTAP"](4, 8, 1)
    assert model.time_step == 4
    assert model.rate_code is True


def test_lif_time_step_is_one():
    model = REGISTRY["LIF"](4, 8, 1)
    assert model.time_step == 1
    assert model.rate_code is False


def test_lif_wo_tap_uses_rate_coding():
    model = REGISTRY["LIFwoTAP"](4, 8, 1)
    assert model.time_step == 4
    assert model.rate_code is True


# --------------------------- GRSN 论文等式 ---------------------------

def test_grsn_gate_inputs_are_previous_spike_not_current_x():
    """论文 Eq.17：F(o_{t-1}) / I(o_{t-1})。

    构造两个相同 spike_prev 但不同 x 的 state，门输出应相同；
    相同 x 但不同 spike_prev，门输出应不同。
    """
    torch.manual_seed(0)
    cell = GRSNCell(input_size=4, hidden_size=6)
    H = 6
    # state 布局：[h, c, spike_prev] 各 H
    B = 2
    # 固定 h 和 c，只改 spike_prev
    h = torch.zeros(B, H)
    c = torch.zeros(B, H)
    spike_prev_a = torch.zeros(B, H)
    spike_prev_b = torch.ones(B, H) * 0.7
    state_a = torch.cat([h, c, spike_prev_a], dim=-1)
    state_b = torch.cat([h, c, spike_prev_b], dim=-1)

    x = torch.randn(B, 4)
    new_state_a, _ = cell(x, state_a)
    new_state_b, _ = cell(x, state_b)

    # 门输出不同 → 新 c 不同
    c_a = new_state_a[:, H:2*H]
    c_b = new_state_b[:, H:2*H]
    assert not torch.allclose(c_a, c_b), "spike_prev 变化时 c 应该变化（否则门没用 spike_prev）"

    # 固定 spike_prev，只改 x：c 相同（因为 c 只依赖 spike_prev）
    x2 = torch.randn(B, 4)
    new_state_a2, _ = cell(x2, state_a)  # 同样 state_a（spike_prev=0）
    c_a2 = new_state_a2[:, H:2*H]
    # 对于 spike_prev=0：F_gate=sigmoid(b_f)，I_gate=relu(b_i)，c_new=F*0+(1-F)*I 不依赖 x
    assert torch.allclose(c_a, c_a2), "x 变化但 spike_prev 相同时 c 不应变（门不应依赖 x）"


def test_grsn_beta_is_learnable_parameter():
    """论文要求 β 可学习。"""
    cell = GRSNCell(input_size=4, hidden_size=6)
    assert isinstance(cell.beta_raw, torch.nn.Parameter)
    assert cell.beta_raw.requires_grad is True
    # β_raw 初始为 0 → β = sigmoid(0) = 0.5
    beta = torch.sigmoid(cell.beta_raw)
    assert torch.allclose(beta, torch.full_like(beta, 0.5))


def test_grsn_soft_reset_equation():
    """论文 Eq.16：û = u − ϑ·o。

    人工构造：让 u > v_threshold（100%发放），检查 u_reset = u - v_threshold*1。
    """
    cell = GRSNCell(input_size=4, hidden_size=6, v_threshold=1.0)
    # 把权重置零，bias 置大正值，让 u 极大一定发 spike
    with torch.no_grad():
        for lin in [cell.forget_gate, cell.input_gate, cell.input_proj]:
            lin.weight.zero_()
            lin.bias.zero_()
        cell.input_proj.bias.fill_(10.0)  # 让 input_proj(x) = 10
    H = 6
    B = 2
    h = torch.zeros(B, H)
    c = torch.zeros(B, H)
    spike_prev = torch.zeros(B, H)
    state = torch.cat([h, c, spike_prev], dim=-1)
    x = torch.zeros(B, 4)

    new_state, spike = cell(x, state)
    # u = β*h + (1-β)*current，β=0.5，h=0，current = 10 + c_new
    # c_new = F*0 + (1-F)*I，F=sigmoid(0)=0.5，I=relu(0)=0，故 c_new=0
    # 所以 u = 0.5 * 0 + 0.5 * 10 = 5.0
    # spike = surrogate(5 - 1) = 1
    # u_reset = 5 - 1*1 = 4
    h_new = new_state[:, 0:H]
    assert torch.all(spike == 1.0), f"应该全部发放，但 spike={spike}"
    assert torch.allclose(h_new, torch.full_like(h_new, 4.0), atol=1e-5), f"软复位错误：h_new={h_new[0,0].item()}，应为 4.0"


def test_grsn_gradient_flows_to_beta():
    """β_raw 必须收到梯度。"""
    torch.manual_seed(0)
    model = REGISTRY["GRSN"](input_size=4, hidden_size=6, num_layers=1)
    x = torch.randn(5, 2, 4) * 3  # 放大以确保有 spike
    spikes, _ = model(x)
    loss = spikes.sum()
    loss.backward()
    # 找到 beta_raw
    found_and_has_grad = False
    for name, p in model.named_parameters():
        if "beta_raw" in name:
            assert p.grad is not None, f"{name} 无梯度"
            found_and_has_grad = True
    assert found_and_has_grad, "未找到 beta_raw 参数"


# --------------------------- 推理态状态传递（Critical fix） ---------------------------

def test_grsn_state_propagates_across_sequential_calls():
    """推理场景：连续两次 T_mdp=1 调用，把 final_state 传回去应该和一次 T_mdp=2 调用结果相同。

    这是 Critical bug 的回归测试——修复前 spike_prev/c 每次调用都被 reset_net 清零。
    """
    torch.manual_seed(0)
    model = REGISTRY["GRSN"](input_size=4, hidden_size=6, num_layers=1)
    torch.manual_seed(123)
    x_full = torch.randn(2, 3, 4) * 2  # T=2, B=3

    # 一次性 T=2 forward
    spikes_full, state_full = model(x_full, states=None)

    # 分两次 T=1 forward，手动传递 state
    spikes_step0, state_after0 = model(x_full[0:1], states=None)
    spikes_step1, state_after1 = model(x_full[1:2], states=state_after0)

    spikes_split = torch.cat([spikes_step0, spikes_step1], dim=0)

    # 两种方式得到的 spike 序列必须相同
    assert torch.allclose(spikes_full, spikes_split, atol=1e-5), \
        "连续 act() 调用传递 state 的结果与一次性 forward 不一致——状态传播有问题"


def test_grsn_fresh_state_vs_continued_state_differ():
    """传入 None 和传入上一次的 state 应该给出不同结果（否则 state 没起作用）。"""
    torch.manual_seed(0)
    model = REGISTRY["GRSN"](input_size=4, hidden_size=6, num_layers=1)
    x = torch.randn(3, 2, 4) * 3
    _, state = model(x, states=None)

    torch.manual_seed(999)
    x2 = torch.randn(2, 2, 4) * 3
    _, state_cont = model(x2, states=state)
    _, state_fresh = model(x2, states=None)

    # state_cont 和 state_fresh 必须不同（如果完全一样，说明 state 没影响计算）
    diff = (state_cont - state_fresh).abs().sum().item()
    assert diff > 1e-4, f"传入 state 对结果无影响，diff={diff}"


# --------------------------- 无残留副作用 ---------------------------

def test_no_data_pth_written_during_forward(tmp_path, monkeypatch):
    """LIFwoTAP.py 之前有 torch.save('./data.pth') 副作用，这里确认修好了。"""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    for snn in ["LIF", "LIFwoTAP", "GRSN", "GRSNwoTAP"]:
        model = REGISTRY[snn](4, 6, 2)
        x = torch.randn(5, 2, 4)
        model(x)
    assert not (tmp_path / "data.pth").exists(), "某个神经元仍然向磁盘写 data.pth"
