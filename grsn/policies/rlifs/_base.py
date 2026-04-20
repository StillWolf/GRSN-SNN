"""共享的多层 SNN RNN 包装器。

每个具体神经元类型（LIF / GRSN / 其 wo_TAP 变体）只需实现 `Cell` 的 forward：
  Cell.forward(x_t, state) -> (new_state, spike)

`RecurrentSpikingWrapper` 负责：
- 多层堆叠
- 沿 MDP 时间维 (T_mdp) 展开
- 可选的 SNN 仿真子步（time_step，用于 w/o TAP 的 rate coding）
- 每次完整 forward 后 `functional.reset_net`（避免跨 batch 的脉冲/膜电位泄露）
"""
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional


class RecurrentSpikingWrapper(nn.Module):
    """将一个 Cell 堆叠成多层 RNN 并按 MDP 时间维展开。

    Args:
        cell_cls: Cell 类（例如 GRSNCell）。必须接受 (input_size, hidden_size) 并有 forward(x, h)。
        input_size, hidden_size, num_layers: 常规 RNN 维度。
        time_step: 单个 MDP step 内的 SNN 仿真子步数。
            - TAP 变体传 1（每个 MDP step 恰好一次仿真）
            - w/o TAP 变体传 4（paper: T=4 rate coding）
        rate_code: 若 True，对 time_step 个子步的 spike 做平均（paper 的 rate coding）。
            TAP 变体通常为 False（time_step=1 时等价），w/o TAP 设 True。
    """

    def __init__(self, cell_cls, input_size, hidden_size, num_layers,
                 time_step=1, rate_code=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.rate_code = rate_code

        cells = [cell_cls(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            cells.append(cell_cls(hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def _run_layer(self, cell, x_t, h):
        """单层在 (x_t, h) 上跑 time_step 个子步。"""
        spike_accum = None
        for _ in range(self.time_step):
            h, spike = cell(x_t, h)
            if self.rate_code:
                spike_accum = spike if spike_accum is None else spike_accum + spike
        if self.rate_code:
            spike_out = spike_accum / self.time_step
        else:
            spike_out = spike
        return h, spike_out

    def forward(self, x, states=None):
        """按 MDP 时间维展开。

        Args:
            x: (T_mdp, B, input_size)
            states: (num_layers, B, hidden_size) 或 None

        Returns:
            spikes_last_layer: (T_mdp, B, hidden_size)
            final_states: (num_layers, B, hidden_size)
        """
        T_mdp, B = x.shape[0], x.shape[1]
        if states is None:
            states = torch.zeros(self.num_layers, B, self.hidden_size,
                                 dtype=x.dtype, device=x.device)

        outputs = []
        current_states = states
        for t in range(T_mdp):
            new_states = []
            layer_input = x[t]
            layer_spike = None
            for i, cell in enumerate(self.cells):
                h, layer_spike = self._run_layer(cell, layer_input, current_states[i])
                new_states.append(h)
                layer_input = layer_spike
            outputs.append(layer_spike.unsqueeze(0))
            current_states = torch.stack(new_states, dim=0)

        functional.reset_net(self.cells)
        return torch.cat(outputs, dim=0), current_states
