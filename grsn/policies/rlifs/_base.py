"""共享的多层 SNN RNN 包装器。

每个具体神经元类型（LIF / GRSN / 其 wo_TAP 变体）只需实现 `Cell`：
  Cell.state_size -> int（每层 state 张量的第三维大小）
  Cell.forward(x_t, state) -> (new_state, spike)

`RecurrentSpikingWrapper` 负责：
- 多层堆叠
- 沿 MDP 时间维 (T_mdp) 展开
- 可选的 SNN 仿真子步（time_step，用于 w/o TAP 的 rate coding）

状态管理：state 完全由外部张量承载（从 caller 传入，在 forward 结束时返回），
使得推理阶段（每步 T_mdp=1）门控/膜电位能跨 MDP step 保留。各 Cell 通过
`state_size` 声明自己需要的维度（LIF: hidden_size；GRSN: 3*hidden_size）。
"""
import torch
import torch.nn as nn


class RecurrentSpikingWrapper(nn.Module):
    """将一个 Cell 堆叠成多层 RNN 并按 MDP 时间维展开。

    Args:
        cell_cls: Cell 类。需实现 `state_size` 属性和 `forward(x, state)`。
        input_size, hidden_size, num_layers: 常规 RNN 维度。
        time_step: 单个 MDP step 内的 SNN 仿真子步数。
            - TAP 变体传 1；w/o TAP 传 4。
        rate_code: 若 True，对 time_step 个子步的 spike 做平均。
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
        # 每层 state 张量的第三维大小（所有 cell 同构，取第一个即可）
        self.state_size_per_layer = self.cells[0].state_size

    def _run_layer(self, cell, x_t, state):
        """单层在 (x_t, state) 上跑 time_step 个子步。"""
        spike_accum = None
        s = state
        for _ in range(self.time_step):
            s, spike = cell(x_t, s)
            if self.rate_code:
                spike_accum = spike if spike_accum is None else spike_accum + spike
        if self.rate_code:
            spike_out = spike_accum / self.time_step
        else:
            spike_out = spike
        return s, spike_out

    def forward(self, x, states=None):
        """按 MDP 时间维展开。

        Args:
            x: (T_mdp, B, input_size)
            states: (num_layers, B, state_size_per_layer) 或 None。
                - None 时用零初始化（等价于 episode 起点）。
                - 推理循环要把上一次 forward 返回的 final_states 传回来。

        Returns:
            spikes_last_layer: (T_mdp, B, hidden_size)
            final_states: (num_layers, B, state_size_per_layer)
        """
        T_mdp, B = x.shape[0], x.shape[1]
        if states is None:
            states = torch.zeros(self.num_layers, B, self.state_size_per_layer,
                                 dtype=x.dtype, device=x.device)

        outputs = []
        current_states = states
        for t in range(T_mdp):
            new_states = []
            layer_input = x[t]
            layer_spike = None
            for i, cell in enumerate(self.cells):
                s_new, layer_spike = self._run_layer(cell, layer_input, current_states[i])
                new_states.append(s_new)
                layer_input = layer_spike
            outputs.append(layer_spike.unsqueeze(0))
            current_states = torch.stack(new_states, dim=0)

        return torch.cat(outputs, dim=0), current_states
