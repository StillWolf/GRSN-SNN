import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron, functional
import math
from random import randint


class LIFCell(neuron.LIFNode):
    def __init__(self, hidden_size, tau=2.0, decay_input=True, v_threshold=1., v_reset=None, surrogate_function=surrogate.ATan(),
                 detach_reset=False):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.hidden_size = hidden_size

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
        self.v = h
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return self.v, spike


class RecurrentLIFNode(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.base_cell = LIFCell
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = self.create_cells(hidden_size, num_layers)
        self.linear_ih = nn.Linear(input_size, hidden_size)
        self.time_step = 4

    def create_cells(self, hidden_size, num_layers):
        cells = []
        cells.append(self.base_cell(hidden_size))
        for i in range(num_layers - 1):
            cells.append(self.base_cell(hidden_size))
        return nn.Sequential(*cells)

    def forward(self, x, states=None):
        T = x.shape[0]
        batch_size = x.shape[1]
        output = []
        if states is None:
            states_list = torch.zeros(size=[self.num_layers, batch_size, self.hidden_size]).to(x)
        else:
            states_list = states
        x = self.linear_ih(x)
        current_state = states_list.clone()
        for t in range(T):
            new_states_list = torch.zeros_like(states_list.data)
            spike_list = torch.zeros_like(states_list.data)
            for k in range(self.time_step):
                current_state[0], spike = self.cells[0](x[t], current_state[0])
                new_states_list[0] = new_states_list[0] + current_state[0]
                spike_list[0] = spike_list[0] + spike
            spike_list[0] = spike_list[0] / self.time_step
            new_states_list[0] = new_states_list[0] / self.time_step
            for i in range(1, self.num_layers):
                y = new_states_list[i - 1]
                for k in range(self.time_step):
                    current_state[i], spike = self.cells[i](y, current_state[i])
                    new_states_list[i] = new_states_list[i] + current_state[i]
                    spike_list[i] = spike_list[i] + spike
                spike_list[i] = spike_list[i] / self.time_step
                new_states_list[i] = new_states_list[i] / self.time_step
            # output.append(new_states_list[-1].clone().unsqueeze(0))
            output.append(spike_list[-1].clone().unsqueeze(0))
            states_list = new_states_list.clone()
        return torch.cat(output, dim=0), new_states_list


if __name__ == "__main__":
    x = torch.ones((10, 16, 32), requires_grad=True)
    h = torch.zeros((2, 16, 16), requires_grad=True)
    snn = RecurrentLIFNode(32, 16, 2)
    for i in range(100):
        spike, h = snn(x*randint(1, 50), h)
        print(h, spike)
