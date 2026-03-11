import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron, functional
import math
from random import randint


class RecurrentCell(neuron.LIFNode):
    def __init__(self, input_size, hidden_size, tau=2.0, decay_input=True, v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                 detach_reset=False):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.forget_gate = nn.Linear(input_size, hidden_size)
        self.input_gate = nn.Linear(input_size, hidden_size)
        # self.spike_gate = nn.Linear(hidden_size, 2 * hidden_size)
        self.hidden_size = hidden_size
        self.reset_parameters()
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def neuronal_reset(self, spike):
        self.v = self.v - spike * self.v_threshold

    def forward(self, x, h=None, last_spike=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
        # if last_spike is None:
        #     last_spike = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
        # spike_rst = torch.split(self.spike_gate(last_spike), self.hidden_size, dim=1)
        F = torch.sigmoid(self.forget_gate(x))
        C = torch.relu(self.input_gate(x))
        hidden = F * h + (1 - F) * C

        self.neuronal_charge(hidden)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return hidden, spike


class RecurrentLIFNode(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.base_cell = RecurrentCell
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = self.create_cells(input_size, hidden_size, num_layers)

    def create_cells(self, input_size, hidden_size, num_layers):
        cells = []
        cells.append(self.base_cell(input_size, hidden_size))
        for i in range(num_layers - 1):
            cells.append(self.base_cell(hidden_size, hidden_size))
        return nn.Sequential(*cells)

    def forward(self, x, states=None):
        T = x.shape[0]
        batch_size = x.shape[1]
        output = []
        if states is None:
            states_list = torch.zeros(size=[self.num_layers, batch_size, self.hidden_size]).to(x)
        else:
            states_list = states
        # last_spike = torch.zeros_like(states_list.data)
        for t in range(T):
            new_states_list = torch.zeros_like(states_list.data)
            spike_list = torch.zeros_like(states_list.data)
            new_states_list[0], spike_list[0] = self.cells[0](x[t], states_list[0])
            for i in range(1, self.num_layers):
                y = new_states_list[i - 1]
                new_states_list[i], spike_list[i] = self.cells[i](y, states_list[i])
            output.append(spike_list[-1].clone().unsqueeze(0))
            states_list = new_states_list.clone()
            # last_spike = spike_list
        functional.reset_net(self.cells)
        return torch.cat(output, dim=0), new_states_list


if __name__ == "__main__":
    x = torch.ones((10, 16, 32), requires_grad=True)
    h = torch.zeros((2, 16, 16), requires_grad=True)
    snn = RecurrentLIFNode(32, 16, 2)
    for i in range(100):
        spike, h = snn(x*randint(1, 50), h)
        print(h, spike)
