import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron
from random import randint
import math


class LIFCell(neuron.LIFNode):
    def __init__(self, input_size, hidden_size, tau=2.0, decay_input=True, v_threshold=1., v_reset=None, surrogate_function=surrogate.ATan(),
                 detach_reset=False, test=False):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.linear_ih = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.test = test

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
        self.v = h
        y_ih = self.linear_ih(x)
        self.neuronal_charge(y_ih)
        v_wo_rst = self.v.detach()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        self.test = True
        if self.test:
            return self.v, spike, v_wo_rst
        else:
            return self.v, spike


# class LIFCell(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, bias=True,
#                  surrogate_function=surrogate.Sigmoid()):
#         super().__init__()
#         self.linear_ih = nn.Linear(input_size, hidden_size, bias=bias)
#         self.LIF = LIFNode(surrogate_function=surrogate_function)
#         self.hidden_size = hidden_size
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         sqrt_k = math.sqrt(1 / self.hidden_size)
#         for param in self.parameters():
#             nn.init.uniform_(param, -sqrt_k, sqrt_k)
#
#     def forward(self, x, h=None):
#         if h is None:
#             h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
#         y_ih = self.linear_ih(x)
#         self.LIF.v = h
#         spike = self.LIF(y_ih)
#         return self.LIF.v


class RecurrentLIFNode(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, test=True):
        super().__init__()
        self.base_cell = LIFCell
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = self.create_cells(input_size, hidden_size, num_layers, test)
        self.test = test

    def create_cells(self, input_size, hidden_size, num_layers, test):
        cells = []
        cells.append(self.base_cell(input_size, hidden_size, test=test))
        for i in range(num_layers - 1):
            cells.append(self.base_cell(hidden_size, hidden_size, test=test))
        return nn.Sequential(*cells)

    def forward(self, x, states=None):
        T = x.shape[0]
        batch_size = x.shape[1]
        output = []
        self.test = True
        if states is None:
            states_list = torch.zeros(size=[self.num_layers, batch_size, self.hidden_size]).to(x)
        else:
            states_list = states
        if self.test:
            volt_list = torch.zeros(size=[T, batch_size, self.hidden_size]).to(x)
            out_list = torch.zeros(size=[T, batch_size, self.hidden_size]).to(x)
        for t in range(T):
            new_states_list = torch.zeros_like(states_list.data)
            state_wo_rst_list = torch.zeros_like(states_list.data)
            spike_list = torch.zeros_like(states_list.data)
            if self.test:
                new_states_list[0], spike_list[0], state_wo_rst_list[0] = self.cells[0](x[t], states_list[0])
            else:
                new_states_list[0], spike_list[0] = self.cells[0](x[t], states_list[0])
            for i in range(1, self.num_layers):
                y = new_states_list[i - 1]
                if self.test:
                    new_states_list[i], spike_list[i], state_wo_rst_list[i] = self.cells[i](y, states_list[i])
                else:
                    new_states_list[i], spike_list[i] = self.cells[i](y, states_list[i])
            # output.append(new_states_list[-1].clone().unsqueeze(0))
            if self.test:
                volt_list[t] = state_wo_rst_list[-1].clone()
                out_list[t] = spike_list[-1].clone().unsqueeze(0)
            output.append(spike_list[-1].clone().unsqueeze(0))
            states_list = new_states_list.clone()
        if self.test and T != 1:
            state = {
                'volt_list': volt_list,
                'out_list': out_list,
                'alpha': 2.0,
                'beta': (1-1/self.cells[0].tau),
                'v_threshold': self.cells[0].v_threshold,
                'v_reset': self.cells[0].v_reset,
            }
            torch.save(state, './data.pth')
        return torch.cat(output, dim=0), new_states_list


if __name__ == "__main__":
    x = torch.ones((10, 16, 32), requires_grad=True)
    h = torch.zeros((2, 16, 16), requires_grad=True)
    snn = RecurrentLIFNode(32, 16, 2)
    for i in range(100):
        spike, h = snn(x*randint(1, 50), h)
        print(h, spike)
