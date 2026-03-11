import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate
from random import randint
import math


class AdaptiveCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 surrogate_function=surrogate.Sigmoid()):
        super().__init__()
        self.forget_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.surrogate_function = surrogate_function

        self.reset_parameters()

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
        F = torch.sigmoid(self.forget_gate(x))
        I = torch.relu(self.input_gate(x))
        # hidden = self.hidden_embd(h)
        # input = self.input_embd(x)
        h = F * h + (1 - F) * I
        spike = self.surrogate_function(h)
        h = (1 - spike) * h
        return h, spike


class RecurrentLIFNode(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.base_cell = AdaptiveCell
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
        for t in range(T):
            new_states_list = torch.zeros_like(states_list.data)
            spike_list = torch.zeros_like(states_list.data)
            new_states_list[0], spike_list[0] = self.cells[0](x[t], states_list[0])
            for i in range(1, self.num_layers):
                y = new_states_list[i - 1]
                new_states_list[i], spike_list[i] = self.cells[i](y, states_list[i])
            # output.append(new_states_list[-1].clone().unsqueeze(0))
            output.append(spike_list[-1].clone().unsqueeze(0))
            states_list = new_states_list.clone()
        return torch.cat(output, dim=0), new_states_list


if __name__ == "__main__":
    x = torch.ones((10, 16, 32), requires_grad=True)
    h = torch.zeros((2, 16, 16), requires_grad=True)
    snn = RecurrentLIFNode(32, 16, 2)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(125)
    # SNN = MultiStep(10, 4)
    # x = torch.zeros((15, 10, 10))
    # out, h = SNN(x)
    # print(out.shape, h.shape)
    for i in range(100):
        spike, h = snn(x*randint(1, 50), h)
        print(h, spike)
