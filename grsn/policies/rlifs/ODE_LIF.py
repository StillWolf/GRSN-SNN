import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron, functional
import math
from random import randint




if __name__ == "__main__":
    x = torch.ones((10, 16, 32), requires_grad=True)
    h = torch.zeros((2, 16, 16), requires_grad=True)
    snn = RecurrentLIFNode(32, 16, 1)
    for i in range(10):
        spike, h = snn(x*randint(1, 50), h)
        print(h, spike)
