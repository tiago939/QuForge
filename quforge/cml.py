import torch
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)


class ReLU(nn.ReLU):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return torch.relu(x)


class Sigmoid(nn.Sigmoid):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Tanh(nn.Tanh):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope)
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.nn.LeakyReLU(x, negative_slope=self.negative_slope)


class Softmax(nn.Softmax):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


def randn(shape, device='cpu', dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=dtype)
