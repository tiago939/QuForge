import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time

sys.path.append('../')
from quforge import gates as ops
from quforge import sparse
from quforge.gates import State as State


class Circuit(nn.Module):
    def __init__(self, D):
        super(Circuit, self).__init__()

        self.layers = nn.Sequential(
            sparse.RGate(D=D, mtx_id=0, j=0, k=1, index=[0, 1]),
            sparse.CNOT(D=D, control=0, target=1, N=2),
            sparse.RGate(D=D, mtx_id=0, j=0, k=1, index=[0, 1]),
        )

    def forward(self, x):

        x = self.layers(x)

        return x


device='cuda'
D = 3
model = Circuit(D=D).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

state = State('0-0', D=D, device=device)
target = State('1-1', D=D, device=device)

for epochs in range(100):

    output = model(state)
    loss = torch.sum(abs(target - output)**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()





