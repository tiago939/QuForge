import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

sys.path.append('../')
from quforge import gates as ops
from quforge.gates import State as State

class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()

        self.H = ops.Hadamard(index=[0, 1])
        self.RX = ops.Rotation(mtx_id=0, j=0, k=1, index=[0, 1])
        self.cnot = ops.CNOT(control=0, target=2, N=3)

    def forward(self, x, angle):

        R = ops.Rotation(mtx_id=0, j=0, k=1, angle=angle)

        x = self.H(x)
        x = self.RX(x)
        x = self.cnot(x)

        return x 

device='cuda'

model = Circuit().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

entrada = State('0-0')
entrada = entrada.to(device)
print(entrada)

for epochs in range(1):

    saida = model(entrada, 0.5)
    m, p = ops.measure(saida, index=[0])
    print(p)

    mean = ops.mean(saida, 'Z', index=0)
    
    loss = torch.sum((0.0-abs(mean))**2)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    print(model.RX.angle.grad)
    optimizer.step()






