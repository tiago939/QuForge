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

        self.H1 = ops.Hadamard(index=0)
        self.H2 = ops.Hadamard(index=1)
        self.RX1 = ops.Rotation(mtx_id=0, j=0, k=1, index=0)
        self.RX2 = ops.Rotation(mtx_id=0, j=0, k=1, index=1)
        self.cnot = ops.CNOT(control=0, target=2, N=3)

    def forward(self, x, angle):

        R = ops.Rotation(mtx_id=0, j=0, k=1, angle=angle)

        x = self.H1(x)
        x = self.H2(x)
        x = self.RX1(x)
        x = self.RX2(x)
        x = self.cnot(x)

        return x 

device='cuda'

model = Circuit().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

entrada = (State('000') + State('111'))/(2**0.5)
entrada = entrada.to(device)

for epochs in range(1):

    saida = model(entrada, 0.5)
    m = ops.measure(saida, index=[0])

    mean = ops.mean(saida, 'Z', index=0)
    
    loss = torch.sum((0.0-abs(mean))**2)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    print(model.RX1.angle.grad)
    optimizer.step()






