import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time

sys.path.append('../')
from quforge import gates as qf
from quforge.gates import State as State

device='cuda'


class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()

        self.layers = nn.Sequential(
            ops.Hadamard(index=[0, 1]),
            ops.Rotation(mtx_id=0, j=0, k=1, index=[0, 1]),
            ops.CNOT(control=0, target=2, N=3)
        )

        self.R = ops.Rotation(mtx_id=0, j=0, k=1, index=[0])

    def forward(self, x):
        
        x = self.R(param=[0.5])
        x = self.layers(x)

        return x



circuit = qf.Circuit(dim=2, wires=2, device=device)
circuit.add(qf.HGate)
circuit.add(qf.RGate, mtx_id=0)
circuit.add(qf.CNOT)




circuit = qf.Circuit(dim=2, wires=2, device=device)
circuit.H()
circuit.R(angle=[0.5])
circuit.CNOT()
circuit.X(index=[1])
circuit.SWAP()

x = State('0-0', device=device)

output = circuit(x)




