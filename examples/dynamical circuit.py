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

circuit = qf.Circuit(dim=2, wires=2, device=device)

circuit.add(qf.HGate)
circuit.add(qf.RGate, mtx_id=0)
circuit.add(qf.CNOT)

x = State('0-0', device=device)

output = circuit(x)




