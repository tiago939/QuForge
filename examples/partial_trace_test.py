import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import random

sys.path.append('../')
from quforge import gates as ops
from quforge.gates import State as State

D = 15

x1 = 0.0
for i in range(D):
    c = 2*random.random()-1 + 1j*(2*random.random()-1)
    x1 += c*State(str(i))
x1 = x1/(torch.sum((abs(x1)**2))**0.5)

x2 = 0.0
for i in range(D):
    c = 2*random.random()-1 + 1j*(2*random.random()-1)
    x2 += c*State(str(i))
x2 = x2/(torch.sum((abs(x2)**2))**0.5)

x3 = 0.0
for i in range(D):
    c = 2*random.random()-1 + 1j*(2*random.random()-1)
    x3 += c*State(str(i))
x3 = x3/(torch.sum((abs(x3)**2))**0.5)

x = torch.kron(x1, x2)
x = torch.kron(x, x3)

d = ops.density_matrix(x1)
m = ops.partial_trace(x, index=[0])

print('Error=', abs(torch.sum(d-m)).item())




