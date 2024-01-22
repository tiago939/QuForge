import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time

sys.path.append('../')
from quforge import gates as ops
from quforge.gates import State as State


device='cpu'

x = State('0-0-0').to(device)

torch.cuda.synchronize()
t0 = time.time()
cnot = ops.CNOT(control=1, target=0, N=3).to(device)
torch.cuda.synchronize()
t1 = time.time()
print(t1-t0)

torch.cuda.synchronize()
t0 = time.time()
x = cnot(x)
t1 = time.time()
print(t1-t0)




