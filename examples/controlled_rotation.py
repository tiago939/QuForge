import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time

sys.path.append('../')
from quforge import gates as qf 

state = qf.State('1-0', D=3)

cr = qf.CR(D=3)

out = cr(state)
print(out)

 
