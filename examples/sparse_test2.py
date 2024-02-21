import torch
import itertools
import sys

sys.path.append('../')
from quforge import gates as ops
from quforge.gates import State as State
from quforge import sparse

x = State('1', D=4)

U = sparse.Hadamard(D=4)
y = U(x)
print(y)


