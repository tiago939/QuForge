import torch
import numpy as np
import sys
import time

sys.path.append('../')
from quforge import gates as ops
from quforge.gates import State as State
from quforge import sparse

device = 'cuda'

def calculate_sparse_tensor_memory(U):
    indices_memory = U.indices().element_size() * U.indices().numel()
    values_memory = U.values().element_size() * U.values().numel()
    total_memory = indices_memory + values_memory
    return total_memory

Ds = [2**i for i in range(0,8)]
init_time = []
run_time = []
memory = []
for D in Ds:
    print(D)
    state = ops.State('0 0', D=D, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    cnot = ops.CNOT(control=0, target=1, D=D, device=device)
    torch.cuda.synchronize()
    t1 = time.time()
    print('Initialization time: ', t1-t0)
    init_time.append(t1-t0)

    torch.cuda.synchronize()
    t0 = time.time()
    y = cnot(state)
    torch.cuda.synchronize()
    t1 = time.time()
    print('Operation time: ', t1-t0)
    run_time.append(t1-t0)

    U = cnot.U
    size = 2*U.shape[0]*U.shape[0]*64/8
    #size = 2*U.coalesce().indices().shape[1]*64/8
    print('Size: ', size)
    memory.append(size)

np.save('init_time_dense.npy', init_time)
np.save('run_time_dense.npy', run_time)
np.save('memory_dense.npy', memory)