import torch
import torch.nn as nn
import sys
import time
import matplotlib.pyplot as plt

sys.path.append('../')
from quforge import gates as qf
from quforge import sparse

Ds = [2] + [2**i for i in range(1, 7)]

class CircuitDense(nn.Module):
    def __init__(self, D):
        super(CircuitDense, self).__init__()

        self.layers = nn.Sequential(
            qf.RGate(D=D, mtx_id=0, j=0, k=1, index=[0, 1]),
            qf.CNOT(D=D, control=0, target=1, N=2),
            qf.RGate(D=D, mtx_id=0, j=0, k=1, index=[0, 1]),
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class CircuitSparse(nn.Module):
    def __init__(self, D):
        super(CircuitSparse, self).__init__()

        self.layers = nn.Sequential(
            sparse.RGate(D=D, mtx_id=0, j=0, k=1, index=[0, 1]),
            sparse.CNOT(D=D, control=0, target=1, N=2),
            sparse.RGate(D=D, mtx_id=0, j=0, k=1, index=[0, 1]),
        )

    def forward(self, x):

        x = self.layers(x)

        return x


T_init_dense_cpu = []
T_init_sparse_cpu = []
T_init_dense_gpu = []
T_init_sparse_gpu = []

T_run_dense_cpu = []
T_run_sparse_cpu = []
T_run_dense_gpu = []
T_run_sparse_gpu = []

T_grad_dense_cpu = []
T_grad_sparse_cpu = []
T_grad_dense_gpu = []
T_grad_sparse_gpu = []


for D in Ds:
    print(D)
    device = 'cpu'

    state = qf.State('0-0', D=D, device=device)
    state.requires_grad_(True)

    t0 = time.time()
    U = CircuitDense(D=D).to(device)
    t1 = time.time()
    T_init_dense_cpu.append(t1-t0)

    t0 = time.time()
    output = U(state)
    t1 = time.time()
    T_run_dense_cpu.append(t1-t0)

    t0 = time.time()
    grad = torch.autograd.grad(output.sum(), state, create_graph=True, retain_graph=True)
    t1 = time.time()
    T_grad_dense_cpu.append(t1-t0)

    t0 = time.time()
    U = CircuitSparse(D=D).to(device)
    t1 = time.time()
    T_init_sparse_cpu.append(t1-t0)

    t0 = time.time()
    output = U(state)
    t1 = time.time()
    T_run_sparse_cpu.append(t1-t0)

    t0 = time.time()
    grad = torch.autograd.grad(output.sum(), state, create_graph=True, retain_graph=True)
    t1 = time.time()
    T_grad_sparse_cpu.append(t1-t0)

    device = 'cuda'

    state = qf.State('0-0', D=D, device=device)
    state.requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.time()
    U = CircuitDense(D=D).to(device)
    torch.cuda.synchronize()
    t1 = time.time()
    T_init_dense_gpu.append(t1-t0)

    torch.cuda.synchronize()
    t0 = time.time()
    output = U(state)
    torch.cuda.synchronize()
    t1 = time.time()
    T_run_dense_gpu.append(t1-t0)

    t0 = time.time()
    grad = torch.autograd.grad(output.sum(), state, create_graph=True, retain_graph=True)
    t1 = time.time()
    T_grad_dense_gpu.append(t1-t0)

    torch.cuda.synchronize()
    t0 = time.time()
    U = CircuitSparse(D=D).to(device)
    torch.cuda.synchronize()
    t1 = time.time()
    T_init_sparse_gpu.append(t1-t0)

    torch.cuda.synchronize()
    t0 = time.time()
    output = U(state)
    torch.cuda.synchronize()
    t1 = time.time()
    T_run_sparse_gpu.append(t1-t0)

    torch.cuda.synchronize()
    t0 = time.time()
    grad = torch.autograd.grad(output.sum(), state, create_graph=True, retain_graph=True)
    torch.cuda.synchronize()
    t1 = time.time()
    T_grad_sparse_gpu.append(t1-t0)

plt.plot(Ds[1::], T_init_dense_cpu[1::], label='dense cpu')
plt.plot(Ds[1::], T_init_sparse_cpu[1::], label='sparse cpu')
plt.plot(Ds[1::], T_init_dense_gpu[1::], label='dense gpu')
plt.plot(Ds[1::], T_init_sparse_gpu[1::], label='sparse gpu')

plt.legend()
plt.title('Circuit initialization time')
plt.xlabel('D')
plt.ylabel('t (s)')
plt.show()

plt.plot(Ds[1::], T_run_dense_cpu[1::], label='dense cpu')
plt.plot(Ds[1::], T_run_sparse_cpu[1::], label='sparse cpu')
plt.plot(Ds[1::], T_run_dense_gpu[1::], label='dense gpu')
plt.plot(Ds[1::], T_run_sparse_gpu[1::], label='sparse gpu')

plt.legend()
plt.title('Circuit executation time')
plt.xlabel('D')
plt.ylabel('t (s)')
plt.show()

plt.plot(Ds[1::], T_grad_dense_cpu[1::], label='dense cpu')
plt.plot(Ds[1::], T_grad_sparse_cpu[1::], label='sparse cpu')
plt.plot(Ds[1::], T_grad_dense_gpu[1::], label='dense gpu')
plt.plot(Ds[1::], T_grad_sparse_gpu[1::], label='sparse gpu')

plt.legend()
plt.title('Circuit gradient time')
plt.xlabel('D')
plt.ylabel('t (s)')
plt.show()




