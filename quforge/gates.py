import torch
import torch.nn as nn
import numpy as np 
from math import log as log
import itertools

D = 3

pi = np.pi
omega = np.exp(2*1j*pi/D)

base = torch.zeros((D, D, 1))
for i in range(D):
    base[i][i] = 1.0 + 1j*0.0

def delta(x,y):
    if x == y:
        return 1.0
    else:
        return 0.0

def State(dits, state=None, device='cpu'):
    s = torch.eye(1, dtype=torch.complex64)
    if state is None and dits is None:
        for dit in dits:
            s = torch.kron(s, base[0])

    elif state is None and type(dits) == str:
        for dit in dits:
            s = torch.kron(s, base[int(dit)])

    else:
        for dit in dits:
            if dit == 'h':
                s = torch.kron(s, state)
            else:
                s = torch.kron(s, base[int(dit)])

    s = s.to(device)
    return s


def density_matrix(state):
    rho = torch.matmul(state, state.T)
    return rho


def partial_trace(state, index):
    #index: list of qudits to take the partial trace over
    rho = density_matrix(state)
    N = int(log(state.shape[0], D))
    L = list(itertools.product(range(D), repeat=N-len(index)))
    P = []
    for l in L:
        p = []
        cnt = 0
        for i in range(N):
            if i in index:
                p.append('h')
            else:
                p.append(l[cnt])
                cnt += 1
        P.append(p)

    U = 0
    for p in P:
        u = torch.eye(1, device=state.device)
        for i in p:
            if i == 'h':
                u = torch.kron(u, torch.eye(D, dtype=torch.complex64, device=state.device))
            else:
                u = torch.kron(u, State(dits=str(i)))
        U += torch.matmul(u.T, torch.matmul(rho, u))
    
    return U


def projector(index):
    P = torch.zeros((D, D), dtype=torch.complex64)
    P[index][index] = 1.0

    return P


def measure(state=None, index=[0], shots=1):
    rho = partial_trace(state, index)
    p = [abs(rho[i][i]).item() for i in range(len(rho))]
    p = p/np.sum(p)

    a = np.array(range(len(rho)))
    positions = np.random.choice(a, p=p, size=shots)

    L = list(itertools.product(range(D), repeat=len(index)))
    histogram = dict()
    keys = []
    for l in L:
        key = ''
        for i in range(len(index)):
            key += str(l[i])
        keys.append(key)
        histogram[key] = 0
    for position in positions:
        histogram[keys[position]] += 1

    return histogram


def mean(state, observable='Z', index=0):

    if observable == 'Z':
        U = ZGate(index=0, device=state.device)

    output = torch.matmul(state.T, U(state))[0][0]

    return output
        

# def project(state, index=[0]):
#     rho = partial_trace(state, index)
#     p = [abs(rho[i][i]).item() for i in range(len(rho))]
#     p = p/np.sum(p)

#     a = np.array(range(len(rho)))
#     position = np.random.choice(a, p=p, size=1)[0]

#     L = list(itertools.product(range(D), repeat=len(index)))[position]
#     U = torch.eye(1)
#     counter = 0
#     size = int(log(state.shape[0], D))
#     for i in range(size):
#         if i not in index:
#             U = torch.kron(U, torch.eye(D))
#         else:
#             U = torch.kron(U, projector(L[counter]))
#             counter += 1

#     state = torch.matmul(U, state)
#     state = state/(torch.sum(abs(state)**2)**0.5)
    
#     return state, L

def project(state, index=[0]):
    p = [(abs(state[i])**2).item() for i in range(len(state))]
    p = p/np.sum(p)

    a = np.array(range(len(state)))
    position = np.random.choice(a, p=p, size=1)[0]

    L = list(itertools.product(range(D), repeat=int(log(state.shape[0], D))))[position]
    U = torch.eye(1)
    counter = 0
    size = int(log(state.shape[0], D))
    for i in range(size):
        if i not in index:
            U = torch.kron(U, torch.eye(D))
        else:
            U = torch.kron(U, projector(L[i]))
            counter += 1

    state = torch.matmul(U, state)
    state = state/(torch.sum(abs(state)**2)**0.5)
    
    return state, L


def Sx(j, k):
    return torch.kron(base[j], base[k].T) + torch.kron(base[k], base[j].T) + 0*1j

def Sy(j, k):
    return -1j*torch.kron(base[j], base[k].T) + 1j*torch.kron(base[k], base[j].T)

def Sz(j, k):
    return torch.kron(base[j], base[j].T) - torch.kron(base[k], base[k].T) + 0*1j

sigma = [Sx, Sy, Sz]

class Rotation(nn.Module):
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, mtx_id=0, j=0, k=1, index=0, angle=False):
        super(Rotation, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        if angle is False:
            self.angle = nn.Parameter(4*pi*torch.rand(1))
        else:
            self.angle = angle
        self.index = index

    def forward(self, x):
        S = sigma[self.mtx_id](self.j, self.k).to(x.device)
        L = int(log(x.shape[0], D))
        M = torch.matrix_exp(-0.5*1j*self.angle*S)
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, M)
            else:
                U = torch.kron(U, torch.eye(D, device=x.device))

        return torch.matmul(U, x)
    

class Hadamard(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, index=0, inverse=False):
        super(Hadamard, self).__init__()

        self.index = index

        self.M = torch.ones((D, D), dtype=torch.complex64)
        for i in range(1, D):
            for j in range(1, D):
                self.M[i, j] = omega**(j*i)
        self.M = self.M/(D**0.5)
        if inverse:
            self.M = torch.conj(self.M).T.contiguous()
        self.M = nn.Parameter(self.M).requires_grad_(False)

    def forward(self, x):
        L = int(log(x.shape[0], D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(D, device=x.device))
        
        return torch.matmul(U, x)


class XGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, index=0):
        super(XGate, self).__init__()

        self.index = index
        self.M = torch.zeros((D, D), dtype=torch.complex64)
        for i in range(D):
            for j in range(D):
                self.M[j][i] = torch.matmul(base[j].T, base[(i+s) % D])
        self.M = nn.Parameter(self.M).requires_grad_(False)
        
    def forward(self, x):
        L = int(log(x.shape[0], D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(D, device=x.device))

        return torch.matmul(U, x)


class ZGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, index=0, device='cpu'):
        super(ZGate, self).__init__()

        self.index = index
        self.M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                self.M[j][i] = (omega**(j*s))*delta(i,j)
        self.M = nn.Parameter(self.M).requires_grad_(False)
        
    def forward(self, x):
        L = int(log(x.shape[0], D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(D, device=x.device))
        
        return torch.matmul(U, x)


class CNOT(nn.Module):
    #control: control qudit
    #target: target qudit
    #N: number of qudits
    def __init__(self, control=0, target=1, N=2, inverse=False):
        super(CNOT, self).__init__()

        self.U = torch.zeros((D**N, D**N), dtype=torch.complex64)
        L = list(itertools.product(range(D), repeat=N))
        for l1 in L:
            i = int(''.join([str(k) for k in l1]), D)
            l1n = list(l1)
            bra = torch.eye(1)
            for k in l1n:
                bra = torch.kron(bra, base[k])
            bra = bra.T

            for l2 in L:
                j = int(''.join([str(k) for k in l2]), D)
                l2n = list(l2)
                l2n[target] = (l2n[control] + l2n[target]) % D
                ket = torch.eye(1)
                for k in l2n:
                    ket = torch.kron(ket, base[k])
                self.U[i][j] = torch.matmul(bra, ket)
        if inverse:
            self.U = torch.conj(self.U).T.contiguous()
        self.U = nn.Parameter(self.U).requires_grad_(False)

    def forward(self, x):
            
        return torch.matmul(self.U, x)

