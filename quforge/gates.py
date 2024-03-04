import torch
import torch.nn as nn
import numpy as np 
from math import log as log
import itertools

pi = np.pi

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def State(dits, D=2, device='cpu'):
    base = torch.zeros((D, D, 1), device=device)
    for i in range(D):
        base[i][i] = 1.0 + 1j*0.0

    state = torch.eye(1, dtype=torch.complex64, device=device)
    st = ''
    for i in range(len(dits)):
        s = dits[i]
        if s.isdigit() is False: 
            state = torch.kron(state, base[int(st)])
            st = ''
        elif i == len(dits)-1:
            st += s
            state = torch.kron(state, base[int(st)])
        else:
            st += s
    return state


def density_matrix(state):
    rho = torch.matmul(state, torch.conj(state).T)
    return rho


def partial_trace(state, index):
    #index: list of qudits to take the partial trace over
    rho = density_matrix(state)
    N = round(log(state.shape[0], D))
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
                u = torch.kron(u, State(dits=str(i)).to(state.device))
        U += torch.matmul(u.T, torch.matmul(rho, u))
    
    return U


def projector(index):
    P = torch.zeros((D, D), dtype=torch.complex64)
    P[index][index] = 1.0

    return P


def measure(state=None, index=[0], shots=1):
    #input:
        #state: state to measure
        #index: list of qudits to measure
        #shots: number of measurements
    #output:
        #histogram: histogram of the measurements
        #p: distribution probability
    rho = partial_trace(state, index)
    p = abs(torch.diag(rho))
    p = p/torch.sum(p)

    a = np.array(range(len(rho)))
    positions = np.random.choice(a, p=p.detach().cpu().numpy(), size=shots)

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

    return histogram, p
        

def project(state, index=[0]):
    p = [(abs(state[i])**2).item() for i in range(len(state))]
    p = p/np.sum(p)

    a = np.array(range(len(state)))
    position = np.random.choice(a, p=p, size=1)[0]

    L = list(itertools.product(range(D), repeat=int(log(state.shape[0], D))))[position]
    U = torch.eye(1, device=state.device)
    counter = 0
    size = int(log(state.shape[0], D))
    for i in range(size):
        if i not in index:
            U = torch.kron(U, torch.eye(D, device=state.device))
        else:
            U = torch.kron(U, projector(L[i]).to(state.device))
            counter += 1

    state = torch.matmul(U, state)
    state = state/(torch.sum(abs(state)**2)**0.5)
    
    return state


def mean(state, observable='Z', index=0):

    if isinstance(observable, str):
        if observable == 'Z':
            U = ZGate(index=0, device=state.device)
    elif isinstance(observable, np.ndarray):
        M = torch.tensor(observable).to(state.device)
        U = CustomGate(M, index)
    else:
        M = observable.to(state.device)
        U = CustomGate(M, index)

    output = torch.matmul(state.T, U(state))[0][0]

    return output


def Sx(j, k, base):
    return torch.kron(base[j], base[k].T) + torch.kron(base[k], base[j].T) + 0*1j

def Sy(j, k, base):
    return -1j*torch.kron(base[j], base[k].T) + 1j*torch.kron(base[k], base[j].T)

def Sz(j, base):
    #return torch.kron(base[j], base[j].T) - torch.kron(base[k], base[k].T) + 0*1j
    f = (2.0/(j*(j+1)))**0.5
    s = 0.0
    for k in range(0, j+1):
        s += ((-j)**delta(k,j))*torch.kron(base[k], base[k].T)
    return f*s + 0*1j

sigma = [Sx, Sy, Sz]

def base(D, device='cpu'):
    base = torch.eye(D, device=device).reshape((D,D,1))
    return base

class Circuit(nn.Module):
    '''This class allows users to add gates dynamically'''
    def __init__(self, dim=2, wires=1, device='cpu'):
        super(Circuit, self).__init__()

        self.dim = dim 
        self.wires = wires
        self.device = device
        self.circuit = nn.Sequential()

    def add(self, module, **kwargs):
        gate = module(D=self.dim, device=self.device, **kwargs)
        self.circuit.add_module(str(len(self.circuit)), gate)

    def add_gate(self, gate, **kwargs):
        self.circuit.add_module(str(len(self.circuit)), gate)

    def H(self, **kwargs):
        self.add_gate(HGate(D=self.dim, device=self.device, **kwargs))

    def R(self, **kwargs):
        self.add_gate(RGate(D=self.dim, device=self.device, **kwargs))

    def CNOT(self, **kwargs):
        self.add_gate(CNOT(D=self.dim, device=self.device, **kwargs))

    def X(self, **kwargs):
        self.add_gate(XGate(D=self.dim, device=self.device, **kwargs))

    def Y(self, **kwargs):
        self.add_gate(YGate(D=self.dim, device=self.device, **kwargs))

    def Z(self, **kwargs):
        self.add_gate(ZGate(D=self.dim, device=self.device, **kwargs))

    def SWAP(self, **kwargs):
        self.add_gate(SWAP(D=self.dim, device=self.device, **kwargs))

    def forward(self, x):
        return self.circuit(x)


class CustomGate(nn.Module):
    def __init__(self, M, index=0):
        super(CustomGate, self).__init__()
        self.M = M.type(torch.complex64)
        self.index = index

    def forward(self, x):
        L = round(log(x.shape[0], D))
        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(D, dtype=torch.complex64, device=x.device))
        
        return torch.matmul(U, x)


class RGate(nn.Module):
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, mtx_id=0, j=0, k=1, index=[0], D=2, device='cpu', angle=False):
        super(RGate, self).__init__()

        self.D = D
        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = index

        if angle is False:
            self.angle = nn.Parameter(4*pi*torch.rand(len(index), device=device))
        else:
            self.angle = angle
        self.index = index

        S = sigma[self.mtx_id](self.j, self.k, base(D, device=device))
        self.register_buffer('S', S)

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i in self.index:
                if param is False:
                    M = torch.matrix_exp(-0.5*1j*self.angle[i]*self.S)
                else:
                    M = torch.matrix_exp(-0.5*1j*self.param[i]*self.S)
                U = torch.kron(U, M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device))
        
        return torch.matmul(U, x)
    

class HGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, D=2, index=[0], inverse=False, device='cpu'):
        super(HGate, self).__init__()

        self.index = index
        self.device = device
        self.D = D
        omega = np.exp(2*1j*pi/D)

        M = torch.ones((D, D), dtype=torch.complex64, device=device)
        for i in range(1, D):
            for j in range(1, D):
                M[i, j] = omega**(j*i)
        M = M/(D**0.5)
        if inverse:
            M = torch.conj(self.M).T.contiguous()
        self.register_buffer('M', M)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=self.device, dtype=torch.complex64)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=self.device, dtype=torch.complex64))
        return torch.matmul(U, x)


class XGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, D=2, index=[0], device='cpu'):
        super(XGate, self).__init__()

        self.index = index
        self.D = D
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = torch.matmul(base(D)[j].T, base(D)[(i+s) % D])
        self.register_buffer('M', M)   
        
    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, dtype=torch.complex64, device=x.device))
        return torch.matmul(U, x)


class ZGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, D=2, s=1, index=[0], device='cpu'):
        super(ZGate, self).__init__()

        omega = np.exp(2*1j*pi/D)

        self.index = index
        self.D = D
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = (omega**(j*s))*delta(i,j)
        self.register_buffer('M', M)   
        
    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device,  dtype=torch.complex64))
        
        return torch.matmul(U, x)


class YGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, D=2, index=[0], device='cpu'):
        super(YGate, self).__init__()

        self.index = index
        self.D = D
        X = XGate(s=s, device=device).M
        Z = ZGate(device=device).M
        M = torch.matmul(Z, X)/1j
        self.register_buffer('M', M) 
        
    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device, dtype=torch.complex64))
        
        return torch.matmul(U, x)


class XdGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, D=2, index=0, device='cpu'):
        super(XdGate, self).__init__()

        self.D = D
        self.index = index
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = torch.matmul(base(D, device=device)[j].T, base(D, device=device)[(D-i) % D])
        self.register_buffer('M', M)   
        
    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device))
        return torch.matmul(U, x)


class Identity(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, N=1, device='cpu'):
        super(Identity, self).__init__()

        self.U = torch.eye(D**N, dytpe=torch.complex64, device=device)
        
    def forward(self, x):
        return torch.matmul(U, x)


class CNOT(nn.Module):
    #control: control qudit
    #target: target qudit
    #N: number of qudits
    def __init__(self, control=0, target=1, N=2, D=2, device='cpu', inverse=False):
        super(CNOT, self).__init__()        
        L = torch.tensor(list(itertools.product(range(D), repeat=N)))
        l2ns = L.clone()
        l2ns[:, target] = (l2ns[:, control] + l2ns[:, target]) % D
        indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64)).to(device)
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)    
        
    def forward(self, x):
        return torch.matmul(self.U, x)


class SWAP(nn.Module):
    #swap the state of two qudits
    def __init__(self, qudit1=0, qudit2=1, D=2, N=2, device='cpu'):
        super(SWAP, self).__init__()

        self.U1 = CNOT(control=qudit1, target=qudit2, N=N, device=device)
        self.U2 = CNOT(control=qudit2, target=qudit1, N=N, device=device)
        self.U3 = XdGate(index=qudit1, device=device, D=D)
        self.U4 = XdGate(index=qudit2, device=device, D=D)

    def forward(self, x):
        x = self.U4(x)
        x = self.U1(x)
        x = self.U3(x)
        x = self.U2(x)
        x = self.U3(x)
        x = self.U1(x)

        return x


class CCNOT(nn.Module):
    #Toffoli gate, also know as CCNOT
    #control_1: control of qudit 1
    #control_2: control of qudit 2
    #target: target of qudit 3
    #N: number of qudits
    def __init__(self, control_1=0, control_2=1, target=2, N=3, inverse=False):
        super(CCNOT, self).__init__()        
        L = torch.tensor(list(itertools.product(range(D), repeat=N)))
        l2ns = L.clone()
        l2ns[:, target] = (l2ns[:, control_1]*l2ns[:, control_2] + l2ns[:, target]) % D
        indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64))        
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)
        
    def forward(self, x):
        return torch.matmul(self.U, x)


class MCX(nn.Module):
    #multi-controlled cx gate
    #control: list of control qudits
    #target: qudit target
    #N: number of qudits
    def __init__(self, control=[0], target=1, N=3, inverse=False):
        super(MCX, self).__init__()        
        L = torch.tensor(list(itertools.product(range(D), repeat=N)))
        l2ns = L.clone()
        control_value = 1
        for i in range(len(control)):
            control_value *= l2ns[:, control[i]]
        l2ns[:, target] = (control_value + l2ns[:, target]) % D
        indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64))        
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)
        
    def forward(self, x):
        return torch.matmul(self.U, x)


class CR(nn.Module):
    '''
    Controlled rotation gate
    TO DO: optimize this gate
    '''
    def __init__(self, control=0, target=1, D=2, mtx_id=0, j=0, k=1, device='cpu'):
        super(CR, self).__init__()

        self.D = D
        self.control = control
        self.target = target

        self.angle = nn.Parameter(4*pi*torch.rand(1, device=device))
        S = sigma[mtx_id](j, k, base(D, device=device))
        self.register_buffer('S', S)
    
    def forward(self, x):
        L = round(log(x.shape[0], self.D))

        U = 0.0
        for d in range(self.D):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(L):
                if i == self.control:
                    u = torch.kron(u, base(self.D)[d] @ base(self.D)[d].T)
                elif i == self.target:
                    M = torch.matrix_exp(-0.5*1j*self.angle*d*self.S)
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.D, device=x.device, dtype=torch.complex64))
            U += u
        return U @ x

