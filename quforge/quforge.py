import torch
import torch.nn as nn
import numpy as np 
from math import log as log
from math import factorial as factorial
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


def partial_trace(state, index, D):
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
                u = torch.kron(u, State(dits=str(i), D=D).to(state.device))
        U += torch.matmul(u.T, torch.matmul(rho, u))
    
    return U


def projector(index, D):
    P = torch.zeros((D, D), dtype=torch.complex64)
    P[index][index] = 1.0

    return P


def measure(state=None, index=[0], shots=1, D=2):
    #input:
        #state: state to measure
        #index: list of qudits to measure
        #shots: number of measurements
    #output:
        #histogram: histogram of the measurements
        #p: distribution probability
    rho = partial_trace(state, index, D)
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
        

def project(state, index=[0], D=2):
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
            U = torch.kron(U, projector(L[i], D).to(state.device))
            counter += 1

    state = torch.matmul(U, state)
    state = state/(torch.sum(abs(state)**2)**0.5)
    
    return state, L


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


def Sx_old(j, k, base):
    return torch.kron(base[j-1], base[k-1].T) + torch.kron(base[k-1], base[j-1].T) + 0*1j

def Sy_old(j, k, base):
    return -1j*torch.kron(base[j-1], base[k-1].T) + 1j*torch.kron(base[k-1], base[j-1].T)

def Sz_old(j, k, base):
    #return torch.kron(base[j], base[j].T) - torch.kron(base[k], base[k].T) + 0*1j
    f = (2.0/(j*(j+1)))**0.5
    s = 0.0
    for k in range(0, j+1):
        s += ((-j)**delta(k,j))*torch.kron(base[k], base[k].T)
    return f*s + 0*1j

def Sx(j, k, D=2, device='cpu'):
    #0 <= j < k < D
    S = torch.zeros((D, D), device=device)
    S[j][k] = 1.0
    S[k][j] = 1.0
    return S

def Sy(j, k, D=2, device='cpu'):
    #0 <= j < k < D
    S = torch.zeros((D, D), device=device, dtype=torch.complex64)
    S[j][k] = -1j
    S[k][j] = 1j
    return S

def Sz(j, D=2, device='cpu'):
    #1 <= j < D
    f = (2.0/(j*(j+1)))**0.5
    S = torch.zeros((D,D), device=device)
    for k in range(0, j+1):
        S[k][k] = f*(-j)**delta(j, k)
    return S

sigma = [Sx, Sy, Sz]


def nQudit(n,indice):
    esquerda = indice
    direita = n-indice-1
    return esquerda,direita


def base(D, device='cpu'):
    base = torch.eye(D, device=device).reshape((D,D,1))
    return base


def power_even_x(x, p):
    s = 0.0
    for i in range(1, p+1):
        if i % 2 == 0:
            s += (x**i)/factorial(i)
    return s


def power_odd_x(x, p):
    s = 0.0
    for i in range(1, p+1):
        if i % 2 != 0:
            s += (x**i)/factorial(i)
    return s

def power_even_y(x, p):
    s = 0.0
    for i in range(1, p+1):
        if i % 2 == 0:
            s += (x**i)/factorial(i)
    return s

def power_odd_y_jk(x, p):
    s = 0.0
    for i in range(1, p+1):
        if i % 2 != 0:
            s += 1j*(x**i)/factorial(i)
    return s

def power_odd_y_kj(x, p):
    s = 0.0
    for i in range(1, p+1):
        if i % 2 != 0:
            s += (-1j)*(x**i)/factorial(i)
    return s

def fidelity(state1, state2):
    F = abs(torch.matmul(torch.conj(state1).T, state2))**2
    return F.real

def argmax(x):
    return torch.argmax(x)

def mean(x):
    return torch.mean(x)

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name, *args, **kwargs):
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(*args, **kwargs)
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(*args, **kwargs)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported")

class optim:
    Adam = staticmethod(lambda *args, **kwargs: OptimizerFactory.get_optimizer('adam', *args, **kwargs))
    SGD = staticmethod(lambda *args, **kwargs: OptimizerFactory.get_optimizer('sgd', *args, **kwargs))

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

class ModuleList(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(ModuleList, self).__init__(*args, **kwargs)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

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
        self.add_gate(CNOT(D=self.dim, N=self.wires, device=self.device, **kwargs))

    def X(self, **kwargs):
        self.add_gate(XGate(D=self.dim, device=self.device, **kwargs))

    def Y(self, **kwargs):
        self.add_gate(YGate(D=self.dim, device=self.device, **kwargs))

    def Z(self, **kwargs):
        self.add_gate(ZGate(D=self.dim, device=self.device, **kwargs))

    def SWAP(self, **kwargs):
        self.add_gate(SWAP(D=self.dim, device=self.device, **kwargs))

    def CZ(self, **kwargs):
        self.add_gate(CZGate(D=self.dim, N=self.wires, device=self.device, **kwargs))

    def CCNOT(self, **kwargs):
        self.add_gate(CCNOT(D=self.dim, device=self.device, **kwargs))

    def Custom(self, **kwargs):
        self.add_gate(CustomGate(D=self.dim, device=self.device, **kwargs))

    def forward(self, x):
        return self.circuit(x)


class CustomGate(nn.Module):
    def __init__(self, M, D=2, N=1, index=0, device='cpu'):
        super(CustomGate, self).__init__()
        self.M = M.type(torch.complex64).to(device)
        self.index = index
        self.D = D
        self.N = N

    def forward(self, x):
        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(self.N):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, dtype=torch.complex64, device=x.device))
        
        return torch.matmul(U, x)


class RXGate(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate
    #cutoff: cutoff of the infinite sum
    def __init__(self, j=0, k=1, index=[0], D=2, device='cpu', angle=False, cutoff=10):
        super(RXGate, self).__init__()

        self.D = D
        self.device = device
        self.index = index
        self.cutoff = cutoff
        self.j = j 
        self.k = k

        if angle is False:
            self.angle = nn.Parameter(torch.randn(len(index), device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(len(index), device=device))

        S = Sx(j=j, k=k, D=D, device=device)
        self.register_buffer('S', S)

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i in self.index:
                if param is False:
                    M = torch.eye(self.D, device=x.device, dtype=torch.complex64)
                    M[self.j][self.k] += power_odd_x(-0.5*1j*self.angle[i], self.cutoff)
                    M[self.k][self.j] += power_odd_x(-0.5*1j*self.angle[i], self.cutoff)
                    M[self.j][self.j] += power_even_x(-0.5*1j*self.angle[i], self.cutoff)
                    M[self.k][self.k] += power_even_x(-0.5*1j*self.angle[i], self.cutoff)
                else:
                    M = torch.eye(self.D, device=x.device, dtype=torch.complex64)
                    M[self.j][self.k] += power_odd_x(-0.5*1j*param[i], self.cutoff)
                    M[self.k][self.j] += power_odd_x(-0.5*1j*param[i], self.cutoff)
                    M[self.j][self.j] += power_even_x(-0.5*1j*param[i], self.cutoff)
                    M[self.k][self.k] += power_even_x(-0.5*1j*param[i], self.cutoff)
                U = torch.kron(U, M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device, dtype=torch.complex64))
        
        return torch.matmul(U, x)

class RXGate2(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate
    #cutoff: cutoff of the infinite sum
    def __init__(self, j=0, k=1, index=0, N=1, D=2, device='cpu', angle=False, cutoff=10):
        super(RXGate2, self).__init__()

        self.D = D
        self.device = device
        self.index = index
        self.cutoff = cutoff
        self.j = j 
        self.k = k

        if angle is False:
            self.angle = nn.Parameter(4*pi*torch.rand(1, device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(1, device=device))

        S = Sx(j=j, k=k, D=D, device=device)
        self.register_buffer('S', S)

        esq,dir = nQudit(N,index)

        I_esq = torch.eye(D**esq,device=device)
        I_dir = torch.eye(D**dir,device=device)

        self.register_buffer('I_esq', I_esq)
        self.register_buffer('I_dir', I_dir)

    def forward(self, x, param=False):
        M = torch.eye(self.D, device=x.device, dtype=torch.complex64)
        M[self.j][self.k] += power_odd_x(-0.5*1j*self.angle[0], self.cutoff)
        M[self.k][self.j] += power_odd_x(-0.5*1j*self.angle[0], self.cutoff)
        M[self.j][self.j] += power_even_x(-0.5*1j*self.angle[0], self.cutoff)
        M[self.k][self.k] += power_even_x(-0.5*1j*self.angle[0], self.cutoff)
        
        U = torch.kron(self.I_esq, M)
        U = torch.kron(U, self.I_dir)

        return torch.matmul(U, x)


class RYGate(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate
    #cutoff: cutoff of the infinite sum
    def __init__(self, j=0, k=1, index=[0], D=2, device='cpu', angle=False, cutoff=10):
        super(RYGate, self).__init__()

        self.D = D
        self.device = device
        self.index = index
        self.cutoff = cutoff
        self.j = j 
        self.k = k

        if angle is False:
            self.angle = nn.Parameter(torch.randn(len(index), device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(len(index), device=device))

        S = Sy(j=j, k=k, D=D, device=device)
        self.register_buffer('S', S)

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i in self.index:
                if param is False:
                    M = torch.eye(self.D, device=x.device, dtype=torch.complex64)
                    M[self.j][self.k] += power_odd_y_jk(-0.5*1j*self.angle[i], self.cutoff)
                    M[self.k][self.j] += power_odd_y_kj(-0.5*1j*self.angle[i], self.cutoff)
                    M[self.j][self.j] += power_even_y(-0.5*1j*self.angle[i], self.cutoff)
                    M[self.k][self.k] += power_even_y(-0.5*1j*self.angle[i], self.cutoff)
                else:
                    M = torch.eye(self.D, device=x.device, dtype=torch.complex64)
                    M[self.j][self.k] += power_odd_y_jk(-0.5*1j*param[i], self.cutoff)
                    M[self.k][self.j] += power_odd_y_kj(-0.5*1j*param[i], self.cutoff)
                    M[self.j][self.j] += power_even_y(-0.5*1j*param[i], self.cutoff)
                    M[self.k][self.k] += power_even_y(-0.5*1j*param[i], self.cutoff)
                U = torch.kron(U, M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device, dtype=torch.complex64))
        
        return torch.matmul(U, x)


class RZGate(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate
    #cutoff: cutoff of the infinite sum
    def __init__(self, j=1, index=[0], D=2, device='cpu', angle=False, cutoff=10):
        super(RZGate, self).__init__()

        self.D = D
        self.device = device
        self.index = index
        self.j = j 

        if angle is False:
            self.angle = nn.Parameter(torch.randn(len(index), device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(len(index), device=device))

        S = Sz(j=j, D=D, device=device)
        self.register_buffer('S', S)

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i in self.index:
                M = torch.eye(self.D, device=x.device, dtype=torch.complex64)
                if param is False:
                    for k in range(0, self.j+1):
                        M[k][k] = torch.exp(-0.5*1j*self.angle[i]*self.S[k][k])
                else:
                    for k in range(0, self.j+1):
                        M[k][k] = torch.exp(-0.5*1j*param[i]*self.S[k][k])
                U = torch.kron(U, M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device, dtype=torch.complex64))
        
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

        S = sigma[self.mtx_id](self.j, self.k, D, device=device)
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
    r"""
    Hadamard Gate for qudits.

    The Hadamard gate creates a superposition of states in a qudit system.

    **Details:**

    * **Matrix Representation:**
    
    .. math::
          
            H = \frac{1}{\sqrt{D}}
            \begin{pmatrix}
            1 & 1 & \cdots & 1 \\\\
            1 & \omega & \cdots & \omega^{D-1} \\\\
            1 & \omega^2 & \cdots & \omega^{2(D-1)} \\\\
            \vdots & \vdots & \ddots & \vdots \\\\
            1 & \omega^{D-1} & \cdots & \omega^{(D-1)(D-1)}
            \end{pmatrix}

    Args:
        D (int): The dimension of the qudit. Default is 2.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        inverse (bool): Whether to apply the inverse of the Hadamard gate. Default is False.
        device (str): The device to perform the computations on. Default is 'cpu'.

    Attributes:
        index (list of int): The indices of the qudits to which the gate is applied.
        device (str): The device to perform the computations on.
        D (int): The dimension of the qudit.
        M (torch.Tensor): The matrix representation of the Hadamard gate.

    Examples:
        >>> import torch
        >>> import quforge as qf
        >>> gate = qf.HGate(D=2, index=[0])
        >>> state = torch.tensor([1, 0], dtype=torch.complex64)
        >>> result = gate(state)
        >>> print(result)
    """

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
            M = torch.conj(M).T.contiguous()
        self.register_buffer('M', M)

    def forward(self, x):
        """
        Apply the Hadamard gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.

        Returns:
            torch.Tensor: The resulting state after applying the Hadamard gate.
        """
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device, dtype=torch.complex64))
        return torch.matmul(U, x)


class HGate2(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, index=0, inverse=False, dim=3, N=2, device='cpu'):
        super(HGate2, self).__init__()
      
        pi = np.pi
        omega = np.exp(2*1j*pi/dim)
        M = torch.ones((dim, dim), dtype=torch.complex64).to(device)
        for i in range(1, dim):
            for j in range(1, dim):
                M[i, j] = omega**(j*i)
        M = M/(dim**0.5)

        esq,dir = nQudit(N,index)

        I_esq = torch.eye(dim**esq,device=device)
        I_dir = torch.eye(dim**dir,device=device)

        U = torch.kron(M,I_dir)
        U = torch.kron(I_esq,U)

        self.register_buffer('U', U)

    def forward(self, x):
    
        return torch.matmul(self.U, x)


class XGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, D=2, index=[0], device='cpu', inverse=False):
        super(XGate, self).__init__()

        self.index = index
        self.D = D
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = torch.matmul(base(D)[j].T, base(D)[(i+s) % D])
        if inverse:
            M = torch.conj(M.T)
        self.register_buffer('M', M)
            

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, dtype=torch.complex64, device=x.device))
        return torch.matmul(U, x)


class ZGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, D=2, s=1, index=[0], device='cpu', inverse=False):
        super(ZGate, self).__init__()

        omega = np.exp(2*1j*pi/D)

        self.index = index
        self.D = D
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = (omega**(j*s))*delta(i,j)
        if inverse:
            M = torch.conj(M.T)
        self.register_buffer('M', M)
        
    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device,  dtype=torch.complex64))
        
        return torch.matmul(U, x)

    def gate(self):
        return self.M


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
            if i in self.index:
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
    def __init__(self, control_1=0, control_2=1, target=2, D=2, N=3, inverse=False, device='cpu'):
        super(CCNOT, self).__init__()        
        L = torch.tensor(list(itertools.product(range(D), repeat=N))).to(device)
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


class CZGate(nn.Module):
    '''
    Controlled Z gate
    TO DO: optimize this gate
    '''
    def __init__(self, control=0, target=1, D=2, N=2, device='cpu'):
        super(CZGate, self).__init__()

        self.D = D
        self.control = control
        self.target = target

        U = 0.0
        for d in range(D):
            u = torch.eye(1, device=device, dtype=torch.complex64)
            for i in range(N):
                if i == self.control:
                    ZGate(D=D, device=device).gate()
                    u = torch.kron(u, base(self.D)[d] @ base(self.D)[d].T)
                elif i == self.target:
                    M = ZGate(D=D, device=device, s=d).gate()
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.D, device=device, dtype=torch.complex64))
            U += u
        print(U)
        self.register_buffer('U', U)
    
    def forward(self, x):
        return self.U @ x

