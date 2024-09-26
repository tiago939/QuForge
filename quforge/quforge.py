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

def State(dits, dim=2, device='cpu'):
    base = torch.zeros((dim, dim, 1), device=device)
    for i in range(dim):
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


def partial_trace(state, index, dim):
    #index: list of qudits to take the partial trace over
    rho = density_matrix(state)
    N = round(log(state.shape[0], dim))
    L = list(itertools.product(range(dim), repeat=N-len(index)))
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
                u = torch.kron(u, torch.eye(dim, dtype=torch.complex64, device=state.device))
            else:
                u = torch.kron(u, State(dits=str(i), dim=dim).to(state.device))
        U += torch.matmul(u.T, torch.matmul(rho, u))
    
    return U


def projector(index, dim):
    P = torch.zeros((dim, dim), dtype=torch.complex64)
    P[index][index] = 1.0

    return P


def measure(state=None, index=[0], shots=1, dim=2):
    #input:
        #state: state to measure
        #index: list of qudits to measure
        #shots: number of measurements
    #output:
        #histogram: histogram of the measurements
        #p: distribution probability
    rho = partial_trace(state, index, dim)
    p = abs(torch.diag(rho))
    p = p/torch.sum(p)

    a = np.array(range(len(rho)))
    positions = np.random.choice(a, p=p.detach().cpu().numpy(), size=shots)

    L = list(itertools.product(range(dim), repeat=len(index)))
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
        

def project(state, index=[0], dim=2):
    p = [(abs(state[i])**2).item() for i in range(len(state))]
    p = p/np.sum(p)

    a = np.array(range(len(state)))
    position = np.random.choice(a, p=p, size=1)[0]

    L = list(itertools.product(range(dim), repeat=int(log(state.shape[0], D))))[position]
    U = torch.eye(1, device=state.device)
    counter = 0
    size = int(log(state.shape[0], dim))
    for i in range(size):
        if i not in index:
            U = torch.kron(U, torch.eye(dim, device=state.device))
        else:
            U = torch.kron(U, projector(L[i], dim).to(state.device))
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

def eye(dim, device='cpu', sparse=False):
    '''
    Create a sparse identity matrix
    Input:
        -D: qudit dimension
        -device: cpu or cuda
    Output:
        -eye_sparse: sparse identity matrix
    '''
    if sparse is True:
        indices = torch.arange(dim, device=device).repeat(2, 1)
        values = torch.ones(dim, dtype=torch.complex64, device=device)
        M = torch.sparse_coo_tensor(indices, values, (dim, dim))
    else:
        M = torch.eye(dim, dtype=torch.complex64, device=device)

    return M


def kron(matrix1, matrix2, sparse=False):
    '''
    Tensor product of dense or sparse matrix
    Input:
        matrix1: first matrix
        matrix2: second matrix
    Output:
        matrix: matrix tensor product
    '''

    if sparse is True:
        D1 = matrix1.shape[0]
        D2 = matrix2.shape[0]

        # Coalesce the sparse matrices
        sparse1_coalesced = matrix1.coalesce()
        sparse2_coalesced = matrix2.coalesce()

        # Extract the values and the indexes
        values1 = sparse1_coalesced.values()
        index1 = sparse1_coalesced.indices()

        values2 = sparse2_coalesced.values()
        index2 = sparse2_coalesced.indices()

        # Expand the indexes for tensor product
        expanded_index1 = index1.unsqueeze(2)
        expanded_index2 = index2.unsqueeze(2).permute(0, 2, 1)

        # Evaluate the tensor products
        pos = (expanded_index1 * D2 + expanded_index2).view(2, -1)
        val = (values1.unsqueeze(1) * values2.unsqueeze(0)).view(-1)

        # Sparse matrix 
        matrix = torch.sparse_coo_tensor(pos, val, size=(D1 * D2, D1 * D2)).to(matrix1.device)

    elif sparse is False:
        matrix = torch.kron(matrix1, matrix2)

    return matrix

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

def base(D, device='cpu'):
    base = torch.eye(D, device=device).reshape((D,D,1))
    return base

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
    def __init__(self, dim=2, wires=1, device='cpu', sparse=False):
        super(Circuit, self).__init__()

        self.dim = dim 
        self.wires = wires
        self.device = device
        self.circuit = nn.Sequential()
        self.sparse = sparse

    def add(self, module, **kwargs):
        gate = module(D=self.dim, device=self.device, **kwargs)
        self.circuit.add_module(str(len(self.circuit)), gate)

    def add_gate(self, gate, **kwargs):
        self.circuit.add_module(str(len(self.circuit)), gate)

    def H(self, **kwargs):
        self.add_gate(H(dim=self.dim, device=self.device, **kwargs))

    def RX(self, **kwargs):
        self.add_gate(RX(dim=self.dim, device=self.device, sparse=self.sparse, **kwargs))

    def RY(self, **kwargs):
        self.add_gate(RY(dim=self.dim, device=self.device, sparse=self.sparse, **kwargs))

    def RZ(self, **kwargs):
        self.add_gate(RZ(dim=self.dim, device=self.device, sparse=self.sparse, **kwargs))

    def X(self, **kwargs):
        self.add_gate(X(dim=self.dim, device=self.device, **kwargs))

    def Y(self, **kwargs):
        self.add_gate(Y(dim=self.dim, device=self.device, **kwargs))

    def Z(self, **kwargs):
        self.add_gate(Z(dim=self.dim, device=self.device, **kwargs))

    def CNOT(self, **kwargs):
        self.add_gate(CNOT(dim=self.dim, wires=self.wires, sparse=self.sparse, device=self.device, **kwargs))

    def SWAP(self, **kwargs):
        self.add_gate(SWAP(dim=self.dim, device=self.device, **kwargs))

    def CZ(self, **kwargs):
        self.add_gate(CZ(dim=self.dim, wries=self.wires, device=self.device, **kwargs))

    def CCNOT(self, **kwargs):
        self.add_gate(CCNOT(dim=self.dim, device=self.device, **kwargs))

    def MCX(self, **kwargs):
        self.add_gate(MCX(dim=self.dim, wries=self.wires, device=self.device, **kwargs))

    def CRX(self, **kwargs):
        self.add_gate(CRX(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def CRY(self, **kwargs):
        self.add_gate(CRY(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def CRZ(self, **kwargs):
        self.add_gate(CRZ(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def Custom(self, **kwargs):
        self.add_gate(CustomGate(dim=self.dim, device=self.device, **kwargs))

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


class RX(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate

    def __init__(self, j=0, k=1, index=[0], dim=2, device='cpu', angle=False, sparse=False):
        super(RX, self).__init__()

        self.dim = dim
        self.index = index
        self.j = j 
        self.k = k
        self.sparse = sparse

        if angle is False:
            self.angle = nn.Parameter(torch.randn(len(index), device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(len(index), device=device))

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.dim))
        U = eye(1, device=x.device, sparse=self.sparse)

        for i in range(L):
            if i in self.index:
                indices = torch.tensor([[self.j, self.k, self.j, self.k], [self.j, self.k, self.k, self.j]], device=x.device)
                values = torch.zeros(4, dtype=torch.complex64, device=x.device)
                if param is False:
                    values[0] = torch.cos(self.angle[i]/2)
                    values[1] = torch.cos(self.angle[i]/2)
                    values[2] = -1j*torch.sin(self.angle[i]/2)
                    values[3] = -1j*torch.sin(self.angle[i]/2)
                else:
                    values[0] = torch.cos(param[i]/2)
                    values[1] = torch.cos(param[i]/2)
                    values[2] = -1j*torch.sin(param[i]/2)
                    values[3] = -1j*torch.sin(param[i]/2)
                
                if self.sparse is False:
                    M = torch.zeros((self.dim, self.dim), device=x.device, dtype=torch.complex64)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = kron(U, M, sparse=self.sparse)
            else:
                U = kron(U, eye(self.dim, device=x.device, sparse=self.sparse), sparse=self.sparse)

        return U @ x


class RY(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate

    def __init__(self, j=0, k=1, index=[0], dim=2, device='cpu', angle=False, sparse=False):
        super(RY, self).__init__()

        self.dim = dim
        self.index = index
        self.sparse = sparse
        self.j = j 
        self.k = k

        if angle is False:
            self.angle = nn.Parameter(torch.randn(len(index), device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(len(index), device=device))

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.dim))
        U = eye(1, device=x.device, sparse=self.sparse)

        for i in range(L):
            if i in self.index:
                indices = torch.tensor([[self.j, self.k, self.j, self.k], [self.j, self.k, self.k, self.j]], device=x.device)
                values = torch.zeros(4, dtype=torch.complex64, device=x.device)
                if param is False:
                    values[0] = torch.cos(self.angle[i]/2)
                    values[1] = torch.cos(self.angle[i]/2)
                    values[2] = -torch.sin(self.angle[i]/2)
                    values[3] = -torch.sin(self.angle[i]/2)
                else:
                    values[0] = torch.cos(param[i]/2)
                    values[1] = torch.cos(param[i]/2)
                    values[2] = -torch.sin(self.angle[i]/2)
                    values[3] = -torch.sin(self.angle[i]/2)
                
                if self.sparse is False:
                    M = torch.zeros((self.dim, self.dim), device=x.device, dtype=torch.complex64)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = kron(U, M, sparse=self.sparse)
            else:
                U = kron(U, eye(self.dim, device=x.device, sparse=self.sparse), sparse=self.sparse)

        return U @ x


class RZ(nn.Module):
    #j,k: indexes of the generalized Pauli matrices
    #index: index of the qudit to apply the gate
    #cutoff: cutoff of the infinite sum
    def __init__(self, j=1, index=[0], dim=2, device='cpu', angle=False, sparse=False):
        super(RZ, self).__init__()

        self.dim = dim
        self.index = index
        self.j = j
        self.sparse = sparse

        if angle is False:
            self.angle = nn.Parameter(torch.randn(len(index), device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(len(index), device=device))

    def forward(self, x, param=False):
        L = round(log(x.shape[0], self.dim))
        U = eye(1, device=x.device, sparse=self.sparse)
        for i in range(L):
            if i in self.index:
                if param is False:
                    indices = torch.tensor([range(self.j + 1), range(self.j + 1)], device=x.device)
                    angle = (self.angle[i] / 2)*np.sqrt(2 / (self.j * (self.j + 1)))
                    values = angle*torch.ones(self.j + 1, dtype=torch.complex64, device=x.device)
                    values[self.j] = values[self.j]*(-self.j)
                    values = torch.cos(values) - 1j*torch.sin(values)

                else:
                    indices = torch.tensor([range(self.j + 1), range(self.j + 1)], device=x.device)
                    angle = (param[i] / 2)*np.sqrt(2 / (self.j * (self.j + 1)))
                    values = angle*torch.ones(self.j + 1, dtype=torch.complex64, device=x.device)
                    values[self.j] = values[self.j]*(-self.j)
                    values = torch.cos(values) - 1j*torch.sin(values)
                
                M = eye(self.dim, sparse=self.sparse, device=x.device)
                if self.sparse is False:
                    M.index_put_(tuple(indices), values)
                else:
                    M = M + torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = kron(U, M, sparse=self.sparse)
            else:
                U = kron(U, eye(self.dim, sparse=self.sparse))
        
        return U @ x


class H(nn.Module):
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

    def __init__(self, dim=2, index=[0], inverse=False, device='cpu'):
        super(H, self).__init__()

        self.index = index
        self.device = device
        self.dim = dim
        omega = np.exp(2*1j*pi/dim)

        M = torch.ones((dim, dim), dtype=torch.complex64, device=device)
        for i in range(1, dim):
            for j in range(1, dim):
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
        L = round(log(x.shape[0], self.dim))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, device=x.device, dtype=torch.complex64))
        return torch.matmul(U, x)


class X(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, dim=2, index=[0], device='cpu', inverse=False):
        super(X, self).__init__()

        self.index = index
        self.dim = dim
        M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
        for i in range(dim):
            for j in range(dim):
                M[j][i] = torch.matmul(base(dim)[j].T, base(dim)[(i+s) % dim])
        if inverse:
            M = torch.conj(M.T)
        self.register_buffer('M', M)
            

    def forward(self, x):
        L = round(log(x.shape[0], self.dim))
        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, dtype=torch.complex64, device=x.device))
        return torch.matmul(U, x)


class Z(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, dim=2, s=1, wires=[0], device='cpu', inverse=False):
        super(Z, self).__init__()

        omega = np.exp(2*1j*pi/D)

        self.index = index
        self.dim = dim
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = (omega**(j*s))*delta(i,j)
        if inverse:
            M = torch.conj(M.T)
        self.register_buffer('M', M)
        
    def forward(self, x):
        L = round(log(x.shape[0], self.dim))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.D, device=x.device,  dtype=torch.complex64))
        
        return torch.matmul(U, x)

    def gate(self):
        return self.M


class Y(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, dim=2, index=[0], device='cpu'):
        super(Y, self).__init__()

        self.index = index
        self.dim = dim
        X = XGate(s=s, device=device).M
        Z = ZGate(device=device).M
        M = torch.matmul(Z, X)/1j
        self.register_buffer('M', M) 
        
    def forward(self, x):
        L = round(log(x.shape[0], self.dim))
        U = torch.eye(1, device=x.device, dtype=torch.complex64)
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, device=x.device, dtype=torch.complex64))
        
        return torch.matmul(U, x)


class Xd(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, dim=2, index=0, device='cpu'):
        super(Xd, self).__init__()

        self.dim = dim
        self.index = index
        M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
        for i in range(dim):
            for j in range(dim):
                M[j][i] = torch.matmul(base(dim, device=device)[j].T, base(dim, device=device)[(dim-i) % dim])
        self.register_buffer('M', M)   
        
    def forward(self, x):
        L = round(log(x.shape[0], self.dim))
        U = torch.eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, device=x.device))
        return torch.matmul(U, x)


class Identity(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, dim=2, wires=1, device='cpu'):
        super(Identity, self).__init__()

        self.U = torch.eye(dim**wires, dytpe=torch.complex64, device=device)
        
    def forward(self, x):
        return torch.matmul(U, x)


def cnot_qudits_Position(c, t, n, d, device='cpu'):
    values = torch.arange(d,dtype=torch.float).to(device)
    L = torch.stack(torch.meshgrid(*([values] * n)), dim=-1).to(device).reshape(-1, n)
    L[:,t]=(L[:,t]+L[:,c])%d
    tt = d**torch.arange(n-1, -1, -1, dtype=torch.float).to(device).reshape(n,1)
    lin = torch.matmul(L,tt).to(device)
    col = torch.arange(d**n,dtype=torch.float).to(device).reshape(d**n,1)
    return  torch.cat((lin, col), dim=1).to(device)


def CNOT_sparse(c, t, d, n, device='cpu'):
    # CNOT sparse matrix
    D = d**n
    indices = cnot_qudits_Position(c,t,n,d,device=device)
    values = torch.ones(D).to(device)
    eye_sparse = torch.sparse_coo_tensor(indices.t(), values, (D, D),dtype=torch.complex64).to(device)

    return eye_sparse


class CNOT(nn.Module):
    #index = [c, t] : c=control, t=target
    #wires: total number of qudits in the circuit
    def __init__(self, index=[0,1], wires=2, dim=2, device='cpu', sparse=False, inverse=False):
        super(CNOT, self).__init__()

        if sparse is False:   
            L = torch.tensor(list(itertools.product(range(dim), repeat=wires)))
            l2ns = L.clone()
            l2ns[:, index[1]] = (l2ns[:, index[0]] + l2ns[:, index[1]]) % dim
            indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
            U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64)).to(device)
        else:
            U = CNOT_sparse(index[0], index[1], dim, wires, device=device)

        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)    
        
    def forward(self, x):
        return self.U @ x


class SWAP(nn.Module):
        #swap the state of two qudits
    def __init__(self, index=[0,1], dim=2, wires=2, device='cpu'):
        super(SWAP, self).__init__()

        c = index[0]
        t = index[1]
        D = dim**wires 
        U = torch.zeros((D, D), device=device, dtype=torch.complex64)
        for k in range (D):
            localr = dec2den(k, wires, dim)
            locall = localr.copy()
            locall[c] = localr[t]
            locall[t] = localr[c]
            globall = den2dec(locall, dim)
            U[globall, k] = 1

        self.register_buffer('U', U) 

    def forward(self, x):
        return self.U @ x


class CCNOT(nn.Module):
    #CCNOT gate, also know as Toffoli gate
    #index = [c1,c2,t] : c1=control 1, c2=control2, t=target
    #wires: number of qudits
    def __init__(self, index=[0,1,2], dim=2, wires=3, inverse=False, device='cpu'):
        super(CCNOT, self).__init__()        
        L = torch.tensor(list(itertools.product(range(dim), repeat=wires))).to(device)
        l2ns = L.clone()
        l2ns[:, index[2]] = (l2ns[:, index[0]]*l2ns[:, index[1]] + l2ns[:, index[2]]) % dim
        indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64))        
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)
        
    def forward(self, x):
        return torch.matmul(self.U, x)


class MCX(nn.Module):
    #multi-controlled cx gate
    #wires: number of qudits
    def __init__(self, index=[0, 1], dim=2, wires=2, inverse=False):
        super(MCX, self).__init__()        
        L = torch.tensor(list(itertools.product(range(dim), repeat=wires)))
        l2ns = L.clone()
        control_value = 1
        for i in range(len(index)-1):
            control_value *= l2ns[:, index[i]]
        l2ns[:, index[-1]] = (control_value + l2ns[:, index[-1]]) % dim
        indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64))        
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)
        
    def forward(self, x):
        return torch.matmul(self.U, x)


class CZ(nn.Module):
    '''
    Controlled Z gate
    index = [c, t] : c=control, t=target
    '''
    def __init__(self, index=[0,1], dim=2, wires=2, device='cpu'):
        super(CZ, self).__init__()

        self.dim = dim
        self.index = index
        self.wires = wires

        U = 0.0
        for d in range(dim):
            u = torch.eye(1, device=device, dtype=torch.complex64)
            for i in range(wires):
                if i == index[0]:
                    u = torch.kron(u, base(dim)[d] @ base(dim)[d].T)
                elif i == index[1]:
                    M = Z(dim=dim, device=device, s=d).gate()
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(dim, device=device, dtype=torch.complex64))
            U += u

        self.register_buffer('U', U)
    
    def forward(self, x):
        return self.U @ x


class CRX(nn.Module):
    '''
    Controlled RX gate
    index = [c, t] : c=control, t=target
    N = total number of qudits in the circuit
    '''
    def __init__(self, index=[0,1], dim=2, N=2, j=0, k=1, device='cpu', sparse=False):
        super(CRX, self).__init__()

        self.index = index
        self.dim = dim
        self.j = j
        self.k = k
        self.angle = nn.Parameter(pi*torch.randn(1, device=device))
        self.N = N
        self.sparse = sparse
    
    def forward(self, x):
        c = self.index[0]
        t = self.index[1]
        
        D = self.dim**self.N
        U = torch.zeros((D, D), dtype=torch.complex64, device=x.device)
        Dl = D // self.dim
        indices_list = []
        values_list = []

        for m in range (0, Dl):
            local = dec2den(m, self.N-1, self.dim)
            if self.N == 2:
                angle = (local[0]*self.angle)/2
            else:
                angle = (local[c]*self.angle)/2

            listj = local.copy()
            listj.insert(t, self.j-1)
            intj = den2dec(listj, self.dim)
            listk = local.copy()
            listk.insert(t, self.k-1)
            intk = den2dec(listk, self.dim)

            indices = torch.tensor([[intj, intk, intj, intk], [intj, intk, intk, intj]])

            values = torch.zeros(4, dtype=torch.complex64)
            values[0] = torch.cos(angle)
            values[1] = torch.cos(angle)
            values[2] = -1j*torch.sin(angle)
            values[3] = -1j*torch.sin(angle)

            for l in range(0, self.dim):
                if l != self.j-1 and l != self.k-1:
                    listl = local.copy()
                    listl.insert(t,l)
                    intl = den2dec(listl, self.dim) 
                    new_index = torch.tensor([[intl]])
                    new_value = torch.tensor([1.0])
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat((values, new_value))

            indices_list.append(indices)
            values_list.append(values)

        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if self.sparse is False:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=x.device)

        return U @ x


class CRY(nn.Module):
    '''
    Controlled RX gate
    index = [c, t] : c=control, t=target
    N = total number of qudits in the circuit
    '''
    def __init__(self, index=[0,1], dim=2, N=2, j=0, k=1, device='cpu', sparse=False):
        super(CRY, self).__init__()

        self.index = index
        self.dim = dim
        self.j = j
        self.k = k
        self.angle = nn.Parameter(pi*torch.randn(1, device=device))
        self.N = N
        self.sparse = sparse
    
    def forward(self, x):
        c = self.index[0]
        t = self.index[1]
        
        D = self.dim**self.N
        Dl = D // self.dim
        indices_list = []
        values_list = []

        for m in range (0, Dl):
            local = dec2den(m, self.N-1, self.dim)
            if self.N == 2:
                angle = (local[0]*self.angle)/2
            else:
                angle = (local[c]*self.angle)/2

            listj = local.copy()
            listj.insert(t, self.j-1)
            intj = den2dec(listj, self.dim)
            listk = local.copy()
            listk.insert(t, self.k-1)
            intk = den2dec(listk, self.dim)

            indices = torch.tensor([[intj, intk, intj, intk], [intj, intk, intk, intj]])

            values = torch.zeros(4, dtype=torch.complex64)
            values[0] = torch.cos(angle)
            values[1] = torch.cos(angle)
            values[2] = -torch.sin(angle)
            values[3] = -torch.sin(angle)

            for l in range(0, self.dim):
                if l != self.j-1 and l != self.k-1:
                    listl = local.copy()
                    listl.insert(t,l)
                    intl = den2dec(listl, self.dim) 
                    new_index = torch.tensor([[intl]])
                    new_value = torch.tensor([1.0])
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat((values, new_value))

            indices_list.append(indices)
            values_list.append(values)

        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if self.sparse is False:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=x.device)

        return U @ x


class CRZ(nn.Module):
    '''
    Controlled RX gate
    index = [c, t] : c=control, t=target
    N = total number of qudits in the circuit
    '''
    def __init__(self, index=[0, 1], dim=2, N=2, j=1, device='cpu', sparse=False):
        super(CRZ, self).__init__()

        self.index = index
        self.dim = dim
        self.j = j
        self.angle = nn.Parameter(pi*torch.randn(1, device=device))
        self.N = N
        self.sparse = sparse
    
    def forward(self, x):
        c = self.index[0]
        t = self.index[1]
        
        D = self.dim**self.N
        Dl = D // self.dim
        indices_list = []
        values_list = []

        indices = []
        values = []
        for m in range (0, Dl):
            local = dec2den(m, self.N-1, self.dim)
            if self.N == 2:
                loc = local[0]
            else:
                loc = local[c]
            angle = ((loc*self.angle)/2)*np.sqrt(2/(self.j*(self.j+1)))

            for k in range(0, self.dim):
                listk = local.copy()
                listk.insert(t,k) # insert k in position t of the list
                intk = den2dec(listk, self.dim) # integer for the k state
                if k < self.j:
                    indices.append([intk, intk])
                    values.append(torch.cos(angle) - 1j*torch.sin(angle))
                elif k == self.j:
                    angle = self.j * angle
                    indices.append([intk, intk])
                    values.append(torch.cos(angle) + 1j*torch.sin(angle))
                elif k > self.j:
                    indices.append([intk, intk])
                    values.append(1.0)

        indices = torch.tensor(indices)
        indices = indices.T
        values = torch.tensor(values)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if self.sparse is False:
            U = torch.zeros((D, D), device=x.device, dtype=torch.complex64)
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=x.device)

        return U @ x

