import torch
import torch.nn as nn
import numpy as np 
from math import log as log
import itertools

pi = np.pi

def Sx(j, k, base):

    return torch.kron(base[j], base[k].T) + torch.kron(base[k], base[j].T) + 0*1j

def Sy(j, k, base):

    return -1j*torch.kron(base[j], base[k].T) + 1j*torch.kron(base[k], base[j].T)

def Sz(j, k, base):

    return torch.kron(base[j], base[j].T) - torch.kron(base[k], base[k].T) + 0*1j

sigma = [Sx, Sy, Sz]


def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def eye(D, device='cpu'):
    '''
    Create a sparse identity matrix
    Input:
        -D: qudit dimension
        -device: cpu or cuda
    Output:
        -eye_sparse: sparse identity matrix
    '''
    indices = torch.arange(D, device=device).repeat(2, 1)
    values = torch.ones(D, dtype=torch.complex64, device=device)
    eye_sparse = torch.sparse_coo_tensor(indices, values, (D, D))

    return eye_sparse


def tosparse(u):
    '''
    Create a sparse matrix from a dense matrix
    Input:
        -u: dense matrix 
    Output:
        -U_sparse: sparse matrix
    '''
    nonzero_coords = torch.nonzero(u)
    values = u[nonzero_coords[:, 0], nonzero_coords[:, 1]]

    U_sparse = torch.sparse_coo_tensor(nonzero_coords.t(), values)

    return U_sparse


def nQudit(n, index):
    left = index
    right = n-index-1

    return left, right


def kron(matrix1, matrix2, D1, D2, device='cpu'):
    '''
    Sparse tensor product 
    Input:
        matrix1: first matrix
        matrix2: second matrix
        D1: dimension of the first matrix
        D2: dmension of the second matrix
        device: cpu or cuda
    Output:
        matrix: sparse matrix tensor product
    '''

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

    return matrix


def base(D, device='cpu'):
    base = torch.eye(D, device=device).reshape((D, D, 1))
    return base


def adjust_dimension(tensor):
    # check the dimensions of the tensor
    if tensor.dim() == 1:
        # If the tensor has dimension n, then add an extra dimension
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 2:
        # If the tensor does not have dimension m x n or n, throw an error message
        raise ValueError("The tensor must have dimension either m x n or n")

    return tensor


def remove_elements(N, list, device='cpu'):
    tensor = torch.arange(N).to(device)

    list_tensor = torch.tensor(list, device=device)

    # Creates a Boolean mask indicating the elements to be removed
    mask = ~torch.isin(tensor, list_tensor)

    # Apply the mask to keep only elements not present in the list
    tensor_result = tensor[mask]

    return tensor_result


def NewState(state, index, dim,N,device='cpu'):
    indices_complementares = remove_elements(N,index,device=device)
    # Concatena os indices da lista_indices com os indices complementares
    index.sort()
  
    lista_indices = torch.tensor(index).to(device)
    nova_ordem_indices = torch.cat((lista_indices, indices_complementares)).to(device)
  
    # Permuta os qudits de acordo com a nova ordem de indices
    novo_estado_quantico = state.view((dim,) * N)
    novo_estado_quantico = novo_estado_quantico.permute(*nova_ordem_indices.tolist())
    novo_estado_quantico = novo_estado_quantico.reshape((dim**N,))

    return novo_estado_quantico.reshape(novo_estado_quantico.shape[0],1)


def pTraceB(state, da, db, device='cpu'):
    state_reshaped = state.view(da, db)
    state_conj = torch.conj(state_reshaped).to(device)
    rhoA = torch.matmul(state_reshaped, state_conj.transpose(0, 1)).to(device)
    del state_reshaped, state_conj
    return rhoA


class density_matrix(nn.Module):
    
    def __init__(self, index, D=2, N=1, device='cpu'):
        super(density_matrix, self).__init__()

        self.index = index
        self.D = D
        self.n = N
        self.device = device

        self.da = int( D**len(index) )
        self.db = int( D**( N-len(index) ) )
        
    def forward(self, x):
        state1 = NewState(x, self.index, self.D, self.n,device=self.device)
        rhoA = pTraceB(state1, self.da, self.db,device=self.device)
        del state1
        return rhoA


class prob(nn.Module):
    
    def __init__(self, index, dim=3,N=2,device='cpu'):
        super(prob, self).__init__()

        self.index = index
        self.dim = dim
        self.n = N
        self.device = device
        self.dm = density_matrix(index,dim=dim,N=N,device=device)

    def forward(self, state):
        probs = []
        for i in range(state.shape[1]):
            state_ = state[:,i]
            state_ = state_.view(state_.shape[0],-1)
            rhoA = self.dm(state_) 
            p = abs(torch.diag(rhoA))
            p = p/torch.sum(p)
            probs.append(p)
            
        return torch.stack(probs)


class AmplitudeEncoder(nn.Module):
    def __init__(self, n, d,device='cpu'):
        super(AmplitudeEncoder, self).__init__()
        self.n = n
        self.d = d
        self.device = device


    def forward(self, x):
        x = adjust_dimension(x)

        x1 = torch.sum(x,dim=1, keepdim=True).to(self.device)

        x = x/x1

        zero = torch.zeros(x.shape[0], (self.d**self.n)-x.shape[1] ).to(self.device)
        y = torch.cat([x,zero],dim=1).to(self.device)

        return y.to(torch.complex64).T


def cnot_qudits_Position(c,t,n,d,device='cpu'):
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
    #control: control qudit
    #target: target qudit
    #N: number of qudits
    def __init__(self, control=0, target=1, D=2, N=2, device='cpu'):
        super(CNOT, self).__init__()
        U = CNOT_sparse(control, target, D, N, device=device)

        self.register_buffer('U', U)

    def forward(self, x):
        return self.U @ x


class XGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, index=[0], D=2, N=1, device='cpu'):
        super(XGate, self).__init__()

        self.index = index

        self.index = index
        self.D = D
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = torch.matmul(base(D)[j].T, base(D)[(i+s) % D])
        M = M.to_sparse()
        self.register_buffer('M', M)   

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = kron(U, self.M, D1=U.shape[0], D2=self.M.shape[0])
            else:
                U = kron(U, eye(self.D, device=x.device), D1=U.shape[0], D2=self.D)

        return U @ x


class ZGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, index=[0], D=2, N=1, device='cpu'):
        super(ZGate, self).__init__()

        self.device = device
        self.index = index

        omega = np.exp(2*1j*pi/D)

        self.index = index
        self.D = D
        M = torch.zeros((D, D), dtype=torch.complex64, device=device)
        for i in range(D):
            for j in range(D):
                M[j][i] = (omega**(j*s))*delta(i,j)
        M = M.to_sparse()
        self.register_buffer('M', M)   

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = kron(U, self.M, D1=U.shape[0], D2=self.M.shape[0])
            else:
                U = kron(U, eye(self.D, device=x.device), D1=U.shape[0], D2=self.D)

        return U @ x


class YGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, index=[0], D=2, N=1, device='cpu'):
        super(YGate, self).__init__()

        self.index = index
        self.D = D
        X = XGate(s=s, device=device).M
        Z = ZGate(device=device).M
        M = torch.matmul(Z, X)/1j
        self.register_buffer('M', M) 

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = eye(1, device=x.device)
        for i in range(L):
            if i == self.index:
                U = kron(U, self.M, D1=U.shape[0], D2=self.M.shape[0])
            else:
                U = kron(U, eye(self.D, device=x.device), D1=U.shape[0], D2=self.D)

        return U @ x


class RGate(nn.Module):
    #TO DO: matrix exponential for sparse matrices
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices (k must be greater than j)
    #index: index of the qudit to apply the gate
    def __init__(self, mtx_id=0, j=0, k=1, index=[0], D=2, N=1, device='cpu'):
        super(RGate, self).__init__()

        self.index = index
        self.D = D
        self.device = device

        S = sigma[mtx_id](j, k, base(D, device=device)).to(device)
        self.angle = nn.Parameter(torch.rand(len(index), device=device))
        self.register_buffer('S', S)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = eye(1, device=x.device)
        for i in range(L):
            if i in self.index:
                M = torch.matrix_exp(-0.5*1j*self.angle[i]*self.S)
                M = M.to_sparse()
                U = kron(U, M, D1=U.shape[0], D2=M.shape[0])
            else:
                U = kron(U, eye(D, device=self.device), D1=U.shape[0], D2=D)

        return U @ x


class HGate(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, index=[0], D=2, N=1, device='cpu', inverse=False):
        super(HGate, self).__init__()

        self.D = D
        self.index = index
        self.device = device

        pi = np.pi
        omega = np.exp(2*1j*pi/D)
        M = torch.ones((D, D), dtype=torch.complex64).to(device)
        for i in range(1, D):
            for j in range(1, D):
                M[i, j] = omega**(j*i)
        M = M/(D**0.5)
        if inverse:
            M = torch.conj(self.M).T.contiguous()
        M = tosparse(M)
        self.register_buffer('M', M)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))
        U = eye(1, device=self.device)
        for i in range(L):
            if i in self.index:
                U = kron(U, self.M, U.shape[0], self.M.shape[0])
            else:
                U = kron(U, eye(self.D, device=self.device), U.shape[0], self.D)
        return U @ x