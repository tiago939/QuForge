import numpy as np
import torch
import torch.nn as nn

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


def fidelity(state1, state2):
    F = abs(torch.matmul(torch.conj(state1).T, state2))**2
    return F.real

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


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


def argmax(x):
    return torch.argmax(x)


def mean(x):
    return torch.mean(x)


def dec2den(j,N,d):
    # convert from decimal to denary representation
    den = [0 for k in range(0,N)]
    jv = j
    for k in range(0,N):
        if jv >= d**(N-1-k):
            den[k] = jv//(d**(N-1-k))
            jv = jv - den[k]*d**(N-1-k)
    return den

def projector(index, dim):
    P = torch.zeros((dim, dim), dtype=torch.complex64)
    P[index][index] = 1.0

    return P

def den2dec(local,d):
    # convert from denary to decimal representation
    # local = list with the local computational base state values 
    # d = individual qudit dimension
    N = len(local)
    j = 0
    for k in range(0,N):
        j += local[k]*d**(N-1-k)
    return j # value of the global computational basis index


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

def zeros(m,n, device='cpu'):
    M = torch.zeros((m, n), device=device)
    return M

def ones(m,n, device='cpu'):
    M = torch.ones((m, n), device=device)
    return M

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