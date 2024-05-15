import torch
import torch.nn as nn
import numpy as np
from math import log as log
import itertools
from torch.nn.parameter import Parameter
import math

def State(dits, dim, state=None, device='cpu'):
    D = dim
    base = Base(D)
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



def Sx(j, k, base):
    return torch.kron(base[j-1], base[k-1].T) + torch.kron(base[k-1], base[j-1].T) + 0*1j

def Sy(j, k, base):
    return -1j*torch.kron(base[j-1], base[k-1].T) + 1j*torch.kron(base[k-1], base[j-1].T)

def Sz(j,k, base):

    f = (2.0/(j*(j+1)))**0.5
    s = 0.0
    for k in range(0, j+1):
        s += ((-j)**delta(k,j))*torch.kron(base[k], base[k].T)
    return f*s + 0*1j

sigma = [Sx, Sy, Sz]


def Identity_sparse(D,device='cpu'):
    # Cria uma matriz esparsa representando a matriz identidade de dimensão D
    indices = torch.arange(D).repeat(2, 1).to(device)
    values = torch.ones(D).to(device)
    eye_sparse = torch.sparse_coo_tensor(indices, values, (D, D)).to(device)

    return eye_sparse


def sparse(u, device='cpu'):
    device = torch.device(device)
    u = u.to(device)
    nonzero_coords = torch.nonzero(u, as_tuple=False).t().to(device)
    values = u[tuple(nonzero_coords)].to(device)
    U_sparse = torch.sparse_coo_tensor(nonzero_coords, values, u.size(), device=device)
    return U_sparse




def nQudit(n,indice):
    esquerda = indice
    direita = n-indice-1
    return esquerda,direita



def sparse_kron(sparse1, sparse2, n1, n2,device='cpu'):
    '''
    Função otimizada para calcular o produto tensorial.
    Input:
        sparse1: matriz esparsa
        sparse2: matriz esparsa
        n1: dimensão da matriz esparsa sparse1
        n2: dimensão da matriz esparsa sparse2
    Output:
        Matriz esparsa resultante do produto tensorial
    '''

    # Coalesce as matrizes esparsas
    sparse1_coalesced = sparse1.coalesce()
    sparse2_coalesced = sparse2.coalesce()

    # Extraia valores e índices
    values1 = sparse1_coalesced.values()
    indices1 = sparse1_coalesced.indices()

    values2 = sparse2_coalesced.values()
    indices2 = sparse2_coalesced.indices()

    # Expandir os índices para realizar a multiplicação tensorial
    expanded_indices1 = indices1.unsqueeze(2)
    expanded_indices2 = indices2.unsqueeze(2).permute(0, 2, 1)

    # Calcular os produtos tensoriais
    pos = (expanded_indices1 * n2 + expanded_indices2).view(2, -1)
    val = (values1.unsqueeze(1) * values2.unsqueeze(0)).view(-1)

    # Criar a matriz esparsa resultante
    result = torch.sparse_coo_tensor(pos, val, size=(n1 * n2, n1 * n2)).to(device)

    return result



def Base(D, device='cpu'):
    base = torch.eye(D).unsqueeze(2).to(device)
    return base

def delta(x,y):
    if x == y:
        return 1.0
    else:
        return 0.0




def cnot_qudits_Position(c,t,n,d,device='cpu'):
    values = torch.arange(d,dtype=torch.float).to(device)
    L = torch.stack(torch.meshgrid(*([values] * n)), dim=-1).to(device).reshape(-1, n)


    L[:,t]=(L[:,t]+L[:,c])%d

    tt = d**torch.arange(n-1, -1, -1, dtype=torch.float).to(device).reshape(n,1)

    lin = torch.matmul(L,tt).to(device)
    col = torch.arange(d**n,dtype=torch.float).to(device).reshape(d**n,1)
    return  torch.cat((lin, col), dim=1).to(device)

def CNOT_sparse(c,t,d,n,device='cpu'):
    # Cria uma matriz esparsa representando a porta CNOT
    D = d**n
    indices = cnot_qudits_Position(c,t,n,d,device=device)

    values = torch.ones(D).to(device)
    eye_sparse = torch.sparse_coo_tensor(indices.t(), values, (D, D),dtype=torch.complex64).to(device)

    return eye_sparse

class CNOT(nn.Module):
    #control: control qudit
    #target: target qudit
    #N: number of qudits
    def __init__(self, wires = None, dim=3,N=2,device='cpu'):
        super(CNOT, self).__init__()
        D = dim
        U = CNOT_sparse(wires[0],wires[1],dim,N,device=device)

        self.register_buffer('U', U)

    def forward(self, x):
        return torch.matmul(self.U, x)



class X(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, wires=0, dim=3, N=2, device='cpu'):
        super(X, self).__init__()

        self.index = wires

        base = Base(dim)
        self.M = torch.zeros((dim, dim), dtype=torch.complex64).to(device)
        for i in range(dim):
            for j in range(dim):
                self.M[j][i] = torch.matmul(base[j].T, base[(i+s) % dim]).to(device)
        self.M = nn.Parameter(self.M).requires_grad_(False)

        self.M1 = sparse(self.M,device=device)
        del self.M

        esq,dir = nQudit(N,wires)

        I_esq = Identity_sparse(dim**esq,device=device)
        I_dir = Identity_sparse(dim**dir,device=device)

        U = sparse_kron(self.M1,I_dir,dim,dim**dir,device=device)
        U = sparse_kron(I_esq,U,dim**esq,dim*(dim**dir),device=device)

        del esq,dir,I_esq,I_dir

        self.U = U

    def forward(self, x):

        return torch.sparse.mm(self.U,x)



class Z(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, s=1, wires=0, dim=3, N=2, device='cpu' ):
        super(Z, self).__init__()

        self.device = device
        self.index = wires

        self.M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
        pi = np.pi
        omega = np.exp(2*1j*pi/dim)
        for i in range(dim):
            for j in range(dim):
                self.M[j][i] = (omega**(j*s))*delta(i,j)
        self.M = nn.Parameter(self.M).requires_grad_(False)

        self.M1 = sparse(self.M,device=device)
        del self.M

        esq,dir = nQudit(N,wires)

        I_esq = Identity_sparse(dim**esq,device=device)
        I_dir = Identity_sparse(dim**dir,device=device)

        U = sparse_kron(self.M1,I_dir,dim,dim**dir,device=device)
        U = sparse_kron(I_esq,U,dim**esq,dim*(dim**dir),device=device)

        del esq,dir,I_esq,I_dir

        self.U = U

    def forward(self, x):

        return torch.sparse.mm(self.U,x)#self.U@x



class Y(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self,  s=1, wires=0, dim=3, N=2, device='cpu' ):
        super(Y, self).__init__()


        X = X(s=s,N=N, dim=dim,device=device).M1
        Z = Z(dim=dim,N=N,device=device).M1

        M = torch.sparse.mm(Z, X)/1j
        self.M1 = M
        del M,X,Z

        esq,dir = nQudit(N,wires)

        I_esq = Identity_sparse(dim**esq,device=device)
        I_dir = Identity_sparse(dim**dir,device=device)

        U = sparse_kron(self.M1,I_dir,dim,dim**dir,device=device)
        U = sparse_kron(I_esq,U,dim**esq,dim*(dim**dir),device=device)

        del esq,dir,I_esq,I_dir

        self.U = U


    def forward(self, x):

        return torch.sparse.mm(self.U,x)#self.U@x




class R(nn.Module):
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, angle=False, mtx_id=0, j=1, k=2, wires=0,dim=3,N=2,device='cpu'):
        super(R, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = wires
        self.dim = dim

        self.base = Base(dim,device=device)

        self.esq,self.dir = nQudit(N,wires)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = Identity_sparse(self.dim**self.dir,device=self.device)
        self.angle = angle
        if angle is False:
            self.angle = Parameter(torch.FloatTensor(1).uniform_(0, 2*math.pi))

    def forward(self, x):


        M = torch.matrix_exp(-0.5*1j*self.angle*self.S).to(self.device)

        M = sparse(M,device=self.device)


        U = sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return torch.sparse.mm(U,x)#U@x


class RX(nn.Module):
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, angle=False, mtx_id=0, j=1, k=2, wires=0,dim=3,N=2,device='cpu'):
        super(RX, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = wires
        self.dim = dim

        self.base = Base(dim,device=device)

        self.esq,self.dir = nQudit(N,wires)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = Identity_sparse(self.dim**self.dir,device=self.device)
        self.angle = angle
        if angle is False:
            self.angle = Parameter(torch.FloatTensor(1).uniform_(0, 2*math.pi))

    def forward(self, x):


        M = torch.matrix_exp(-0.5*1j*self.angle*self.S).to(self.device)

        M = sparse(M,device=self.device)


        U = sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return torch.sparse.mm(U,x)#U@x



class RY(nn.Module):

    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, angle=False, mtx_id=1, j=1, k=2, wires=0,dim=3,N=2,device='cpu'):
        super(RY, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = wires
        self.dim = dim

        self.base = Base(dim,device=device)

        self.esq,self.dir = nQudit(N,wires)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = Identity_sparse(self.dim**self.dir,device=self.device)
        self.angle = angle
        if angle is False:
            self.angle = Parameter(torch.FloatTensor(1).uniform_(0, 2*math.pi))

    def forward(self, x):


        M = torch.matrix_exp(-0.5*1j*self.angle*self.S).to(self.device)

        M = sparse(M,device=self.device)


        U = sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return torch.sparse.mm(U,x)#U@x


class RZ(nn.Module):

    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, angle=False, mtx_id=2, j=1, k=2, wires=0,dim=3,N=2,device='cpu'):
        super(RZ, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = wires
        self.dim = dim

        self.base = Base(dim,device=device)

        self.esq,self.dir = nQudit(N,wires)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = Identity_sparse(self.dim**self.dir,device=self.device)
        self.angle = angle
        if angle is False:
            self.angle = Parameter(torch.FloatTensor(1).uniform_(0, 2*math.pi))

    def forward(self, x):


        M = torch.matrix_exp(-0.5*1j*self.angle*self.S).to(self.device)

        M = sparse(M,device=self.device)


        U = sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return torch.sparse.mm(U,x)#U@x


class H(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, wires=0, inverse=False, dim=3, N=2,device='cpu'):
        super(H, self).__init__()

        pi = np.pi
        omega = np.exp(2*1j*pi/dim)
        self.M = torch.ones((dim, dim), dtype=torch.complex64).to(device)
        for i in range(1, dim):
            for j in range(1, dim):
                self.M[i, j] = omega**(j*i)
        self.M = self.M/(dim**0.5)
        if inverse:
            self.M = torch.conj(self.M).T.contiguous()
        self.M = nn.Parameter(self.M).requires_grad_(False)

        self.M1 = sparse(self.M,device=device)
        del self.M

        esq,dir = nQudit(N,wires)

        I_esq = Identity_sparse(dim**esq,device=device)
        I_dir = Identity_sparse(dim**dir,device=device)

        U = sparse_kron(self.M1,I_dir,dim,dim**dir,device=device)
        U = sparse_kron(I_esq,U,dim**esq,dim*(dim**dir),device=device)

        del esq,dir,I_esq,I_dir

        self.U = U

    def forward(self, x):

        return torch.sparse.mm(self.U,x)#torch.sparse.mm(x.T, self.U).T




def remove_elements(N, my_list, device='cpu'):
    tensor = torch.arange(N).to(device)

    # Criar um tensor com os elementos da lista para verificar a ocorrência
    list_tensor = torch.tensor(my_list, device=device)

    # Criar um tensor com os índices dos elementos que devem ser removidos
    indices_to_remove = torch.zeros(N, dtype=torch.bool, device=device)
    for element in my_list:
        indices_to_remove[element] = True

    # Criar o tensor resultante removendo os elementos
    tensor_result = tensor[~indices_to_remove]

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


def pTraceB(state, da, db,device='cpu'):
    state_reshaped = state.view(da, db)
    state_conj = torch.conj(state_reshaped).to(device)
    rhoA = torch.matmul(state_reshaped, state_conj.transpose(0, 1)).to(device)
    del state_reshaped,state_conj
    return rhoA


class density_matrix(nn.Module):

    def __init__(self, wires = None, dim=3,N=2,device='cpu'):
        super(density_matrix, self).__init__()

        self.index = wires
        self.dim = dim
        self.n = N
        self.device = device

        self.da = int(dim**len(wires))
        self.db = int( dim**( N-len(wires) ) )


    def forward(self, x):
        state1 = NewState(x, self.index, self.dim, self.n,device=self.device)
        rhoA = pTraceB(state1, self.da, self.db,device=self.device)
        del state1
        return rhoA



class prob(nn.Module):

    def __init__(self, wires = None, dim=3,N=2,device='cpu'):
        super(prob, self).__init__()

        #self.index = wires
        self.dim = dim
        self.n = N
        self.device = device
        self.dm = density_matrix(wires,dim=dim,N=N,device=device)


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


def contar_elementos(lista):
    count = 0
    for elemento in lista:
        if isinstance(elemento, list):
            count += 1

    return count


class density_matrixT(nn.Module):

    def __init__(self, index=None, dim=3,N=2,device='cpu'):
        super(density_matrixT, self).__init__()

        self.index = index
        self.dim = dim
        self.n = N
        self.device = device

        self.count = contar_elementos(index)
        self.da = []
        self.db = []
        if self.count == 0:
            self.da.append(int(dim**len(index)))
            self.db.append(int( dim**( N-len(index) ) ))
        else:
            for i in range(self.count):
                self.da.append( int(dim**len(index[i])) )
                self.db.append(int( dim**( N-len(index[i]) ) ))


    def forward(self, x):
        rhoA = []
        if self.count == 0:
            state1 = NewState(x, self.index, self.dim, self.n,device=self.device)
            rhoA.append(pTraceB(state1, self.da[0], self.db[0],device=self.device))
            del state1
        else:
            for i in range(self.count):
                state1 = NewState(x, self.index[i], self.dim, self.n,device=self.device)
                rhoA.append(pTraceB(state1, self.da[i], self.db[i],device=self.device))
                del state1
        return torch.stack(rhoA)


class mean(nn.Module):

    def __init__(self, wires=None, H = None, dim=3,N=2,device='cpu'):
        super(mean, self).__init__()

        #self.index = wires
        self.dim = dim
        self.n = N
        self.device = device
        self.H = H.to(torch.complex64)

        self.dm = density_matrixT(index=wires,dim=dim,N=N,device=device)


    def forward(self, state):
        med = []

        for i in range(state.shape[1]):
            state_ = state[:,i]
            state_ = state_.view(state_.shape[0],-1)
            rhoA = self.dm(state_)

            Hrho = torch.matmul(self.H,rhoA)

            soma_diagonais = abs(torch.diagonal(Hrho, dim1=1, dim2=2).sum(dim=1))
            med.append(soma_diagonais)

        return torch.stack(med)


class CR(nn.Module):
    '''
    Controlled rotation gate
    TO DO: optimize this gate
    '''
    def __init__(self, wires=None, dim=2, mtx_id=0, j=1, k=2, device='cpu'):
        super(CR, self).__init__()

        self.D = dim
        self.control = wires[0]
        self.target =  wires[1]
        pi = np.pi
        self.angle = nn.Parameter(4*pi*torch.rand(1, device=device))
        S = sigma[mtx_id](j, k, Base(dim, device=device))
        self.register_buffer('S', S)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))

        U = 0.0
        for d in range(self.D):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(L):
                if i == self.control:
                    u = torch.kron(u, Base(self.D)[d] @ Base(self.D)[d].T)
                elif i == self.target:
                    M = torch.matrix_exp(-0.5*1j*self.angle*d*self.S)
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.D, device=x.device, dtype=torch.complex64))
            U += u
        return torch.sparse.mm(U,x)#U @ x


class CRX(nn.Module):
    '''
    Controlled rotation gate
    TO DO: optimize this gate
    '''
    def __init__(self, wires=None, dim=2, mtx_id=0, j=1, k=2, device='cpu'):
        super(CRX, self).__init__()

        self.D = dim
        self.control = wires[0]
        self.target =  wires[1]
        pi = np.pi
        self.angle = nn.Parameter(4*pi*torch.rand(1, device=device))
        S = sigma[mtx_id](j, k, Base(dim, device=device))
        self.register_buffer('S', S)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))

        U = 0.0
        for d in range(self.D):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(L):
                if i == self.control:
                    u = torch.kron(u, Base(self.D)[d] @ Base(self.D)[d].T)
                elif i == self.target:
                    M = torch.matrix_exp(-0.5*1j*self.angle*d*self.S)
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.D, device=x.device, dtype=torch.complex64))
            U += u
        return torch.sparse.mm(U,x)#U @ x



class CRY(nn.Module):
    '''
    Controlled rotation gate
    TO DO: optimize this gate
    '''
    def __init__(self, wires=None, dim=2, mtx_id=1, j=1, k=2, device='cpu'):
        super(CRY, self).__init__()

        self.D = dim
        self.control = wires[0]
        self.target =  wires[1]
        pi = np.pi
        self.angle = nn.Parameter(4*pi*torch.rand(1, device=device))
        S = sigma[mtx_id](j, k, Base(dim, device=device))
        self.register_buffer('S', S)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))

        U = 0.0
        for d in range(self.D):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(L):
                if i == self.control:
                    u = torch.kron(u, Base(self.D)[d] @ Base(self.D)[d].T)
                elif i == self.target:
                    M = torch.matrix_exp(-0.5*1j*self.angle*d*self.S)
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.D, device=x.device, dtype=torch.complex64))
            U += u
        return torch.sparse.mm(U,x)#U @ x



class CRZ(nn.Module):
    '''
    Controlled rotation gate
    TO DO: optimize this gate
    '''
    def __init__(self, wires=None, dim=2, mtx_id=2, j=1, k=2, device='cpu'):
        super(CRZ, self).__init__()

        self.D = dim
        self.control = wires[0]
        self.target =  wires[1]
        pi = np.pi
        self.angle = nn.Parameter(4*pi*torch.rand(1, device=device))
        S = sigma[mtx_id](j, k, Base(dim, device=device))
        self.register_buffer('S', S)

    def forward(self, x):
        L = round(log(x.shape[0], self.D))

        U = 0.0
        for d in range(self.D):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(L):
                if i == self.control:
                    u = torch.kron(u, Base(self.D)[d] @ Base(self.D)[d].T)
                elif i == self.target:
                    M = torch.matrix_exp(-0.5*1j*self.angle*d*self.S)
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.D, device=x.device, dtype=torch.complex64))
            U += u
        return torch.sparse.mm(U,x)#U @ x




def ajustar_dimensao(tensor):
    # Verificar as dimensões do tensor
    if tensor.dim() == 1:
        # Se o tensor tiver dimensão n, adicione uma dimensão extra
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 2:
        # Se o tensor não tiver dimensão mxn ou n, exiba uma mensagem de erro
        raise ValueError("O tensor deve ter dimensão mxn ou n")

    return tensor



def normalizar_linhas(tensor):
    # Calcula a norma de cada linha do tensor
    norma_linhas = torch.norm(tensor, dim=1, keepdim=True)

    # Normaliza as linhas do tensor
    tensor_normalizado = tensor / norma_linhas

    return tensor_normalizado



class AmplitudeEncoder(nn.Module):
    def __init__(self, dim=2, N=2,device='cpu'):
        super(AmplitudeEncoder, self).__init__()
        self.n = N
        self.d = dim
        self.device = device


    def forward(self, x):
        x = ajustar_dimensao(x)

        x = normalizar_linhas(x)

        zero = torch.zeros(x.shape[0], (self.d**self.n)-x.shape[1] ).to(self.device)
        y = torch.cat([x,zero],dim=1).to(self.device)

        return y.to(torch.complex64).T





class RotT(nn.Module):
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, mtx_id=0, j=1, k=2, index=0,dim=3,N=2,device='cpu'):
        super(RotT, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = index
        self.dim = dim

        self.base = Base(dim,device=device)

        self.esq,self.dir = nQudit(N,index)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = Identity_sparse(self.dim**self.dir,device=self.device)


    def forward(self, x,angle):


        M = torch.matrix_exp(-0.5*1j*angle*self.S).to(self.device)

        M = sparse(M,device=self.device)


        U = sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return U@x

def redimensionar_vetor(tensor):
    if tensor.dim() == 1:
        # Se o tensor já for de dimensão 1, retorna ele mesmo
        return tensor.unsqueeze(0)
    else:
        # Se for uma matriz mxn, não faz nada
        return tensor


class QuditEncoder(nn.Module):

    def __init__(self, input=False, mtx_id=1, j=1, k=2,dim=3,N=2,device='cpu'):
        super(QuditEncoder, self).__init__()

        self.state = State('0'*N,dim=dim,device=device)
        self.N = N
        self.layer = []
        self.data = input
        for i in range(N):
            self.layer.append( RotT( mtx_id=mtx_id, index=i, j=j, k=k,dim=dim,N=N,device=device ) )


    def forward(self, x=None):
        x1 = None
        if self.data is False:
            x1 = x
        else:
            x1 = self.data

        x1 = redimensionar_vetor(x1)
        out = []
        for j in range(x1.shape[0]):
            y = self.state
            for i in range(self.N):
                y = self.layer[i](y,x1[j][i])
            out.append(y)


        return torch.cat(out,dim=1)



class qmodel(nn.Module):
    #This class allows users to add gates dynamically
    def __init__(self, dim=2, wires=1, device='cpu'):
        super(qmodel, self).__init__()

        self.dim = dim
        self.N = wires
        self.device = device
        self.circuit = nn.Sequential()

    def add(self, module, **kwargs):
        gate = module(dim=self.dim,N=self.N, device=self.device, **kwargs)
        self.circuit.add_module(str(len(self.circuit)), gate)

    def add_gate(self, gate, **kwargs):
        self.circuit.add_module(str(len(self.circuit)), gate)

    def H(self, **kwargs):
        self.add_gate( H(dim=self.dim, N=self.N, device=self.device, **kwargs))

    def X(self, **kwargs):
        self.add_gate( X(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def Y(self, **kwargs):
        self.add_gate( Y(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def Z(self, **kwargs):
        self.add_gate( Z(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def R(self, **kwargs):
        self.add_gate( R(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def RX(self, **kwargs):
        self.add_gate( RX(mtx_id=0,dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def RY(self, **kwargs):
        self.add_gate( RY(mtx_id=1,dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def RZ(self, **kwargs):
        self.add_gate( RZ(mtx_id=2,dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def CNOT(self, **kwargs):
        self.add_gate( CNOT(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def CR(self, **kwargs):
        self.add_gate( CR(dim=self.dim, device=self.device, **kwargs) )

    def CRX(self, **kwargs):
        self.add_gate( CRX(dim=self.dim, device=self.device, **kwargs) )

    def CRY(self, **kwargs):
        self.add_gate( CRY(dim=self.dim, device=self.device, **kwargs) )

    def CRZ(self, **kwargs):
        self.add_gate( CRZ(dim=self.dim, device=self.device, **kwargs) )

    def density_matrix(self, **kwargs):
        self.add_gate( density_matrix(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def prob(self, **kwargs):
        self.add_gate( prob(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def mean(self, **kwargs):
        self.add_gate( mean(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def QuditEncoder(self, **kwargs):
        self.add_gate( QuditEncoder(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def AmplitudeEncoder(self, **kwargs):
        self.add_gate( AmplitudeEncoder(dim=self.dim,N=self.N, device=self.device, **kwargs) )

    def forward(self, x):
        return self.circuit(x)


