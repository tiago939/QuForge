import torch
import torch.nn as nn
import numpy as np
from math import log as log
import itertools
from torch.nn.parameter import Parameter

import Gate as opt

sigma = [opt.Sx, opt.Sy, opt.Sz]


class RotT(nn.Module):
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, mtx_id=0, j=0, k=1, index=0,dim=3,N=2,device='cpu'):
        super(RotT, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = index
        self.dim = dim

        self.base = opt.Base(dim,device=device)

        self.esq,self.dir = opt.nQudit(N,index)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = opt.Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = opt.Identity_sparse(self.dim**self.dir,device=self.device)


    def forward(self, x,angle):


        M = torch.matrix_exp(-0.5*1j*angle*self.S).to(self.device)

        M = opt.sparse(M,device=self.device)


        U = opt.sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = opt.sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return U@x

def redimensionar_vetor(tensor):
    if tensor.dim() == 1:
        # Se o tensor já for de dimensão 1, retorna ele mesmo
        return tensor.unsqueeze(0)
    else:
        # Se for uma matriz mxn, não faz nada
        return tensor


class QubitEncoder(nn.Module):

    def __init__(self, mtx_id=0, j=1, k=2,dim=3,N=2,device='cpu'):
        super(QubitEncoder, self).__init__()

        self.state = opt.State('0'*N,dim=dim,device=device)
        self.N = N
        self.layer = []
        for i in range(N):
            self.layer.append( opt.RotT( mtx_id=mtx_id, index=i, j=j, k=k,dim=dim,N=N,device=device ) )


    def forward(self, x):
        x = redimensionar_vetor(x)
        out = []
        for j in range(x.shape[0]):
            y = self.state
            for i in range(self.N):
                y = self.layer[i](y,x[j][i])
            out.append(y)


        return torch.cat(out,dim=1)



