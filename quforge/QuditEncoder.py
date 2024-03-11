import torch
import torch.nn as nn
import numpy as np
from math import log as log
import itertools
from torch.nn.parameter import Parameter

#import gates2.0

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
