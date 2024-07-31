import torch
import torch.nn as nn
import numpy as np 
from math import log as log
from math import factorial as factorial
import itertools

class HGate(nn.Module):
    r"""
    Hadamard Gate for qudits.

    The Hadamard gate creates a superposition of states in a qudit system.

    **Details:**

    * **Matrix Representation:**
    
    .. math::
          
            H = \\frac{1}{\\sqrt{D}}
            \\begin{pmatrix}
            1 & 1 & \cdots & 1 \\\\
            1 & \omega & \cdots & \omega^{D-1} \\\\
            1 & \omega^2 & \cdots & \omega^{2(D-1)} \\\\
            \vdots & \vdots & \ddots & \vdots \\\\
            1 & \omega^{D-1} & \cdots & \omega^{(D-1)(D-1)}
            \\end{pmatrix}

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