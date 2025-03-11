import numpy as np
import torch
import torch.nn as nn
from math import log as log
import aux
import itertools

class CustomGate(nn.Module):
    r"""
    Custom Quantum Gate for qudits.

    The CustomGate class allows users to define and apply a custom quantum gate to a specific qudit in a multi-qudit system. This gate can be any user-defined matrix, making it highly flexible for custom operations.

    **Details:**

    This gate applies a custom unitary matrix :math:`M` to the specified qudit while leaving other qudits unaffected by applying the identity operation to them.

    Args:
        M (torch.Tensor): The custom matrix to be applied as the gate.
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits (wires) in the circuit. Default is 1.
        index (int): The index of the qudit to which the custom gate is applied. Default is 0.
        device (str): The device to perform the computations on. Default is 'cpu'.

    Attributes:
        M (torch.Tensor): The custom matrix for the gate.
        dim (int): The dimension of the qudits.
        index (int): The index of the qudit on which the custom gate operates.
        wires (int): The total number of qudits in the system.
        device (str): The device for computations ('cpu' or 'cuda').

    Examples:
        >>> import quforge.quforge as qf
        >>> custom_matrix = torch.tensor([[0, 1], [1, 0]])  # Example custom matrix
        >>> gate = qf.CustomGate(M=custom_matrix, dim=2, index=0, wires=2)
        >>> state = qf.State('00')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, M, dim=2, wires=1, index=0, device='cpu'):
        super(CustomGate, self).__init__()
        self.M = M.type(torch.complex64).to(device)
        self.index = index
        self.dim = dim
        self.wires = wires

    def forward(self, x):
        """
        Apply the custom gate to the qudit state.

        Args:
            x (torch.Tensor): The input qudit state.

        Returns:
            torch.Tensor: The resulting state after applying the custom gate.
        """
        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(self.wires):
            if i == self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, device=x.device))
        
        return torch.matmul(U, x)


class RX(nn.Module):
    r"""
    Rotation-X (RX) Gate for qudits.

    The RX gate represents a rotation around the X-axis of the Bloch sphere in a qudit system.

    **Details:**

    * **Matrix Representation:**
    
    .. math::
          
            RX(\theta) = 
            \begin{pmatrix}
            \cos(\frac{\theta}{2}) & -i\sin(\frac{\theta}{2}) \\
            -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
            \end{pmatrix}
          
    for a 2-level qudit (qubit). For higher dimensions, the gate affects only the j-th and k-th levels, leaving the others unchanged.

    Args:
        j (int): The first state to rotate between. Default is 0.
        k (int): The second state to rotate between. Default is 1.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        dim (int): The dimension of the qudit. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        angle (float or bool): The angle of rotation. If False, the angle is learned as a parameter. Default is False.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    Attributes:
        j (int): The first state involved in the rotation.
        k (int): The second state involved in the rotation.
        index (list of int): The indices of the qudits to which the gate is applied.
        angle (torch.nn.Parameter): The angle of rotation (learned or set).
        sparse (bool): Whether the matrix representation is sparse.
        dim (int): The dimension of the qudit.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.RX(angle=1.57, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, j=0, k=1, index=[0], dim=2, wires=1, device='cpu', angle=False, sparse=False):
        super(RX, self).__init__()

        self.dim = dim
        self.wires = wires
        self.index = index
        self.j = j 
        self.k = k
        self.sparse = sparse

        if angle is False:
            self.angle = nn.Parameter(torch.randn(wires, device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(wires, device=device))

    def forward(self, x, param=False):

        """
        Apply the RX gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.
            param (torch.Tensor or bool): If False, use the internal angle. If provided, use it as the rotation angle.

        Returns:
            torch.Tensor: The resulting state after applying the RX gate.
        """
        
        L = round(log(x.shape[0], self.dim))
        U = aux.eye(1, device=x.device, sparse=self.sparse)

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
                    M = aux.eye(dim=self.dim, device=x.device, sparse=self.sparse)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    for n in range(self.dim):
                        new_tuple = torch.tensor([[n], [n]], device=indices.device)
                        is_present = ((indices == new_tuple).all(dim=0)).any()
                        if not is_present:
                            indices = torch.cat((indices, new_tuple), dim=1)
                            values = torch.cat((values, torch.tensor([1], dtype=values.dtype, device=values.device)))
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = aux.kron(U, M, sparse=self.sparse)
            else:
                U = aux.kron(U, aux.eye(self.dim, device=x.device, sparse=self.sparse), sparse=self.sparse)

        return U @ x


class RY(nn.Module):
    r"""
    Rotation-Y (RY) Gate for qudits.

    The RY gate represents a rotation around the Y-axis of the Bloch sphere in a qudit system.

    **Details:**

    * **Matrix Representation:**
    
    .. math::
          
            RY(\theta) = 
            \begin{pmatrix}
            \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\
            \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
            \end{pmatrix}
          
    for a 2-level qudit (qubit). For higher dimensions, the gate affects only the j-th and k-th levels, leaving the others unchanged.

    Args:
        j (int): The first state to rotate between. Default is 0.
        k (int): The second state to rotate between. Default is 1.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        dim (int): The dimension of the qudit. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        angle (float or bool): The angle of rotation. If False, the angle is learned as a parameter. Default is False.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    Attributes:
        j (int): The first state involved in the rotation.
        k (int): The second state involved in the rotation.
        index (list of int): The indices of the qudits to which the gate is applied.
        angle (torch.nn.Parameter): The angle of rotation (learned or set).
        sparse (bool): Whether the matrix representation is sparse.
        dim (int): The dimension of the qudit.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.RY(angle=1.57, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, j=0, k=1, index=[0], dim=2, wires=1, device='cpu', angle=False, sparse=False):
        super(RY, self).__init__()

        self.dim = dim
        self.wires = wires
        self.index = index
        self.sparse = sparse
        self.j = j 
        self.k = k

        if angle is False:
            self.angle = nn.Parameter(torch.randn(wires, device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(wires, device=device))

    def forward(self, x, param=False):
        """
        Apply the RY gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.
            param (torch.Tensor or bool): If False, use the internal angle. If provided, use it as the rotation angle.

        Returns:
            torch.Tensor: The resulting state after applying the RY gate.
        """
        L = round(log(x.shape[0], self.dim))
        U = aux.eye(1, device=x.device, sparse=self.sparse)

        for i in range(L):
            if i in self.index:
                indices = torch.tensor([[self.j, self.k, self.j, self.k], [self.j, self.k, self.k, self.j]], device=x.device)
                values = torch.zeros(4, dtype=torch.complex64, device=x.device)
                if param is False:
                    values[0] = torch.cos(self.angle[i]/2)
                    values[1] = torch.cos(self.angle[i]/2)
                    values[2] = -torch.sin(self.angle[i]/2)
                    values[3] = torch.sin(self.angle[i]/2)
                else:
                    values[0] = torch.cos(param[i]/2)
                    values[1] = torch.cos(param[i]/2)
                    values[2] = -torch.sin(param[i]/2)
                    values[3] = torch.sin(param[i]/2)
                
                if self.sparse is False:
                    M = aux.eye(dim=self.dim, device=x.device, sparse=self.sparse)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    for n in range(self.dim):
                        new_tuple = torch.tensor([[n], [n]], device=indices.device)
                        is_present = ((indices == new_tuple).all(dim=0)).any()
                        if not is_present:
                            indices = torch.cat((indices, new_tuple), dim=1)
                            values = torch.cat((values, torch.tensor([1], dtype=values.dtype, device=values.device)))
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = aux.kron(U, M, sparse=self.sparse)
            else:
                U = aux.kron(U, aux.eye(self.dim, device=x.device, sparse=self.sparse), sparse=self.sparse)

        return U @ x



class RZ(nn.Module):
    r"""
    Rotation-Z (RZ) Gate for qudits.

    The RZ gate represents a rotation around the Z-axis of the Bloch sphere in a qudit system.

    **Details:**

    * **Matrix Representation:**
    
    .. math::
          
            RZ(\theta) = 
            \begin{pmatrix}
            e^{-i\theta/2} & 0 \\
            0 & e^{i\theta/2}
            \end{pmatrix}
          
    for a 2-level qudit (qubit). For higher dimensions, the gate affects only the j-th level, leaving the others unchanged.

    Args:
        j (int): The state to apply the phase rotation. Default is 1.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        dim (int): The dimension of the qudit. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        angle (float or bool): The angle of rotation. If False, the angle is learned as a parameter. Default is False.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    Attributes:
        j (int): The state involved in the phase rotation.
        index (list of int): The indices of the qudits to which the gate is applied.
        angle (torch.nn.Parameter): The angle of rotation (learned or set).
        sparse (bool): Whether the matrix representation is sparse.
        dim (int): The dimension of the qudit.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.RZ(angle=1.57, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, j=1, index=[0], dim=2, wires=1, device='cpu', angle=False, sparse=False):
        super(RZ, self).__init__()

        self.dim = dim
        self.wires = wires
        self.index = index
        self.j = j
        self.sparse = sparse

        if angle is False:
            self.angle = nn.Parameter(torch.randn(wires, device=device))
        else:
            self.angle = nn.Parameter(angle*torch.ones(wires, device=device))

    def forward(self, x, param=False):
        """
        Apply the RZ gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.
            param (torch.Tensor or bool): If False, use the internal angle. If provided, use it as the rotation angle.

        Returns:
            torch.Tensor: The resulting state after applying the RZ gate.
        """
        L = round(log(x.shape[0], self.dim))
        U = aux.eye(1, device=x.device, sparse=self.sparse)
        
        for i in range(L):
            if i in self.index:
                if param is False:
                    indices = torch.tensor([range(self.j + 1), range(self.j + 1)], device=x.device)
                    angle = (self.angle[i] / 2) * np.sqrt(2 / (self.j * (self.j + 1)))
                    values = angle * torch.ones(self.j + 1, dtype=torch.complex64, device=x.device)
                    values[self.j] = values[self.j] * (-self.j)
                    values = torch.cos(values) - 1j * torch.sin(values)
                else:
                    indices = torch.tensor([range(self.j + 1), range(self.j + 1)], device=x.device)
                    angle = (param[i] / 2) * np.sqrt(2 / (self.j * (self.j + 1)))
                    values = angle * torch.ones(self.j + 1, dtype=torch.complex64, device=x.device)
                    values[self.j] = values[self.j] * (-self.j)
                    values = torch.cos(values) - 1j * torch.sin(values)
                
                if self.sparse is False:
                    M = aux.eye(dim=self.dim, device=x.device, sparse=self.sparse)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    for n in range(self.dim):
                        new_tuple = torch.tensor([[n], [n]], device=indices.device)
                        is_present = ((indices == new_tuple).all(dim=0)).any()
                        if not is_present:
                            indices = torch.cat((indices, new_tuple), dim=1)
                            values = torch.cat((values, torch.tensor([1], dtype=values.dtype, device=values.device)))
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = aux.kron(U, M, sparse=self.sparse)
            else:
                U = aux.kron(U, aux.eye(self.dim, sparse=self.sparse))
        
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
        >>> import quforge.quforge as qf
        >>> gate = qf.H(dim=2, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, dim=2, index=[0], inverse=False, device='cpu'):
        super(H, self).__init__()

        self.index = index
        self.device = device
        self.dim = dim
        omega = np.exp(2*1j*np.pi/dim)

        M = torch.ones((dim, dim), dtype=torch.complex64, device=device)
        for i in range(1, dim):
            for j in range(1, dim):
                M[i, j] = omega**(j*i)
        M = M/(dim**0.5)
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
    r"""
    Generalized Pauli-X (X) Gate for qudits.

    The X gate represents a cyclic shift of the computational basis states in a qudit system, generalizing the Pauli-X gate from qubits to higher dimensions.

    **Details:**

    * **Matrix Representation:**
    
    For a qubit (2-level qudit), the Pauli-X gate is represented as:
    
    .. math::
          
            X = 
            \begin{pmatrix}
            0 & 1 \\
            1 & 0
            \end{pmatrix}
    
    For a higher dimensional qudit, the X gate shifts the basis states by \( s \), where \( s \) is a cyclic shift parameter.

    Args:
        s (int): The cyclic shift parameter for the qudit. Default is 1.
        dim (int): The dimension of the qudit. Default is 2.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        device (str): The device to perform the computations on. Default is 'cpu'.
        inverse (bool): Whether to apply the inverse of the X gate. Default is False.

    Attributes:
        index (list of int): The indices of the qudits to which the gate is applied.
        dim (int): The dimension of the qudit.
        M (torch.Tensor): The matrix representation of the X gate.
        inverse (bool): Whether the matrix representation is inverted.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.X(s=1, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, s=1, dim=2, index=[0], device='cpu', inverse=False):
        super(X, self).__init__()

        self.index = index
        self.dim = dim
        
        # Construct the matrix representation of the X gate
        M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
        for i in range(dim):
            for j in range(dim):
                M[j][i] = torch.matmul(aux.base(dim)[j].T, aux.base(dim)[(i + s) % dim])
        
        # Apply the inverse if requested
        if inverse:
            M = torch.conj(M.T)
        
        # Register the matrix as a buffer so it can be used in the forward pass
        self.register_buffer('M', M)
            

    def forward(self, x):
        """
        Apply the X gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.

        Returns:
            torch.Tensor: The resulting state after applying the X gate.
        """
        L = round(log(x.shape[0], self.dim))  # Determine the number of qudits
        U = torch.eye(1, dtype=torch.complex64, device=x.device)  # Identity matrix for the initial state
        
        # Apply the X gate to the specified qudits
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, dtype=torch.complex64, device=x.device))
        
        return torch.matmul(U, x)



class Z(nn.Module):
    r"""
    Generalized Pauli-Z (Z) Gate for qudits.

    The Z gate represents a phase shift of the computational basis states in a qudit system, generalizing the Pauli-Z gate from qubits to higher dimensions.

    **Details:**

    * **Matrix Representation:**
    
    For a qubit (2-level qudit), the Pauli-Z gate is represented as:
    
    .. math::
          
            Z = 
            \begin{pmatrix}
            1 & 0 \\
            0 & -1
            \end{pmatrix}
    
    For a higher-dimensional qudit, the Z gate applies a phase shift based on a complex exponential parameter \( \omega \).

    .. math::
    
            Z_s = \text{diag}(1, \omega^s, \omega^{2s}, \ldots, \omega^{(D-1)s})
    
    where \( \omega = e^{2\pi i / D} \) and \( D \) is the dimension of the qudit.

    Args:
        dim (int): The dimension of the qudit. Default is 2.
        s (int): The phase shift parameter for the qudit. Default is 1.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        device (str): The device to perform the computations on. Default is 'cpu'.
        inverse (bool): Whether to apply the inverse of the Z gate. Default is False.

    Attributes:
        index (list of int): The indices of the qudits to which the gate is applied.
        dim (int): The dimension of the qudit.
        M (torch.Tensor): The matrix representation of the Z gate.
        inverse (bool): Whether the matrix representation is inverted.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.Z(dim=3, s=1, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, dim=2, s=1, index=[0], device='cpu', inverse=False):
        super(Z, self).__init__()

        # Phase factor omega
        omega = np.exp(2 * 1j * np.pi / dim)

        self.index = index
        self.dim = dim

        # Construct the matrix representation of the Z gate
        M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
        for i in range(dim):
            for j in range(dim):
                M[j][i] = (omega ** (j * s)) * aux.delta(i, j)  # Apply phase shift using delta function

        # Apply inverse if requested
        if inverse:
            M = torch.conj(M.T)

        # Register the matrix as a buffer so it can be used in the forward pass
        self.register_buffer('M', M)
        
    def forward(self, x):
        """
        Apply the Z gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.

        Returns:
            torch.Tensor: The resulting state after applying the Z gate.
        """
        L = round(log(x.shape[0], self.dim))  # Determine the number of qudits
        U = torch.eye(1, device=x.device, dtype=torch.complex64)  # Identity matrix for the initial state
        
        # Apply the Z gate to the specified qudits
        for i in range(L):
            if i in self.index:
                U = torch.kron(U, self.M)
            else:
                U = torch.kron(U, torch.eye(self.dim, device=x.device, dtype=torch.complex64))
        
        return torch.matmul(U, x)

    def gate(self):
        """
        Return the matrix representation of the Z gate.

        Returns:
            torch.Tensor: The Z gate matrix.
        """
        return self.M



class Y(nn.Module):
    r"""
    Generalized Pauli-Y (Y) Gate for qudits.

    The Y gate represents a combination of the X and Z gates, generalizing the Pauli-Y gate from qubits to higher dimensions.

    **Details:**

    * **Matrix Representation:**
    
    For a qubit (2-level qudit), the Pauli-Y gate is represented as:
    
    .. math::
          
            Y = 
            \begin{pmatrix}
            0 & -i \\
            i & 0
            \end{pmatrix}
    
    For a higher-dimensional qudit, the Y gate is defined as the product of the Z and X gates:

    .. math::
    
            Y = \frac{1}{i} Z \cdot X

    where \( Z \) and \( X \) are the generalized Pauli-Z and Pauli-X gates, respectively.

    Args:
        s (int): The cyclic shift parameter for the X gate. Default is 1.
        dim (int): The dimension of the qudit. Default is 2.
        index (list of int): The indices of the qudits to which the gate is applied. Default is [0].
        device (str): The device to perform the computations on. Default is 'cpu'.

    Attributes:
        index (list of int): The indices of the qudits to which the gate is applied.
        dim (int): The dimension of the qudit.
        M (torch.Tensor): The matrix representation of the Y gate.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.Y(s=1, index=[0])
        >>> state = qf.State('0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, s=1, dim=2, index=[0], device='cpu'):
        super(Y, self).__init__()

        self.index = index
        self.dim = dim

        # Generate X and Z gates and calculate Y as Z * X / 1j
        x = X(s=s, device=device).M  # Generalized Pauli-X gate
        z = Z(device=device).M       # Generalized Pauli-Z gate
        M = torch.matmul(z, x) / 1j      # Y = Z * X / 1j

        # Register the matrix as a buffer so it can be used in the forward pass
        self.register_buffer('M', M)
        
    def forward(self, x):
        """
        Apply the Y gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudit.

        Returns:
            torch.Tensor: The resulting state after applying the Y gate.
        """
        L = round(log(x.shape[0], self.dim))  # Determine the number of qudits
        U = torch.eye(1, device=x.device, dtype=torch.complex64)  # Identity matrix for the initial state
        
        # Apply the Y gate to the specified qudits
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
                M[j][i] = torch.matmul(aux.base(dim, device=device)[j].T, aux.base(dim, device=device)[(dim-i) % dim])
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


class I(nn.Module):
    #index: index of the qudit to apply the gate
    def __init__(self, dim=2, wires=1, device='cpu', sparse=False):
        super(I, self).__init__()

        self.U = aux.eye(dim**wires, device=device, sparse=sparse)
        
    def forward(self, x):
        return torch.matmul(self.U, x)


class CNOT(nn.Module):
    r"""
    Controlled-NOT (CNOT) Gate for qudits.

    The CNOT gate is a controlled gate where the target qudit is flipped based on the state of the control qudit. For qudits, this gate is generalized to perform a cyclic shift of the target qudit based on the control qudit.

    **Details:**

    * **Matrix Representation:**
    
    For qubits (2-level qudits), the CNOT gate is represented as:
    
    .. math::
          
            CNOT = 
            \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
            \end{pmatrix}
    
    For higher-dimensional qudits, the CNOT gate performs a modular addition of the control and target qudits.

    Args:
        index (list of int): The control and target qudit indices, where `index[0]` is the control qudit and `index[1]` is the target qudit. Default is [0, 1].
        wires (int): The total number of qudits in the circuit. Default is 2.
        dim (int): The dimension of the qudits. Default is 2.
        device (str): The device to perform the computations on. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.
        inverse (bool): Whether to apply the inverse of the CNOT gate. Default is False.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        dim (int): The dimension of the qudits.
        U (torch.Tensor): The matrix representation of the CNOT gate.
        inverse (bool): Whether the matrix representation is inverted.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.CNOT(index=[0, 1], wires=2)
        >>> state = qf.State('0-0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0,1], wires=2, dim=2, device='cpu', sparse=False, inverse=False):
        super(CNOT, self).__init__()

        # Dense matrix implementation
        if sparse is False:
            L = torch.tensor(list(itertools.product(range(dim), repeat=wires)))
            l2ns = L.clone()
            l2ns[:, index[1]] = (l2ns[:, index[0]] + l2ns[:, index[1]]) % dim
            indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
            U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64)).to(device)
        # Sparse matrix implementation
        else:
            U = aux.CNOT_sparse(index[0], index[1], dim, wires, device=device)

        # Apply inverse if requested
        if inverse:
            U = torch.conj(U).T.contiguous()

        # Register the matrix as a buffer so it can be used in the forward pass
        self.register_buffer('U', U)
        
    def forward(self, x):
        """
        Apply the CNOT gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CNOT gate.
        """
        return self.U @ x



class SWAP(nn.Module):
    r"""
    SWAP Gate for qudits.

    The SWAP gate exchanges the states of two qudits, generalizing the SWAP gate for qubits to higher-dimensional qudits.

    **Details:**

    * **Matrix Representation:**
    
    For a qubit (2-level qudit) system, the SWAP gate exchanges the states of two qubits. The matrix representation is:
    
    .. math::
          
            SWAP = 
            \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
            \end{pmatrix}
    
    For qudits, the SWAP gate exchanges the states of the two qudits specified by their indices.

    Args:
        index (list of int): The indices of the qudits to be swapped. Default is [0, 1].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 2.
        device (str): The device to perform the computations on. Default is 'cpu'.

    Attributes:
        index (list of int): The indices of the qudits to be swapped.
        dim (int): The dimension of the qudits.
        U (torch.Tensor): The matrix representation of the SWAP gate.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.SWAP(index=[0, 1], dim=2, wires=2)
        >>> state = qf.State('0-1')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0,1], dim=2, wires=2, device='cpu'):
        super(SWAP, self).__init__()

        c = index[0]  # Control qudit index
        t = index[1]  # Target qudit index
        D = dim ** wires  # Total dimension of the system (dim^wires)

        # Initialize the SWAP gate matrix as a zero matrix
        U = torch.zeros((D, D), device=device, dtype=torch.complex64)

        # Construct the SWAP matrix by swapping states of qudits
        for k in range(D):
            localr = aux.dec2den(k, wires, dim)  # Convert from decimal to local qudit representation
            locall = localr.copy()
            locall[c] = localr[t]  # Swap qudits
            locall[t] = localr[c]  # Swap qudits
            globall = aux.den2dec(locall, dim)  # Convert back to decimal
            U[globall, k] = 1  # Set the matrix element to 1 for the swapped states

        # Register the matrix as a buffer so it can be used in the forward pass
        self.register_buffer('U', U)

    def forward(self, x):
        """
        Apply the SWAP gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the SWAP gate.
        """
        return self.U @ x



class CCNOT(nn.Module):
    r"""
    CCNOT (Toffoli) Gate for qudits.

    The CCNOT gate, also known as the Toffoli gate, is a controlled-controlled NOT gate that flips the target qudit if both control qudits are in the specified states.

    **Details:**

    * **Matrix Representation:**
    
    This gate applies the following transformation:
    
    .. math::
          
            |c_1 c_2 t\rangle \to |c_1 c_2 (c_1 \cdot c_2 \oplus t)\rangle

    where :math:`c_1` and :math:`c_2` are the control qudits, :math:`t` is the target qudit, and :math:`\oplus` is the XOR operation (modulo the dimension of the qudit, D).

    Args:
        index (list of int): The indices of the control and target qudits. The first two are the control qudits, and the third is the target qudit. Default is [0, 1, 2].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits. Default is 3.
        inverse (bool): Whether to apply the inverse of the CCNOT gate. Default is False.
        device (str): The device to perform the computations on. Default is 'cpu'.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        device (str): The device to perform the computations on.
        dim (int): The dimension of the qudits.
        U (torch.Tensor): The matrix representation of the CCNOT gate.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.CCNOT(index=[0, 1, 2], dim=2, wires=3)
        >>> state = qf.State('0-0-0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0,1,2], dim=2, wires=3, inverse=False, device='cpu'):
        super(CCNOT, self).__init__()        
        L = torch.tensor(list(itertools.product(range(dim), repeat=wires))).to(device)
        l2ns = L.clone()
        l2ns[:, index[2]] = (l2ns[:, index[0]] * l2ns[:, index[1]] + l2ns[:, index[2]]) % dim
        indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(indices, torch.tensor([1.0 + 0j], dtype=torch.complex64), torch.tensor([0.0], dtype=torch.complex64))        
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer('U', U)
        
    def forward(self, x):
        """
        Apply the CCNOT gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CCNOT gate.
        """
        return torch.matmul(self.U, x)



class MCX(nn.Module):
    r"""
    Multi-Controlled CX Gate for qudits.

    The MCX gate applies a controlled-X operation where multiple control qudits are used to control a target qudit. This is a generalized version of the controlled-X gate for qudit systems.

    **Details:**

    * **Matrix Representation:**
    
    This gate applies the following transformation:
    
    .. math::
          
            |c_1 c_2 \dots c_n t\rangle \to |c_1 c_2 \dots c_n (c_1 \cdot c_2 \cdot \dots \cdot c_{n-1} \oplus t)\rangle

    where :math:`c_1, c_2, \dots, c_{n-1}` are the control qudits, and :math:`t` is the target qudit. The XOR operation :math:`\oplus` is modulo the dimension of the qudit, D.

    Args:
        index (list of int): The indices of the control and target qudits. The last element is the target qudit, and the others are the control qudits. Default is [0, 1].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits. Default is 2.
        inverse (bool): Whether to apply the inverse of the MCX gate. Default is False.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        dim (int): The dimension of the qudits.
        U (torch.Tensor): The matrix representation of the MCX gate.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.MCX(index=[0, 1], dim=2, wires=2)
        >>> state = qf.State('0-0')
        >>> result = gate(state)
        >>> print(result)
    """

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
        """
        Apply the MCX gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the MCX gate.
        """
        return torch.matmul(self.U, x)



class CZ(nn.Module):
    r"""
    Controlled-Z Gate for qudits.

    The CZ gate applies a Z operation on the target qudit if the control qudit is in a specific state. This is a generalized version of the controlled-Z gate for qudit systems.

    **Details:**

    * **Matrix Representation:**
    
    This gate applies the following transformation:
    
    .. math::
          
            |c t\rangle \to Z_t |c t\rangle \quad \text{if} \quad c = d

    where :math:`c` is the control qudit, :math:`t` is the target qudit, and :math:`d` represents the state that triggers the Z gate.

    Args:
        index (list of int): The indices of the control and target qudits. Default is [0, 1].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits. Default is 2.
        device (str): The device to perform the computations on. Default is 'cpu'.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        dim (int): The dimension of the qudits.
        U (torch.Tensor): The matrix representation of the CZ gate.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.CZ(index=[0, 1], dim=2, wires=2)
        >>> state = qf.State('0-0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, device='cpu'):
        super(CZ, self).__init__()

        self.dim = dim
        self.index = index
        self.wires = wires

        U = 0.0
        for d in range(dim):
            u = torch.eye(1, device=device, dtype=torch.complex64)
            for i in range(wires):
                if i == index[0]:
                    u = torch.kron(u, aux.base(dim)[d] @ aux.base(dim)[d].T)
                elif i == index[1]:
                    M = Z(dim=dim, device=device, s=d).gate()
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(dim, device=device, dtype=torch.complex64))
            U += u

        self.register_buffer('U', U)
    
    def forward(self, x):
        """
        Apply the CZ gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CZ gate.
        """
        return self.U @ x



class CRX(nn.Module):
    r"""
    Controlled-RX Gate for qudits.

    The CRX gate applies an RX (rotation-X) operation on the target qudit, conditioned on the state of the control qudit. This gate is generalized for qudit systems.

    **Details:**

    * **Matrix Representation:**
    
    This gate applies the following transformation:

    .. math::
          
            |c t\rangle \to RX(\theta)_t |c t\rangle \quad \text{if} \quad c = d

    where :math:`c` is the control qudit, :math:`t` is the target qudit, and :math:`RX(\theta)` represents a rotation around the X-axis by angle :math:`\theta`.

    Args:
        index (list of int): The indices of the control and target qudits. Default is [0, 1].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 2.
        j (int): The initial state of the target qudit. Default is 0.
        k (int): The target state after the RX rotation. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        dim (int): The dimension of the qudits.
        angle (torch.Tensor): The angle of rotation, a learnable parameter.
        wires (int): The total number of qudits in the circuit.
        sparse (bool): Whether a sparse matrix representation is used.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.CRX(index=[0, 1], dim=2, wires=2, j=0, k=1)
        >>> state = qf.State('0-0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0,1], dim=2, wires=2, j=0, k=1, device='cpu', sparse=False):
        super(CRX, self).__init__()

        self.index = index
        self.dim = dim
        self.j = j
        self.k = k
        self.angle = nn.Parameter(np.pi*torch.randn(1, device=device))
        self.wires = wires
        self.sparse = sparse
    
    def forward(self, x):
        """
        Apply the CRX gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CRX gate.
        """
        c = self.index[0]
        t = self.index[1]
        j = self.j + 1
        k = self.k + 1
        
        D = self.dim**self.wires
        U = torch.zeros((D, D), dtype=torch.complex64, device=x.device)
        Dl = D // self.dim
        indices_list = []
        values_list = []

        for m in range(Dl):
            local = aux.dec2den(m, self.wires-1, self.dim)
            if self.wires == 2:
                angle = (local[0]*self.angle)/2
            else:
                angle = (local[c]*self.angle)/2

            listj = local.copy()
            listj.insert(t, self.j-1)
            intj = aux.den2dec(listj, self.dim)
            listk = local.copy()
            listk.insert(t, self.k-1)
            intk = aux.den2dec(listk, self.dim)

            indices = torch.tensor([[intj, intk, intj, intk], [intj, intk, intk, intj]])

            values = torch.zeros(4, dtype=torch.complex64)
            values[0] = torch.cos(angle)
            values[1] = torch.cos(angle)
            values[2] = -1j*torch.sin(angle)
            values[3] = -1j*torch.sin(angle)

            for l in range(self.dim):
                if l != self.j-1 and l != self.k-1:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dim)
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
    r"""
    Controlled-RY Gate for qudits.

    The CRY gate applies an RY (rotation-Y) operation on the target qudit, conditioned on the state of the control qudit. This gate is generalized for qudit systems.

    **Details:**

    * **Matrix Representation:**
    
    This gate applies the following transformation:

    .. math::
          
            |c t\rangle \to RY(\theta)_t |c t\rangle \quad \text{if} \quad c = d

    where :math:`c` is the control qudit, :math:`t` is the target qudit, and :math:`RY(\theta)` represents a rotation around the Y-axis by angle :math:`\theta`.

    Args:
        index (list of int): The indices of the control and target qudits. Default is [0, 1].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 2.
        j (int): The initial state of the target qudit. Default is 0.
        k (int): The target state after the RY rotation. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        dim (int): The dimension of the qudits.
        angle (torch.Tensor): The angle of rotation, a learnable parameter.
        wires (int): The total number of qudits in the circuit.
        sparse (bool): Whether a sparse matrix representation is used.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.CRY(index=[0, 1], dim=2, wires=2, j=0, k=1)
        >>> state = qf.State('0-0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0,1], dim=2, wires=2, j=0, k=1, device='cpu', sparse=False):
        super(CRY, self).__init__()

        self.index = index
        self.dim = dim
        self.j = j
        self.k = k
        self.angle = nn.Parameter(np.pi*torch.randn(1, device=device))
        self.wires = wires
        self.sparse = sparse
    
    def forward(self, x):
        """
        Apply the CRY gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CRY gate.
        """
        c = self.index[0]
        t = self.index[1]
        j = self.j + 1
        k = self.k + 1
        
        D = self.dim**self.wires
        U = torch.zeros((D, D), dtype=torch.complex64, device=x.device)
        Dl = D // self.dim
        indices_list = []
        values_list = []

        if c < t:
            c_local = c
        else:
            c_local = c - 1

        for m in range(Dl):
            local = dec2den(m, self.wires-1, self.dim)
            angle = (local[c_local] * self.angle) / 2

            listj = local.copy()
            listj.insert(t, j-1)
            intj = aux.den2dec(listj, self.dim)
            listk = local.copy()
            listk.insert(t, k-1)
            intk = aux.den2dec(listk, self.dim)

            indices = torch.tensor([[intj, intk, intj, intk], [intj, intk, intk, intj]]).to(x.device)

            values = torch.zeros(4, dtype=torch.complex64, device=x.device)
            values[0] = torch.cos(angle)
            values[1] = torch.cos(angle)
            values[2] = -torch.sin(angle)
            values[3] = -torch.sin(angle)

            for l in range(self.dim):
                if l != j-1 and l != k-1:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dim)
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
    r"""
    Controlled-RZ Gate for qudits.

    The CRZ gate applies an RZ (rotation-Z) operation on the target qudit, conditioned on the state of the control qudit. This gate is generalized for qudit systems.

    **Details:**

    * **Matrix Representation:**
    
    This gate applies the following transformation:

    .. math::
          
            |c t\rangle \to RZ(\theta)_t |c t\rangle \quad \text{if} \quad c = d

    where :math:`c` is the control qudit, :math:`t` is the target qudit, and :math:`RZ(\theta)` represents a rotation around the Z-axis by angle :math:`\theta`.

    Args:
        index (list of int): The indices of the control and target qudits. Default is [0, 1].
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits in the circuit. Default is 2.
        j (int): The rotation state threshold for the Z operation. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    Attributes:
        index (list of int): The indices of the control and target qudits.
        dim (int): The dimension of the qudits.
        angle (torch.Tensor): The angle of rotation, a learnable parameter.
        wires (int): The total number of qudits in the circuit.
        sparse (bool): Whether a sparse matrix representation is used.

    Examples:
        >>> import quforge.quforge as qf
        >>> gate = qf.CRZ(index=[0, 1], dim=2, wires=2, j=1)
        >>> state = qf.State('0-0')
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, j=1, device='cpu', sparse=False):
        super(CRZ, self).__init__()

        self.index = index
        self.dim = dim
        self.j = j
        self.angle = nn.Parameter(np.pi*torch.randn(1, device=device))
        self.wires = wires
        self.sparse = sparse
    
    def forward(self, x):
        """
        Apply the CRZ gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CRZ gate.
        """
        c = self.index[0]
        t = self.index[1]
        
        D = self.dim**self.wires
        Dl = D // self.dim
        indices_list = []
        values_list = []

        indices = []
        values = []
        for m in range(Dl):
            local = aux.dec2den(m, self.wires-1, self.dim)
            if self.wires == 2:
                loc = local[0]
            else:
                loc = local[c]
            angle = ((loc * self.angle) / 2) * np.sqrt(2 / (self.j * (self.j + 1)))

            for k in range(self.dim):
                listk = local.copy()
                listk.insert(t, k)  # insert k in position t of the list
                intk = aux.den2dec(listk, self.dim)  # integer for the k state
                if k < self.j:
                    indices.append([intk, intk])
                    values.append(torch.cos(angle) - 1j * torch.sin(angle))
                elif k == self.j:
                    angle = self.j * angle
                    indices.append([intk, intk])
                    values.append(torch.cos(angle) + 1j * torch.sin(angle))
                elif k > self.j:
                    indices.append([intk, intk])
                    values.append(1.0)

        indices = torch.tensor(indices).T
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

class U(nn.Module):

    def __init__(self, dim=2, wires=1, device='cpu'):
        super(U, self).__init__()

        self.counter = 0
        self.U = nn.Parameter(aux.eye(dim=dim**wires, sparse=False, device=device) + torch.randn((dim**wires, dim**wires), device=device) + 1j*torch.randn((dim**wires, dim**wires), device=device))
    
    def forward(self, x):

        U = self.U - torch.conj(self.U.T)
        U = torch.matrix_exp(U)

        return U @ x


class CU(nn.Module):
    
    def __init__(self, dim=2, wires=2, index=[0, 1], control_state = [1], device='cpu'):
        super(CU, self).__init__()

        self.device = device
        self.wires = wires
        self.index = index
        self.dim = dim
        self.control_state = control_state
        self.M = nn.Parameter(aux.eye(dim=dim, sparse=False, device=device) + torch.randn((dim, dim), device=device) + 1j*torch.randn((dim, dim), device=device))
    
    def forward(self, x):

        M = self.M - torch.conj(self.M.T)
        M = torch.matrix_exp(M)

        U = 0.0
        for d in range(self.dim):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(self.wires):
                if i == self.index[0]:
                    u = torch.kron(u, aux.base(self.dim, device=x.device)[d] @ aux.base(self.dim, device=x.device)[d].T)
                elif i == self.index[1] and d in self.control_state:
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.dim, device=x.device, dtype=torch.complex64))
            U += u

        return U @ x