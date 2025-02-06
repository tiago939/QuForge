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


def partial_trace(state, index=[0], dim=2, wires=1):
    da = dim**len(index)
    db = dim**(wires - len(index))

    # Create tensors for indices
    all_indices = torch.arange(wires, device=state.device)
    index_tensor = torch.tensor(index, device=state.device)
    
    # Create mask and complementary indices
    complementary_indices = all_indices[~torch.isin(all_indices, index_tensor)]
    
    # Sort and concatenate indices
    new_order = torch.cat((index_tensor.sort()[0], complementary_indices), dim=0)

    # Transpose
    reshaped_state = state.view((dim,) * wires).permute(*new_order).contiguous().view(da, db)
    
    # matmul and conjugation operations
    state_conj = reshaped_state.conj()
    rho = torch.matmul(reshaped_state, state_conj.transpose(0, 1))

    return rho


def projector(index, dim):
    P = torch.zeros((dim, dim), dtype=torch.complex64)
    P[index][index] = 1.0

    return P


def measure(state=None, index=[0], shots=1, dim=2, wires=1):
    #input:
        #state: state to measure
        #index: list of qudits to measure
        #shots: number of measurements
    #output:
        #histogram: histogram of the measurements
        #p: distribution probability
    rho = partial_trace(state, index, dim, wires)
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

def zeros(m,n, device='cpu'):
    M = torch.zeros((m, n), device=device)
    return M

def ones(m,n, device='cpu'):
    M = torch.ones((m, n), device=device)
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

def dec2den(j,N,d):
    # convert from decimal to denary representation
    den = [0 for k in range(0,N)]
    jv = j
    for k in range(0,N):
        if jv >= d**(N-1-k):
            den[k] = jv//(d**(N-1-k))
            jv = jv - den[k]*d**(N-1-k)
    return den

def den2dec(local,d):
    # convert from denary to decimal representation
    # local = list with the local computational base state values 
    # d = individual qudit dimension
    N = len(local)
    j = 0
    for k in range(0,N):
        j += local[k]*d**(N-1-k)
    return j # value of the global computational basis index

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


class ReLU(nn.ReLU):
    def __init__(self):
        super(ReLU, self).__init__()
    
    def forward(self, x):
        return torch.relu(x)

class Sigmoid(nn.Sigmoid):
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(x)

class Tanh(nn.Tanh):
    def __init__(self):
        super(Tanh, self).__init__()
    
    def forward(self, x):
        return torch.tanh(x)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        return torch.nn.LeakyReLU(x, negative_slope=self.negative_slope)

class Softmax(nn.Softmax):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


class Circuit(nn.Module):
    r"""
    Quantum Circuit for qudits.

    The Circuit class allows users to dynamically add various quantum gates to construct a quantum circuit for qudit systems. It supports a wide range of gates, including single, multi-qudit, and custom gates. The class provides methods to add specific gates as well as a general interface for adding custom gates.

    **Details:**

    This class facilitates the construction of quantum circuits by allowing the sequential addition of gates. The circuit is represented as a sequence of quantum operations (gates) that act on qudit states.

    Args:
        dim (int): The dimension of the qudits. Default is 2.
        wires (int): The total number of qudits (wires) in the circuit. Default is 1.
        device (str): The device to perform the computations on. Default is 'cpu'.
        sparse (bool): Whether to use sparse matrix representations for the gates. Default is False.

    Attributes:
        dim (int): The dimension of the qudits.
        wires (int): The number of qudits in the circuit.
        device (str): The device for computations ('cpu' or 'cuda').
        circuit (nn.Sequential): A sequential container for holding the quantum gates.
        sparse (bool): Whether to use sparse matrices in the gates.

    Methods:
        add(module, **kwargs): Dynamically add a gate module to the circuit.
        add_gate(gate, **kwargs): Add a specific gate instance to the circuit.
        H(**kwargs): Add a Hadamard gate to the circuit.
        RX(**kwargs): Add a rotation-X gate to the circuit.
        RY(**kwargs): Add a rotation-Y gate to the circuit.
        RZ(**kwargs): Add a rotation-Z gate to the circuit.
        X(**kwargs): Add a Pauli-X gate to the circuit.
        Y(**kwargs): Add a Pauli-Y gate to the circuit.
        Z(**kwargs): Add a Pauli-Z gate to the circuit.
        CNOT(**kwargs): Add a controlled-NOT gate to the circuit.
        SWAP(**kwargs): Add a SWAP gate to the circuit.
        CZ(**kwargs): Add a controlled-Z gate to the circuit.
        CCNOT(**kwargs): Add a Toffoli (CCNOT) gate to the circuit.
        MCX(**kwargs): Add a multi-controlled-X gate to the circuit.
        CRX(**kwargs): Add a controlled rotation-X gate to the circuit.
        CRY(**kwargs): Add a controlled rotation-Y gate to the circuit.
        CRZ(**kwargs): Add a controlled rotation-Z gate to the circuit.
        Custom(**kwargs): Add a custom gate to the circuit.

    Examples:
        >>> import quforge.quforge as qf
        >>> circuit = qf.Circuit(dim=2, wires=3, device='cpu')
        >>> circuit.H(index=[0])
        >>> circuit.CNOT(index=[0, 1])
        >>> state = qf.State('0-0-0')
        >>> result = circuit(state)
        >>> print(result)
    """

    def __init__(self, dim=2, wires=1, device='cpu', sparse=False):
        super(Circuit, self).__init__()

        self.dim = dim 
        self.wires = wires
        self.device = device
        self.circuit = nn.Sequential()
        self.sparse = sparse

    def add(self, module, **kwargs):
        """
        Add a gate module dynamically to the circuit.

        Args:
            module: The gate module to add.
            **kwargs: Additional arguments for the gate.
        """
        gate = module(D=self.dim, device=self.device, **kwargs)
        self.circuit.add_module(str(len(self.circuit)), gate)

    def add_gate(self, gate, **kwargs):
        """
        Add a pre-instantiated gate to the circuit.

        Args:
            gate: The gate instance to add.
            **kwargs: Additional arguments for the gate.
        """
        self.circuit.add_module(str(len(self.circuit)), gate)

    def H(self, **kwargs):
        self.add_gate(H(dim=self.dim, device=self.device, **kwargs))

    def RX(self, **kwargs):
        self.add_gate(RX(dim=self.dim, wires=self.wires, device=self.device, sparse=self.sparse, **kwargs))

    def RY(self, **kwargs):
        self.add_gate(RY(dim=self.dim, wires=self.wires, device=self.device, sparse=self.sparse, **kwargs))

    def RZ(self, **kwargs):
        self.add_gate(RZ(dim=self.dim, wires=self.wires, device=self.device, sparse=self.sparse, **kwargs))

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
        self.add_gate(CZ(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def CCNOT(self, **kwargs):
        self.add_gate(CCNOT(dim=self.dim, device=self.device, **kwargs))

    def MCX(self, **kwargs):
        self.add_gate(MCX(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def CRX(self, **kwargs):
        self.add_gate(CRX(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def CRY(self, **kwargs):
        self.add_gate(CRY(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def CRZ(self, **kwargs):
        self.add_gate(CRZ(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def U(self, **kwargs):
        self.add_gate(U(dim=self.dim, device=self.device, wires=self.wires, **kwargs))

    def CU(self, **kwargs):
        self.add_gate(CU(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def Custom(self, **kwargs):
        self.add_gate(CustomGate(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def forward(self, x):
        """
        Apply the circuit to the input qudit state.

        Args:
            x (torch.Tensor): The input qudit state.

        Returns:
            torch.Tensor: The resulting state after applying the circuit.
        """
        return self.circuit(x)



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
                    M = eye(dim=self.dim, device=x.device, sparse=self.sparse)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    for n in range(self.dim):
                        new_tuple = torch.tensor([[n], [n]], device=indices.device)
                        is_present = ((indices == new_tuple).all(dim=0)).any()
                        if not is_present:
                            indices = torch.cat((indices, new_tuple), dim=1)
                            values = torch.cat((values, torch.tensor([1], dtype=values.dtype, device=values.device)))
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = kron(U, M, sparse=self.sparse)
            else:
                U = kron(U, eye(self.dim, device=x.device, sparse=self.sparse), sparse=self.sparse)

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
        U = eye(1, device=x.device, sparse=self.sparse)

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
                    M = eye(dim=self.dim, device=x.device, sparse=self.sparse)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    for n in range(self.dim):
                        new_tuple = torch.tensor([[n], [n]], device=indices.device)
                        is_present = ((indices == new_tuple).all(dim=0)).any()
                        if not is_present:
                            indices = torch.cat((indices, new_tuple), dim=1)
                            values = torch.cat((values, torch.tensor([1], dtype=values.dtype, device=values.device)))
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

                U = kron(U, M, sparse=self.sparse)
            else:
                U = kron(U, eye(self.dim, device=x.device, sparse=self.sparse), sparse=self.sparse)

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
        U = eye(1, device=x.device, sparse=self.sparse)
        
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
                    M = eye(dim=self.dim, device=x.device, sparse=self.sparse)
                    M.index_put_(tuple(indices), values)
                elif self.sparse is True:
                    for n in range(self.dim):
                        new_tuple = torch.tensor([[n], [n]], device=indices.device)
                        is_present = ((indices == new_tuple).all(dim=0)).any()
                        if not is_present:
                            indices = torch.cat((indices, new_tuple), dim=1)
                            values = torch.cat((values, torch.tensor([1], dtype=values.dtype, device=values.device)))
                    M = torch.sparse_coo_tensor(indices, values, (self.dim, self.dim), device=x.device)

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
        omega = np.exp(2*1j*pi/dim)

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
                M[j][i] = torch.matmul(base(dim)[j].T, base(dim)[(i + s) % dim])
        
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
                M[j][i] = (omega ** (j * s)) * delta(i, j)  # Apply phase shift using delta function

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
        X = XGate(s=s, device=device).M  # Generalized Pauli-X gate
        Z = ZGate(device=device).M       # Generalized Pauli-Z gate
        M = torch.matmul(Z, X) / 1j      # Y = Z * X / 1j

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
            U = CNOT_sparse(index[0], index[1], dim, wires, device=device)

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
            localr = dec2den(k, wires, dim)  # Convert from decimal to local qudit representation
            locall = localr.copy()
            locall[c] = localr[t]  # Swap qudits
            locall[t] = localr[c]  # Swap qudits
            globall = den2dec(locall, dim)  # Convert back to decimal
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
                    u = torch.kron(u, base(dim)[d] @ base(dim)[d].T)
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
        self.angle = nn.Parameter(pi*torch.randn(1, device=device))
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
            local = dec2den(m, self.wires-1, self.dim)
            if self.wires == 2:
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

            for l in range(self.dim):
                if l != self.j-1 and l != self.k-1:
                    listl = local.copy()
                    listl.insert(t, l)
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
        self.angle = nn.Parameter(pi*torch.randn(1, device=device))
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
            intj = den2dec(listj, self.dim)
            listk = local.copy()
            listk.insert(t, k-1)
            intk = den2dec(listk, self.dim)

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
        self.angle = nn.Parameter(pi*torch.randn(1, device=device))
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
            local = dec2den(m, self.wires-1, self.dim)
            if self.wires == 2:
                loc = local[0]
            else:
                loc = local[c]
            angle = ((loc * self.angle) / 2) * np.sqrt(2 / (self.j * (self.j + 1)))

            for k in range(self.dim):
                listk = local.copy()
                listk.insert(t, k)  # insert k in position t of the list
                intk = den2dec(listk, self.dim)  # integer for the k state
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
        self.U = nn.Parameter(eye(dim=dim**wires, sparse=False, device=device) + torch.randn((dim**wires, dim**wires), device=device) + 1j*torch.randn((dim**wires, dim**wires), device=device))
    
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
        self.M = nn.Parameter(eye(dim=dim, sparse=False, device=device) + torch.randn((dim, dim), device=device) + 1j*torch.randn((dim, dim), device=device))
    
    def forward(self, x):

        M = self.M - torch.conj(self.M.T)
        M = torch.matrix_exp(M)

        U = 0.0
        for d in range(self.dim):
            u = torch.eye(1, device=x.device, dtype=torch.complex64)
            for i in range(self.wires):
                if i == self.index[0]:
                    u = torch.kron(u, base(self.dim, device=x.device)[d] @ base(self.dim, device=x.device)[d].T)
                elif i == self.index[1] and d in self.control_state:
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, torch.eye(self.dim, device=x.device, dtype=torch.complex64))
            U += u

        return U @ x