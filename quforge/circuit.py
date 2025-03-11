import torch.nn as nn
import quforge.gates as gates

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
        self.add_gate(gates.H(dim=self.dim, device=self.device, **kwargs))

    def RX(self, **kwargs):
        self.add_gate(gates.RX(dim=self.dim, wires=self.wires, device=self.device, sparse=self.sparse, **kwargs))

    def RY(self, **kwargs):
        self.add_gate(gates.RY(dim=self.dim, wires=self.wires, device=self.device, sparse=self.sparse, **kwargs))

    def RZ(self, **kwargs):
        self.add_gate(gates.RZ(dim=self.dim, wires=self.wires, device=self.device, sparse=self.sparse, **kwargs))

    def X(self, **kwargs):
        self.add_gate(gates.X(dim=self.dim, device=self.device, **kwargs))

    def Y(self, **kwargs):
        self.add_gate(gates.Y(dim=self.dim, device=self.device, **kwargs))

    def Z(self, **kwargs):
        self.add_gate(gates.Z(dim=self.dim, device=self.device, **kwargs))

    def CNOT(self, **kwargs):
        self.add_gate(gates.CNOT(dim=self.dim, wires=self.wires, sparse=self.sparse, device=self.device, **kwargs))

    def SWAP(self, **kwargs):
        self.add_gate(gates.SWAP(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def CZ(self, **kwargs):
        self.add_gate(gates.CZ(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def CCNOT(self, **kwargs):
        self.add_gate(gates.CCNOT(dim=self.dim, device=self.device, **kwargs))

    def MCX(self, **kwargs):
        self.add_gate(gates.MCX(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def CRX(self, **kwargs):
        self.add_gate(gates.CRX(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def CRY(self, **kwargs):
        self.add_gate(gates.CRY(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def CRZ(self, **kwargs):
        self.add_gate(gates.CRZ(dim=self.dim, device=self.device, sparse=self.sparse, wires=self.wires, **kwargs))

    def U(self, **kwargs):
        self.add_gate(gates.U(dim=self.dim, device=self.device, wires=self.wires, **kwargs))

    def CU(self, **kwargs):
        self.add_gate(gates.CU(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def Custom(self, **kwargs):
        self.add_gate(gates.CustomGate(dim=self.dim, wires=self.wires, device=self.device, **kwargs))

    def forward(self, x):
        """
        Apply the circuit to the input qudit state.

        Args:
            x (torch.Tensor): The input qudit state.

        Returns:
            torch.Tensor: The resulting state after applying the circuit.
        """
        return self.circuit(x)