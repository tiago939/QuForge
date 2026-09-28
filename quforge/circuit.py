import torch.nn as nn
import quforge.gates as gates
from quforge.trajectory import (
    get_gate_targets,
    apply_gate_batched,
    apply_channel_batched,
    noisy_measure,
    trajectory_expectation,
)


class Circuit(nn.Module):
    r"""
    Quantum Circuit for qudits.

    The Circuit class allows users to dynamically add various quantum gates to construct a quantum
    circuit for qudit systems. It supports a wide range of gates, including single-qudit,
    multi-qudit, and custom gates. The circuit is represented as a sequence of quantum operations
    (gates) that act on qudit states.

    **Arguments:**
        dim (int or list of int): The dimension of the qudits. If an integer, all qudits are assumed
            to have that dimension; if a list is provided, each element specifies the dimension of
            the corresponding qudit. wires (int): The total number of qudits (wires) in the circuit
            (used when `dim` is an integer). If `dim` is a list, wires is taken as the length of
            that list.
        device (str): The device to perform the computations on. Default is 'cpu'.
        sparse (bool): Whether to use sparse matrix representations for the gates. Default is False.

    **Attributes:**
        dim (int or list of int): The dimension(s) of the qudits.
        wires (int): The number of qudits in the circuit.
        device (str): The device for computations ('cpu' or 'cuda').
        circuit (nn.Sequential): A sequential container for holding the quantum gates.
        sparse (bool): Whether to use sparse matrices in the gates.

    **Methods:**
        add(module, **kwargs): Dynamically add a gate module to the circuit.
        add_gate(gate, **kwargs): Add a specific gate instance to the circuit.
        H(**kwargs): Add a Hadamard gate to the circuit.
        X(**kwargs): Add a Pauli-X gate to the circuit.
        Y(**kwargs): Add a Pauli-Y gate to the circuit.
        Z(**kwargs): Add a Pauli-Z gate to the circuit.
        RX(**kwargs): Add a rotation-X gate to the circuit.
        RY(**kwargs): Add a rotation-Y gate to the circuit.
        RZ(**kwargs): Add a rotation-Z gate to the circuit.
        CNOT(**kwargs): Add a controlled-NOT gate to the circuit.
        SWAP(**kwargs): Add a SWAP gate to the circuit.
        CZ(**kwargs): Add a controlled-Z gate to the circuit.
        CCNOT(**kwargs): Add a Toffoli (CCNOT) gate to the circuit.
        MCX(**kwargs): Add a multi-controlled-X gate to the circuit.
        CRX(**kwargs): Add a controlled rotation-X gate to the circuit.
        CRY(**kwargs): Add a controlled rotation-Y gate to the circuit.
        CRZ(**kwargs): Add a controlled rotation-Z gate to the circuit.
        U(**kwargs): Add a universal gate to the circuit.
        CU(**kwargs): Add a controlled-universal gate to the circuit.
        RXX(**kwargs): Add a rotation-XX gate to the circuit.
        RYY(**kwargs): Add a rotation-YY gate to the circuit.
        RZZ(**kwargs): Add a rotation-ZZ gate to the circuit.

    **Example:**
        >>> import quforge.quforge as qf
        >>> circuit = qf.Circuit(dim=[2,3,2], wires=3, device='cpu')  # Multidimensional: qudit0:2, qudit1:3, qudit2:2
        >>> circuit.H(index=[0])
        >>> circuit.CNOT(index=[0, 1])
        >>> state = qf.State('0-1-0', dim=[2,3,2])
        >>> result = circuit(state)
        >>> print(result)
    """

    def __init__(self, dim=2, wires=1, device="cpu", sparse=False,
                 noise_model=None, n_trajectories=1, use_vmap=True):
        super(Circuit, self).__init__()
        # Process dimensions: if dim is a list, use its length as wires.
        if isinstance(dim, int):
            self.dim = dim
            self.wires = wires
            self._dim_list = [dim] * wires
        else:
            self.dim = dim
            self.wires = len(dim)
            self._dim_list = list(dim)
        self.device = device
        self.circuit = nn.Sequential()
        self.sparse = sparse
        self.noise_model = noise_model
        self.n_trajectories = n_trajectories
        self.use_vmap = use_vmap

    def add(self, module, **kwargs):
        """
        Dynamically add a gate module to the circuit.

        Args:
            module: The gate module to add.
            **kwargs: Additional arguments for the gate.
        """
        gate = module(
            dim=self.dim,
            wires=self.wires,
            device=self.device,
            sparse=self.sparse,
            **kwargs
        )
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
        self.add_gate(
            gates.H(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def X(self, **kwargs):
        self.add_gate(
            gates.X(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def Y(self, **kwargs):
        self.add_gate(
            gates.Y(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def Z(self, **kwargs):
        self.add_gate(
            gates.Z(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def RX(self, **kwargs):
        self.add_gate(
            gates.RX(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def RY(self, **kwargs):
        self.add_gate(
            gates.RY(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def RZ(self, **kwargs):
        self.add_gate(
            gates.RZ(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def CNOT(self, **kwargs):
        self.add_gate(
            gates.CNOT(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def SWAP(self, **kwargs):
        self.add_gate(
            gates.SWAP(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def CZ(self, **kwargs):
        self.add_gate(
            gates.CZ(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def CCNOT(self, **kwargs):
        self.add_gate(
            gates.CCNOT(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def MCX(self, **kwargs):
        self.add_gate(
            gates.MCX(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def CRX(self, **kwargs):
        self.add_gate(
            gates.CRX(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def CRY(self, **kwargs):
        self.add_gate(
            gates.CRY(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def CRZ(self, **kwargs):
        self.add_gate(
            gates.CRZ(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def U(self, **kwargs):
        self.add_gate(
            gates.U(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def LowRankU(self, **kwargs):
            self.add_gate(
                gates.LowRankU(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
            )

    def RankU(self, **kwargs):
                self.add_gate(
                    gates.RankU(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
                )

    def CU(self, **kwargs):
        self.add_gate(
            gates.CU(dim=self.dim, wires=self.wires, device=self.device, **kwargs)
        )

    def RXX(self, **kwargs):
        self.add_gate(
            gates.RXX(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def RYY(self, **kwargs):
        self.add_gate(
            gates.RYY(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def RZZ(self, **kwargs):
        self.add_gate(
            gates.RZZ(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def W(self, **kwargs):
        self.add_gate(
            gates.W(
                dim=self.dim,
                wires=self.wires,
                device=self.device,
                sparse=self.sparse,
                **kwargs
            )
        )

    def forward(self, x):
        """
        Apply the circuit to the input qudit state.

        When ``noise_model`` is ``None`` the original noiseless path is
        used.

        Otherwise the state is expanded into a batch of trajectories and
        stochastic Kraus operators are applied after each gate according
        to the noise model.

        Args:
            x (torch.Tensor): The input qudit state, shape ``(D, 1)``.

        Returns:
            torch.Tensor:
                * Noiseless: ``(D, 1)`` — the final state.
                * Noisy: ``(n_trajectories, D, 1)`` — one state per
                  trajectory.  Use ``trajectory_expectation`` or
                  ``noisy_measure`` to extract observables.
        """
        # ---- original noiseless path ----
        if self.noise_model is None:
            return self.circuit(x)

        # ---- noisy trajectory path ----
        n_traj = max(self.n_trajectories, 1)
        psi = x.unsqueeze(0).expand(n_traj, -1, -1).clone()

        for gate in self.circuit:
            # apply gate across trajectories
            psi = apply_gate_batched(gate, psi, use_vmap=self.use_vmap)

            # look up noise channels for this gate
            gate_name = type(gate).__name__
            target_qudits = get_gate_targets(gate)
            errors = self.noise_model.get_errors(gate_name, target_qudits)

            for channel in errors:
                psi = apply_channel_batched(psi, channel, target_qudits, self._dim_list)

        return psi