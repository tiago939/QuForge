from math import log as log
import itertools
import numpy as np
import quforge.aux as aux
import torch
import torch.nn as nn


class H(nn.Module):
    r"""
    Hadamard Gate for qudits.

    The Hadamard gate creates a superposition of states in a qudit system.

    **Matrix Representation:**

    For a qudit of dimension D, the Hadamard gate is defined as

    .. math::

            H = \frac{1}{\sqrt{D}}
            \begin{pmatrix}
            1 & 1 & \cdots & 1 \\\\
            1 & \omega & \cdots & \omega^{D-1} \\\\
            1 & \omega^2 & \cdots & \omega^{2(D-1)} \\\\
            \vdots & \vdots & \ddots & \vdots \\\\
            1 & \omega^{D-1} & \cdots & \omega^{(D-1)(D-1)}
            \end{pmatrix}

    where 

    .. math::
          \omega = \exp\left(\frac{2\pi i}{D}\right).

    **Arguments:**
        dim (int or list of int): 
            The dimension of the qudit(s). If an integer is provided, all qudits are assumed to have that dimension.
            If a list is provided, each element corresponds to the dimension of the corresponding qudit.
        index (list of int): 
            The indices of the qudits to which the gate is applied.
        wires (int): The total number of qudits in the circuit. Default is 1.
        inverse (bool): 
            Whether to apply the inverse of the Hadamard gate. Default is False.
        device (str): 
            The device on which computations are performed. Default is 'cpu'.

    **Attributes:**
        index (list of int): The indices of the qudits on which the gate is applied.
        device (str): The device on which computations are performed.
        dim (int or list of int): The qudit dimension(s).
        M_dict (dict): A dictionary mapping each target qudit index to its Hadamard matrix.

    **Examples:**
        >>> # Uniform qudit dimensions (e.g., 3 qubits):
        >>> gate = qf.H(dim=2, index=[0])
        >>> psi = qf.State('0-1-0', dim=2)  # three-qubit state
        >>> result = gate(psi)
        >>> print(result)

        >>> # Multidimensional qudits (e.g., first qubit is a qubit and second is a qutrit):
        >>> gate = qf.H(dim=[2,3], index=[1], device='cpu')
        >>> psi = qf.State('1-2', dim=[2,3])
        >>> result = gate(psi)
        >>> print(result)
    """

    def __init__(self, dim=2, index=[0], wires=1, inverse=False, device="cpu"):
        super(H, self).__init__()
        self.index = index
        self.device = device
        self.inverse = inverse
        self.wires = wires

        # Build Hadamard matrices for target qudits.
        self.M_dict = {}
        if isinstance(dim, int):
            # Uniform dimension for all qudits.
            self.dim = dim
            omega = np.exp(2 * 1j * np.pi / dim)
            # Build the Hadamard matrix for dimension "dim".
            M = torch.ones((dim, dim), dtype=torch.complex64, device=device)
            for i in range(1, dim):
                for j in range(1, dim):
                    M[i, j] = omega ** (i * j)
            M = M / (dim**0.5)
            if inverse:
                M = torch.conj(M).T.contiguous()
            # For each target qudit, store the same matrix.
            for idx in self.index:
                self.M_dict[idx] = M
        else:
            # Multidimensional qudits: dim is a list.
            self.dim = dim  # list of dimensions, one per qudit.
            for idx in self.index:
                d = dim[idx]
                omega = np.exp(2 * 1j * np.pi / d)
                M = torch.ones((d, d), dtype=torch.complex64, device=device)
                for i in range(1, d):
                    for j in range(1, d):
                        M[i, j] = omega ** (i * j)
                M = M / (d**0.5)
                if inverse:
                    M = torch.conj(M).T.contiguous()
                self.M_dict[idx] = M

    def forward(self, x):
        """
        Apply the Hadamard gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor. It is assumed that x is a column vector
                              whose dimension is the product of the individual qudit dimensions.

        Returns:
            torch.Tensor: The resulting state after applying the Hadamard gate.
        """
        # Determine the total number of wires.
        if isinstance(self.dim, int):
            wires = int(round(np.log(x.shape[0]) / np.log(self.dim)))
        else:
            wires = len(self.dim)

        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(wires):
            if i in self.index:
                U = torch.kron(U, self.M_dict[i])
            else:
                # Use identity of the appropriate dimension.
                if isinstance(self.dim, int):
                    d = self.dim
                else:
                    d = self.dim[i]
                U = torch.kron(U, torch.eye(d, dtype=torch.complex64, device=x.device))
        return torch.matmul(U, x)

    def matrix(self):
        """
        Construct and return the overall unitary matrix corresponding to the Hadamard gate.

        Returns:
            torch.Tensor: The complete unitary operator representing the Hadamard gate applied
                        on the specified qudits, with dimensions matching the product of individual qudit dimensions.
        """

        wires = self.wires
        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        for i in range(wires):
            if i in self.index:
                U = torch.kron(U, self.M_dict[i])
            else:
                if isinstance(self.dim, int):
                    d = self.dim
                else:
                    d = self.dim[i]
                U = torch.kron(
                    U, torch.eye(d, dtype=torch.complex64, device=self.device)
                )

        return U


class X(nn.Module):
    r"""
    Generalized Pauli-X (X) Gate for qudits.

    The X gate represents a cyclic shift of the computational basis states in a qudit system,
    generalizing the Pauli-X gate from qubits to higher dimensions.

    **Matrix Representation:**

    For a qubit (2-level qudit), the Pauli-X gate is represented as:

    .. math::

            X = 
            \begin{pmatrix}
            0 & 1 \\
            1 & 0
            \end{pmatrix}

    For a higher-dimensional qudit, the X gate shifts the basis states by a cyclic parameter \( s \)
    as follows. Given a qudit of dimension \( D \), its matrix elements are defined by

    .. math::

            X_{j,i} = \langle j |X| i \rangle = \langle j | (i+s) \rangle,

    where the addition is modulo \( D \).

    **Arguments:**
        s (int): 
            The cyclic shift parameter for the qudit. Default is 1.
        dim (int or list of int): 
            The dimension of the qudit. If an integer, all qudits are assumed to have that dimension.
            If a list is provided, each element corresponds to the dimension of the respective qudit.
        wires (int): The total number of qudits in the circuit. Default is 1.
        index (list of int): 
            The indices of the qudits to which the gate is applied.
        device (str): 
            The device on which the computations are performed. Default is 'cpu'.
        inverse (bool): 
            Whether to apply the inverse of the X gate. Default is False.

    **Attributes:**
        index (list of int): 
            The indices of the qudits to which the gate is applied.
        dim (int or list of int): 
            The dimension(s) of the qudit(s).
        M or M_dict (torch.Tensor or dict): 
            The matrix representation of the X gate (for uniform dimensions) or a dictionary mapping
            each target qudit index to its corresponding X gate (for multidimensional qudits).
        inverse (bool): 
            Whether the matrix representation is inverted.

    **Examples:**
        >>> # Uniform qudit dimensions (e.g., 3 qubits):
        >>> gate = qf.X(s=1, dim=2, index=[0])
        >>> psi = qf.State('0-1-0', dim=2)  # three-qubit state
        >>> result = gate(psi)
        >>> print(result)

        >>> # Multidimensional qudits (e.g., first qudit is a qubit and second is a qutrit):
        >>> gate = qf.X(s=1, dim=[2,3], index=[1], device='cpu')
        >>> psi = qf.State('0-2', dim=[2,3])
        >>> result = gate(psi)
        >>> print(result)
    """

    def __init__(self, s=1, dim=2, wires=1, index=[0], device="cpu", inverse=False):
        super(X, self).__init__()
        self.index = index
        self.inverse = inverse
        self.device = device
        self.wires = wires

        # Build the X gate(s) for target qudits.
        if isinstance(dim, int):
            self.dim = dim  # Uniform dimension for all qudits.
            M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
            # Use the aux.base function to get basis vectors.
            basis = aux.base(dim, device=device)
            for i in range(dim):
                for j in range(dim):
                    # The matrix element is given by the overlap between basis state j and basis state (i+s)%dim.
                    # Here we mimic this by taking the inner product of the corresponding basis vectors.
                    # Since basis vectors are one-hot, the product is 1 if j equals (i+s)%dim, 0 otherwise.
                    M[j, i] = 1.0 if j == ((i + s) % dim) else 0.0
            if inverse:
                M = torch.conj(M.T)
            # In the uniform case, store the same M for each target qudit.
            self.register_buffer("M", M)
        else:
            # Multidimensional: dim is a list.
            self.dim = dim  # list of dimensions, one per qudit.
            self.M_dict = {}
            for idx in self.index:
                d = dim[idx]
                M = torch.zeros((d, d), dtype=torch.complex64, device=device)
                basis = aux.base(d, device=device)
                for i in range(d):
                    for j in range(d):
                        M[j, i] = 1.0 if j == ((i + s) % d) else 0.0
                if inverse:
                    M = torch.conj(M.T)
                self.M_dict[idx] = M

    def forward(self, x):
        """
        Apply the X gate to the qudit state.

        Args:
            x (torch.Tensor):
                The input state tensor of the qudit. It is assumed to be a column vector
                whose dimension is the product of the individual qudit dimensions.

        Returns:
            torch.Tensor: The resulting state after applying the X gate.
        """
        # Determine the number of qudits.
        if isinstance(self.dim, int):
            wires = int(round(np.log(x.shape[0]) / np.log(self.dim)))
        else:
            wires = len(self.dim)

        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(wires):
            if i in self.index:
                # For uniform qudits, use self.M; for multidimensional, look up the correct M.
                if isinstance(self.dim, int):
                    U = torch.kron(U, self.M)
                else:
                    U = torch.kron(U, self.M_dict[i])
            else:
                # Use identity of appropriate dimension.
                d = self.dim if isinstance(self.dim, int) else self.dim[i]
                U = torch.kron(U, torch.eye(d, dtype=torch.complex64, device=x.device))
        return torch.matmul(U, x)

    def matrix(self):
        """
        Construct and return the overall unitary matrix corresponding to the X gate.

        Returns:
            torch.Tensor: The complete unitary operator representing the X gate applied
                        on the specified qudits, with dimensions matching the product of individual qudit dimensions.
        """

        wires = self.wires
        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        for i in range(wires):
            if i in self.index:
                if isinstance(self.dim, int):
                    U = torch.kron(U, self.M)
                else:
                    U = torch.kron(U, self.M_dict[i])
            else:
                d = self.dim if isinstance(self.dim, int) else self.dim[i]
                U = torch.kron(
                    U, torch.eye(d, dtype=torch.complex64, device=self.device)
                )

        return U


class Z(nn.Module):
    r"""
    Generalized Pauli-Z (Z) Gate for qudits.

    The Z gate applies a phase shift to the computational basis states in a qudit system,
    generalizing the Pauli-Z gate from qubits to higher dimensions.

    **Matrix Representation:**

    For a qubit (2-level qudit), the Pauli-Z gate is represented as:

    .. math::

            Z = 
            \begin{pmatrix}
            1 & 0 \\
            0 & -1
            \end{pmatrix}

    For a higher-dimensional qudit, the Z gate applies a phase shift based on a complex
    exponential. For a qudit of dimension \( D \) with phase parameter \( s \), the matrix is:

    .. math::

            Z_s = \text{diag}(1, \omega^s, \omega^{2s}, \ldots, \omega^{(D-1)s})

    where \( \omega = \exp(2\pi i / D) \).

    **Arguments:**
        dim (int or list of int): 
            The dimension of the qudit. If an integer, all qudits are assumed to have that dimension.
            If a list is provided, each element corresponds to the dimension of the respective qudit.
        wires (int): The total number of qudits in the circuit. Default is 1.
        s (int): 
            The phase shift parameter for the qudit. Default is 1.
        index (list of int): 
            The indices of the qudits to which the gate is applied.
        device (str): 
            The device on which computations are performed. Default is 'cpu'.
        inverse (bool): 
            Whether to apply the inverse of the Z gate. Default is False.

    **Attributes:**
        index (list of int): 
            The indices of the qudits to which the gate is applied.
        dim (int or list of int): 
            The dimension(s) of the qudit(s).
        M or M_dict (torch.Tensor or dict): 
            For uniform dimensions, M holds the matrix representation of the Z gate.
            For multidimensional qudits, M_dict is a dictionary mapping each target qudit index
            to its corresponding Z gate matrix.
        inverse (bool): 
            Whether the matrix representation is inverted.

    **Examples:**
        >>> # Uniform qudit dimensions (e.g., 3 qubits):
        >>> gate = qf.Z(dim=2, s=1, index=[0])
        >>> psi = qf.State('0-1-0', dim=2)  # three-qubit state
        >>> result = gate(psi)
        >>> print(result)

        >>> # Multidimensional qudits (e.g., first qudit is a qubit and second is a qutrit):
        >>> gate = qf.Z(dim=[2,3], s=1, index=[1], device='cpu')
        >>> psi = qf.State('0-2', dim=[2,3])
        >>> result = gate(psi)
        >>> print(result)
    """

    def __init__(self, dim=2, wires=1, s=1, index=[0], device="cpu", inverse=False):
        super(Z, self).__init__()
        self.index = index
        self.inverse = inverse
        self.device = device
        self.wires = wires

        # When dim is a uniform integer.
        if isinstance(dim, int):
            self.dim = dim
            omega = np.exp(2 * 1j * np.pi / dim)
            # Build the Z gate matrix.
            M = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
            for i in range(dim):
                for j in range(dim):
                    # Only diagonal elements are nonzero.
                    M[j, i] = (omega ** (j * s)) * aux.delta(i, j)
            if inverse:
                M = torch.conj(M.T)
            self.register_buffer("M", M)
        else:
            # Multidimensional qudits: dim is a list.
            self.dim = dim  # list of dimensions, one per qudit.
            self.M_dict = {}
            for idx in self.index:
                d = dim[idx]
                omega = np.exp(2 * 1j * np.pi / d)
                M = torch.zeros((d, d), dtype=torch.complex64, device=device)
                for i in range(d):
                    for j in range(d):
                        M[j, i] = (omega ** (j * s)) * aux.delta(i, j)
                if inverse:
                    M = torch.conj(M.T)
                self.M_dict[idx] = M

    def forward(self, x):
        """
        Apply the Z gate to the qudit state.

        Args:
            x (torch.Tensor):
                The input state tensor of the qudit. It is assumed to be a column vector whose dimension
                is the product of the individual qudit dimensions.

        Returns:
            torch.Tensor: The resulting state after applying the Z gate.
        """
        # Determine the number of qudits (wires).
        if isinstance(self.dim, int):
            wires = int(round(np.log(x.shape[0]) / np.log(self.dim)))
        else:
            wires = len(self.dim)

        U = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(wires):
            if i in self.index:
                if isinstance(self.dim, int):
                    U = torch.kron(U, self.M)
                else:
                    U = torch.kron(U, self.M_dict[i])
            else:
                # Identity for non-target qudits.
                d = self.dim if isinstance(self.dim, int) else self.dim[i]
                U = torch.kron(U, torch.eye(d, dtype=torch.complex64, device=x.device))
        return torch.matmul(U, x)

    def matrix(self):
        """
        Construct and return the overall unitary matrix corresponding to the Z gate.

        Returns:
            torch.Tensor: The complete unitary operator representing the Z gate applied
                        on the specified qudits, with dimensions matching the product of individual qudit dimensions.
        """

        wires = self.wires
        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        for i in range(wires):
            if i in self.index:
                if isinstance(self.dim, int):
                    U = torch.kron(U, self.M)
                else:
                    U = torch.kron(U, self.M_dict[i])
            else:
                d = self.dim if isinstance(self.dim, int) else self.dim[i]
                U = torch.kron(
                    U, torch.eye(d, dtype=torch.complex64, device=self.device)
                )

        return U


class Y(nn.Module):
    r"""
    Generalized Pauli-Y (Y) Gate for qudits.

    The Y gate represents a combination of the X and Z gates, generalizing the Pauli-Y gate
    from qubits to higher dimensions. It is defined as

    .. math::

            Y = \frac{1}{i}\, Z \cdot X,

    where the generalized Pauli-X and Pauli-Z gates are applied to the target qudits.

    **Arguments:**
        s (int):
            The cyclic shift parameter for the X gate. Default is 1.
        dim (int or list of int):
            The dimension of the qudit. If an integer, all qudits are assumed to have that dimension.
            If a list is provided, each element corresponds to the dimension of the respective qudit.
        wires (int): The total number of qudits in the circuit. Default is 1.
        index (list of int):
            The indices of the qudits to which the gate is applied.
        device (str):
            The device on which the computations are performed. Default is 'cpu'.

    **Attributes:**
        index (list of int):
            The indices of the qudits on which the gate is applied.
        dim (int or list of int):
            The qudit dimension(s).
        M (torch.Tensor) or M_dict (dict):
            For uniform dimensions, M stores the matrix representation of the Y gate.
            For multidimensional qudits, M_dict maps each target qudit index to its corresponding Y gate matrix.

    **Examples:**
        >>> # Uniform qudit dimensions (e.g., 3 qubits):
        >>> gate = qf.Y(dim=2, index=[0])
        >>> psi = qf.State('0-1-0', dim=2)  # three-qubit state
        >>> result = gate(psi)
        >>> print(result)

        >>> # Multidimensional qudits (e.g., first qudit is a qubit and second is a qutrit):
        >>> gate = qf.Y(dim=[2,3], index=[1], device='cpu')
        >>> psi = qf.State('0-2', dim=[2,3])
        >>> result = gate(psi)
        >>> print(result)
    """

    def __init__(self, s=1, dim=2, wires=1, index=[0], device="cpu"):
        super(Y, self).__init__()
        self.index = index
        self.dim = dim
        self.wires = wires
        self.device = device
        self.s = s

        # Construct the X and Z gate objects.
        # These classes are assumed to have been updated to support multidimensional qudits.
        self.x_gate = X(s=s, dim=dim, index=index, device=device)
        self.z_gate = Z(dim=dim, s=s, index=index, device=device)

        # Compute the Y gate as Y = Z * X / (1j).
        if isinstance(dim, int):
            # Uniform qudit dimensions.
            M = torch.matmul(self.z_gate.M, self.x_gate.M) / 1j
            self.register_buffer("M", M)
        else:
            # Multidimensional qudits: build a dictionary of Y matrices for target qudits.
            self.M_dict = {}
            for i in self.index:
                M = torch.matmul(self.z_gate.M_dict[i], self.x_gate.M_dict[i]) / 1j
                self.M_dict[i] = M

    def forward(self, x):
        """
        Apply the Y gate to the qudit state.

        Args:
            x (torch.Tensor):
                The input state tensor of the qudit. It is assumed to be a column vector whose dimension
                is the product of the individual qudit dimensions.

        Returns:
            torch.Tensor: The resulting state after applying the Y gate.
        """
        # Determine the number of qudits (wires).
        if isinstance(self.dim, int):
            wires = int(round(np.log(x.shape[0]) / np.log(self.dim)))
            U = torch.eye(1, dtype=torch.complex64, device=x.device)
            for i in range(wires):
                if i in self.index:
                    U = torch.kron(U, self.M)
                else:
                    U = torch.kron(
                        U, torch.eye(self.dim, dtype=torch.complex64, device=x.device)
                    )
        else:
            wires = len(self.dim)
            U = torch.eye(1, dtype=torch.complex64, device=x.device)
            for i in range(wires):
                if i in self.index:
                    U = torch.kron(U, self.M_dict[i])
                else:
                    U = torch.kron(
                        U,
                        torch.eye(self.dim[i], dtype=torch.complex64, device=x.device),
                    )
        return torch.matmul(U, x)

    def matrix(self):
        """
        Construct and return the overall unitary matrix corresponding to the Y gate.

        Returns:
            torch.Tensor: The complete unitary operator representing the Y gate applied
                        on the specified qudits, with dimensions matching the product of individual qudit dimensions.
        """

        wires = self.wires
        if isinstance(self.dim, int):
            U = torch.eye(1, dtype=torch.complex64, device=self.device)
            for i in range(wires):
                if i in self.index:
                    U = torch.kron(U, self.M)
                else:
                    U = torch.kron(
                        U,
                        torch.eye(self.dim, dtype=torch.complex64, device=self.device),
                    )
        else:
            U = torch.eye(1, dtype=torch.complex64, device=self.device)
            for i in range(wires):
                if i in self.index:
                    U = torch.kron(U, self.M_dict[i])
                else:
                    U = torch.kron(
                        U,
                        torch.eye(
                            self.dim[i], dtype=torch.complex64, device=self.device
                        ),
                    )

        return U


class RX(nn.Module):
    r"""
    Rotation-X (RX) Gate for qudits.

    The RX gate represents a rotation about the X-axis of the Bloch sphere in a qudit system.
    For a qubit (2-level system), the matrix representation is given by

    .. math::

            RX(\theta) = 
            \begin{pmatrix}
            \cos(\theta/2) & -i\sin(\theta/2) \\
            -i\sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    For higher-dimensional qudits, the RX gate affects only the specified two levels (indexed by
    \(j\) and \(k\)), leaving all other levels unchanged.

    **Arguments:**
        j (int or list of int): 
            The first level to rotate between. If an integer is provided, it is applied to all target qudits.
        k (int or list of int): 
            The second level to rotate between. If an integer is provided, it is applied to all target qudits.
        index (list of int): 
            The indices of the qudits to which the RX gate is applied.
        dim (int or list of int): 
            The dimension of the qudit. If an integer is provided, all qudits are assumed to have that dimension.
            If a list is provided, each element specifies the dimension of the corresponding qudit.
        wires (int): 
            The total number of qudits in the circuit. (Used when `dim` is an integer.)
        device (str): 
            The device on which computations are performed. Default is 'cpu'.
        angle (float or torch.Tensor or None): 
            The rotation angle. If None, create a random parameter, if float or torch.Tensor, use it directly.
        sparse (bool): 
            Whether to use a sparse matrix representation. Default is False.

    **Attributes:**
        j, k: The target levels for rotation (stored per qudit in a mapping if needed).
        index (list of int): The indices of the qudits to which the gate is applied.
        angle (torch.nn.Parameter or torch.Tensor): The rotation angle(s) for each qudit.
        sparse (bool): Whether the matrix representation is sparse.
        dim (int or list of int): The dimension(s) of the qudit(s).

    **Examples:**
        >>> # Single dimensional qudit (qubit) case:
        >>> gate = qf.RX(index=[0])
        >>> state = qf.State('0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: for example, first qudit is a qubit and second is a qutrit.
        >>> # Here, we want to rotate between levels 1 and 0 for the first qudit and between levels 1 and 2 for the second.
        >>> gate = qf.RX(index=[0,1], dim=[2,3], j=[1,1], k=[0,2], device='cpu')
        >>> state = qf.State('1-2', dim=[2,3])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(
        self,
        j=0,
        k=1,
        index=[0],
        dim=2,
        wires=1,
        device="cpu",
        angle=None,
        sparse=False,
    ):
        super(RX, self).__init__()
        self.device = device
        self.sparse = sparse
        self.index = index
        self.wires = wires

        # Process dimensions.
        if isinstance(dim, int):
            self.dim = dim
            num_wires = wires
        else:
            self.dim = dim  # a list of dimensions
            num_wires = len(dim)

        # Process the target levels j and k.
        # If they are provided as integers, apply them to all target qudits.
        if isinstance(j, int):
            self.j_map = {t: j for t in self.index}
        else:
            if len(j) != len(self.index):
                raise ValueError(
                    "If 'j' is provided as a list, its length must equal the number of target qudits in 'index'."
                )
            self.j_map = {t: j_val for t, j_val in zip(self.index, j)}
        if isinstance(k, int):
            self.k_map = {t: k for t in self.index}
        else:
            if len(k) != len(self.index):
                raise ValueError(
                    "If 'k' is provided as a list, its length must equal the number of target qudits in 'index'."
                )
            self.k_map = {t: k_val for t, k_val in zip(self.index, k)}

        # Initialize the rotation angle parameter.
        # The angle parameter is defined for each qudit (wire).
        if angle is None:
            self.angle = nn.Parameter(torch.randn(num_wires, device=device))
        elif isinstance(angle, torch.Tensor):
            self.angle = angle
        else:
            self.angle = nn.Parameter(angle * torch.ones(num_wires, device=device))

    def forward(self, x, param=None):
        """
        Apply the RX gate to the qudit state.

        Args:
            x (torch.Tensor):
                The input state tensor (a column vector) whose dimension is the product of the individual
                qudit dimensions.
            param (torch.Tensor or bool):
                If None, use the internal angle parameter. If provided, use it as the rotation angle.

        Returns:
            torch.Tensor: The resulting state after applying the RX gate.
        """
        # Determine the number of qudits.
        if isinstance(self.dim, int):
            L = round(log(x.shape[0], self.dim))
        else:
            L = len(self.dim)

        # Start with a 1x1 identity operator.
        U = aux.eye(1, device=x.device, sparse=self.sparse)

        # Loop over each qudit.
        for i in range(L):
            # Determine the dimension d for qudit i.
            d = self.dim if isinstance(self.dim, int) else self.dim[i]
            # If this qudit is targeted for rotation.
            if i in self.index:
                # Retrieve the target levels for this qudit.
                j_val = self.j_map[i]
                k_val = self.k_map[i]
                # Check that the specified levels are valid.
                if j_val >= d or k_val >= d:
                    raise ValueError(
                        f"For qudit {i} with dimension {d}, the target levels j={j_val} and k={k_val} are out of range."
                    )
                # Build indices for updating the matrix:
                # We want to update:
                #   (j, j) and (k, k) -> cos(angle/2)
                #   (j, k) and (k, j) -> -i sin(angle/2)
                indices = torch.tensor(
                    [[j_val, k_val, j_val, k_val], [j_val, k_val, k_val, j_val]],
                    device=x.device,
                )
                values = torch.zeros(4, dtype=torch.complex64, device=x.device)
                # Use the provided parameter or the internal angle for qudit i.
                angle_val = self.angle[i] if param is None else torch.tensor(param[i], device=self.device)
                values[0] = torch.cos(angle_val / 2)
                values[1] = torch.cos(angle_val / 2)
                values[2] = -1j * torch.sin(angle_val / 2)
                values[3] = -1j * torch.sin(angle_val / 2)
                # Create a d x d identity matrix (dense or sparse as required).
                M = aux.eye(dim=d, device=x.device, sparse=self.sparse)
                # Update the entries corresponding to the rotation subspace.
                if self.sparse:
                    M = aux.sparse_index_put(M, indices, values, self.device)
                else:
                    M.index_put_(tuple(indices), values)
                # Incorporate this qudit's operation into the overall unitary.
                U = aux.kron(U, M, sparse=self.sparse)
            else:
                # For non-target qudits, use the identity matrix of appropriate dimension.
                M = aux.eye(
                    dim=(d if isinstance(self.dim, int) else self.dim[i]),
                    device=x.device,
                    sparse=self.sparse,
                )
                U = aux.kron(U, M, sparse=self.sparse)
        return U @ x

    def matrix(self, param=None):
        """
        Construct and return the overall unitary operator representing the RX gate applied
        to the specified qudits in the circuit.

        The rotation angle used for each target qudit is taken from the internal parameter `self.angle`
        unless an alternative angle is provided via the `param` argument.

        Args:
            param (list or bool):
                If None, the method uses the internal angle parameter for each qudit.
                If a list is provided, its elements override the corresponding entries in `self.angle`
                for the rotation angles.

        Returns:
            torch.Tensor: The complete unitary operator (as a torch.Tensor) representing the RX gate
                        applied to the qudit state.
        """

        L = self.wires
        U = aux.eye(1, device=self.device, sparse=self.sparse)

        for i in range(L):
            d = self.dim if isinstance(self.dim, int) else self.dim[i]
            if i in self.index:
                j_val = self.j_map[i]
                k_val = self.k_map[i]
                if j_val >= d or k_val >= d:
                    raise ValueError(
                        f"For qudit {i} with dimension {d}, the target levels j={j_val} and k={k_val} are out of range."
                    )
                indices = torch.tensor(
                    [[j_val, k_val, j_val, k_val], [j_val, k_val, k_val, j_val]],
                    device=self.device,
                )
                values = torch.zeros(4, dtype=torch.complex64, device=self.device)
                angle_val = (
                    self.angle[i]
                    if param is None
                    else torch.tensor(param[i], device=self.device)
                )
                values[0] = torch.cos(angle_val / 2)
                values[1] = torch.cos(angle_val / 2)
                values[2] = -1j * torch.sin(angle_val / 2)
                values[3] = -1j * torch.sin(angle_val / 2)
                M = aux.eye(dim=d, device=self.device, sparse=self.sparse)
                M.index_put_(tuple(indices), values)
                U = aux.kron(U, M, sparse=self.sparse)
            else:
                M = aux.eye(
                    dim=(d if isinstance(self.dim, int) else self.dim[i]),
                    device=self.device,
                    sparse=self.sparse,
                )
                U = aux.kron(U, M, sparse=self.sparse)

        return U


class RY(nn.Module):
    r"""
    Rotation-Y (RY) Gate for qudits.

    The RY gate represents a rotation around the Y-axis of the Bloch sphere in a qudit system.
    For a qubit (2-level system), the matrix representation is given by

    .. math::

            RY(\theta) = 
            \begin{pmatrix}
            \cos(\theta/2) & -\sin(\theta/2) \\
            \sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    For higher-dimensional qudits, the RY gate acts only on the specified two levels (indexed by
    \(j\) and \(k\)), leaving all other levels unchanged.

    **Arguments:**
        j (int or list of int): 
            The first level to rotate between. If an integer is provided, it is applied to all target qudits.
        k (int or list of int): 
            The second level to rotate between. If an integer is provided, it is applied to all target qudits.
        index (list of int): 
            The indices of the qudits to which the RY gate is applied.
        dim (int or list of int): 
            The dimension of the qudit. If an integer is provided, all qudits are assumed to have that dimension.
            If a list is provided, each element specifies the dimension of the corresponding qudit.
        wires (int): 
            The total number of qudits in the circuit (used when `dim` is an integer).
        device (str): 
            The device on which computations are performed. Default is 'cpu'.
        angle (float or torch.Tensor or None): 
            The rotation angle. If None, create a random parameter, if float or torch.Tensor, use it directly.
        sparse (bool): 
            Whether to use a sparse matrix representation. Default is False.

    **Attributes:**
        j, k: The target levels for rotation (stored per target qudit in mappings).
        index (list of int): The indices of the qudits to which the gate is applied.
        angle (torch.nn.Parameter): The rotation angle(s) for each qudit.
        sparse (bool): Whether the matrix representation is sparse.
        dim (int or list of int): The dimension(s) of the qudit(s).

    **Examples:**
        >>> # Single dimension qudit (qubit) case:
        >>> gate = RY(index=[0])
        >>> state = qf.State('0', dim=2)
        >>> result = qf.gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits:
        >>> # For example, first qudit is a qubit and second is a qutrit.
        >>> # Here, we want to rotate between levels 0 and 1 on the first qudit,
        >>> # and between levels 1 and 2 on the second qudit.
        >>> gate = qf.RY(index=[0,1], dim=[2,3], j=[0,1], k=[1,2], device='cpu')
        >>> state = qf.State('0-2', dim=[2,3])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(
        self,
        j=0,
        k=1,
        index=[0],
        dim=2,
        wires=1,
        device="cpu",
        angle=None,
        sparse=False,
    ):
        super(RY, self).__init__()
        self.device = device
        self.sparse = sparse
        self.index = index
        self.wires = wires

        # Process dimensions.
        if isinstance(dim, int):
            self.dim = dim
            num_wires = wires
        else:
            self.dim = dim  # a list of dimensions
            num_wires = len(dim)

        # Process target levels j and k.
        # Convert each provided value explicitly to an integer.
        if isinstance(j, int):
            self.j_map = {t: int(j) for t in self.index}
        else:
            if len(j) != len(self.index):
                raise ValueError(
                    "If 'j' is provided as a list, its length must equal the number of target qudits in 'index'."
                )
            self.j_map = {t: int(j_val) for t, j_val in zip(self.index, j)}
        if isinstance(k, int):
            self.k_map = {t: int(k) for t in self.index}
        else:
            if len(k) != len(self.index):
                raise ValueError(
                    "If 'k' is provided as a list, its length must equal the number of target qudits in 'index'."
                )
            self.k_map = {t: int(k_val) for t, k_val in zip(self.index, k)}

        # Initialize the rotation angle parameter.
        if angle is None:
            self.angle = nn.Parameter(torch.randn(num_wires, device=device))
        elif isinstance(angle, torch.Tensor):
            self.angle = angle
        else:
            self.angle = nn.Parameter(angle * torch.ones(num_wires, device=device))

    def forward(self, x, param=None):
        """
        Apply the RY gate to the qudit state.

        Args:
            x (torch.Tensor):
                The input state tensor (a column vector) whose dimension is the product of the individual
                qudit dimensions.
            param (torch.Tensor or bool):
                If None, use the internal angle parameter; if provided, use it as the rotation angle.

        Returns:
            torch.Tensor: The resulting state after applying the RY gate.
        """
        # Determine the number of qudits.
        if isinstance(self.dim, int):
            L = round(log(x.shape[0], self.dim))
        else:
            L = len(self.dim)

        # Start with a 1x1 identity operator.
        U = aux.eye(1, device=x.device, sparse=self.sparse)

        for i in range(L):
            # Determine the dimension d for the i-th qudit.
            d = self.dim if isinstance(self.dim, int) else self.dim[i]
            if i in self.index:
                # Retrieve target levels for this qudit.
                j_val = self.j_map[i]
                k_val = self.k_map[i]
                if j_val >= d or k_val >= d:
                    raise ValueError(
                        f"For qudit {i} with dimension {d}, the target levels j={j_val} and k={k_val} are out of range."
                    )
                # Build indices for updating the submatrix:
                #   (j,j) and (k,k) -> cos(angle/2)
                #   (j,k) -> -sin(angle/2)
                #   (k,j) -> sin(angle/2)
                indices = torch.tensor(
                    [[j_val, k_val, j_val, k_val], [j_val, k_val, k_val, j_val]],
                    device=x.device,
                )
                values = torch.zeros(4, dtype=torch.complex64, device=x.device)
                angle_val = self.angle[i] if param is None else param[i]
                values[0] = torch.cos(angle_val / 2)
                values[1] = torch.cos(angle_val / 2)
                values[2] = -torch.sin(angle_val / 2)
                values[3] = torch.sin(angle_val / 2)
                # Create a d x d identity matrix.
                M = aux.eye(dim=d, device=x.device, sparse=self.sparse)
                # Update the specified entries.
                if self.sparse:
                    M = aux.sparse_index_put(M, indices, values, self.device)
                else:
                    M.index_put_(tuple(indices), values)

                U = aux.kron(U, M, sparse=self.sparse)
            else:
                M = aux.eye(
                    dim=(d if isinstance(self.dim, int) else self.dim[i]),
                    device=x.device,
                    sparse=self.sparse,
                )
                U = aux.kron(U, M, sparse=self.sparse)
        return U @ x

    def matrix(self, param=None):
        """
        Construct and return the overall unitary operator representing the RY gate applied
        to the specified qudits in the circuit.

        The rotation angle used for each target qudit is taken from the internal parameter `self.angle`
        unless an alternative angle is provided via the `param` argument.

        Args:
            param (list or bool):
                If None, the method uses the internal angle parameter for each qudit.
                If a list is provided, its elements override the corresponding entries in `self.angle`
                for the rotation angles.

        Returns:
            torch.Tensor: The complete unitary operator (as a torch.Tensor) representing the RY gate
                        applied to the qudit state.
        """

        L = self.wires

        U = aux.eye(1, device=self.device, sparse=self.sparse)

        for i in range(L):
            d = self.dim if isinstance(self.dim, int) else self.dim[i]
            if i in self.index:
                j_val = self.j_map[i]
                k_val = self.k_map[i]
                if j_val >= d or k_val >= d:
                    raise ValueError(
                        f"For qudit {i} with dimension {d}, the target levels j={j_val} and k={k_val} are out of range."
                    )
                indices = torch.tensor(
                    [[j_val, k_val, j_val, k_val], [j_val, k_val, k_val, j_val]],
                    device=self.device,
                )
                values = torch.zeros(4, dtype=torch.complex64, device=self.device)
                angle_val = (
                    self.angle[i]
                    if param is None
                    else torch.tensor(param[i], device=self.device)
                )
                values[0] = torch.cos(angle_val / 2)
                values[1] = torch.cos(angle_val / 2)
                values[2] = -torch.sin(angle_val / 2)
                values[3] = torch.sin(angle_val / 2)
                M = aux.eye(dim=d, device=self.device, sparse=self.sparse)
                M.index_put_(tuple(indices), values)
                U = aux.kron(U, M, sparse=self.sparse)
            else:
                M = aux.eye(
                    dim=(d if isinstance(self.dim, int) else self.dim[i]),
                    device=self.device,
                    sparse=self.sparse,
                )
                U = aux.kron(U, M, sparse=self.sparse)

        return U


class RZ(nn.Module):
    r"""
    Rotation-Z (RZ) Gate for qudits.

    The RZ gate represents a rotation around the Z-axis of the Bloch sphere in a qudit system.
    For a qubit (2-level system), the matrix representation is given by

    .. math::

            RZ(\theta) = 
            \begin{pmatrix}
            e^{-i\theta/2} & 0 \\
            0 & e^{i\theta/2}
            \end{pmatrix}

    For higher-dimensional qudits, the gate affects only the specified level \(j\) (i.e. multiplies
    the \(j\)th basis vector by a phase \(e^{i\theta}\)) while leaving the other levels unchanged.

    **Arguments:**
        j (int or list of int): 
            The level to which the phase rotation is applied. If an integer is provided, it is applied
            to all target qudits.
        index (list of int): 
            The indices of the qudits to which the RZ gate is applied.
        dim (int or list of int): 
            The dimension of the qudit. If an integer is provided, all qudits are assumed to have that dimension.
            If a list is provided, each element specifies the dimension of the corresponding qudit.
        wires (int): 
            The total number of qudits in the circuit (used when `dim` is an integer).
        device (str): 
            The device on which computations are performed. Default is 'cpu'.
        angle (float or torch.Tensor or None): 
            The rotation angle. If None, create a random parameter, if float or torch.Tensor, use it directly.
        sparse (bool): 
            Whether to use a sparse matrix representation. Default is False.

    **Attributes:**
        j: The target level for the phase rotation (stored per target qudit in a mapping).
        index (list of int): The indices of the qudits to which the gate is applied.
        angle (torch.nn.Parameter): The rotation angle(s) for each qudit.
        sparse (bool): Whether the matrix representation is sparse.
        dim (int or list of int): The dimension(s) of the qudit(s).

    **Examples:**
        >>> # Single dimensional qudit (qubit) case:
        >>> gate = qf.RZ(index=[0])
        >>> state = qf.State('0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits:
        >>> # For example, if the first qudit is a qubit and the second is a qutrit,
        >>> # and we wish to apply the phase rotation only to level 1 of each,
        >>> # then:
        >>> gate = qf.RZ(index=[0,1], dim=[2,3], j=[1,1], device='cpu')
        >>> state = qf.State('0-2', dim=[2,3])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(
        self, j=1, index=[0], dim=2, wires=1, device="cpu", angle=None, sparse=False
    ):
        super(RZ, self).__init__()
        self.sparse = sparse
        self.index = index
        self.wires = wires
        self.device = device

        # Process dimensions.
        if isinstance(dim, int):
            self.dim = dim
            num_wires = wires
        else:
            self.dim = dim  # a list of dimensions
            num_wires = len(dim)

        # Process target level j.
        if isinstance(j, int):
            self.j_map = {t: int(j) for t in self.index}
        else:
            if len(j) != len(self.index):
                raise ValueError(
                    "If 'j' is provided as a list, its length must equal the number of target qudits in 'index'."
                )
            self.j_map = {t: int(j_val) for t, j_val in zip(self.index, j)}

        # Initialize the rotation angle parameter.
        if angle is None:
            self.angle = nn.Parameter(torch.randn(num_wires, device=device))
        elif isinstance(angle, torch.Tensor):
            self.angle = angle
        else:
            self.angle = nn.Parameter(angle * torch.ones(num_wires, device=device))

    def forward(self, x, param=None):
        """
        Apply the RZ gate to the qudit state.

        Args:
            x (torch.Tensor):
                The input state tensor (a column vector) whose dimension is the product of the individual
                qudit dimensions.
            param (torch.Tensor or bool):
                If None, use the internal angle parameter; if provided, use it as the rotation angle.

        Returns:
            torch.Tensor: The resulting state after applying the RZ gate.
        """
        # Determine the number of qudits.
        if isinstance(self.dim, int):
            L = round(log(x.shape[0], self.dim))
        else:
            L = len(self.dim)

        U = aux.eye(1, device=x.device, sparse=self.sparse)

        for i in range(L):
            # Determine dimension for qudit i.
            d = self.dim if isinstance(self.dim, int) else self.dim[i]
            if i in self.index:
                # Get target level for this qudit.
                j_val = self.j_map[i]
                if j_val < 0 or j_val >= d:
                    raise ValueError(
                        f"For qudit {i} with dimension {d}, the target level j={j_val} is out of range."
                    )
                # Determine rotation angle for this qudit.
                angle_val = self.angle[i] if param is None else torch.tensor(param[i], device=self.device)
                # Build the d x d RZ matrix.
                if d == 2:
                    # For qubits, use the standard formulation.
                    # If j_val==0, use diag(e^{iθ/2}, e^{-iθ/2}); if j_val==1, use diag(e^{-iθ/2}, e^{iθ/2}).
                    if j_val == 0:
                        phase0 = torch.exp(1j * angle_val / 2)
                        phase1 = torch.exp(-1j * angle_val / 2)
                    else:
                        phase0 = torch.exp(-1j * angle_val / 2)
                        phase1 = torch.exp(1j * angle_val / 2)
                    M = torch.diag(
                        torch.tensor(
                            [phase0, phase1],
                            dtype=torch.complex64,
                            device=x.device,
                        )
                    )
                    if self.sparse:
                        M = M.to_sparse()
                else:
                    # For d > 2, build an identity and update only the j-th diagonal element.
                    M = aux.eye(dim=d, device=x.device, sparse=self.sparse)
                    # Update the j-th diagonal element with the phase factor.
                    # (This simple choice applies a phase rotation only on the j-th level.)
                    phase = torch.exp(1j * angle_val)
                    # M[j_val, j_val] is replaced.
                    M.index_put_(
                        (
                            torch.tensor([j_val], device=x.device),
                            torch.tensor([j_val], device=x.device),
                        ),
                        torch.tensor(phase, dtype=torch.complex64, device=x.device),
                    )
                U = aux.kron(U, M, sparse=self.sparse)
            else:
                # Use identity for non-target qudits.
                M = aux.eye(
                    dim=(d if isinstance(self.dim, int) else self.dim[i]),
                    device=x.device,
                    sparse=self.sparse,
                )
                U = aux.kron(U, M, sparse=self.sparse)
        return U @ x

    def matrix(self, param=None):
        """
        Construct and return the overall unitary operator representing the RZ gate applied
        to the specified qudits in the circuit.

        The rotation angle used for each target qudit is taken from the internal parameter `self.angle`
        unless an alternative angle is provided via the `param` argument.

        Args:
            param (list or bool):
                If None, the method uses the internal angle parameter for each qudit.
                If a list is provided, its elements override the corresponding entries in `self.angle`
                for the rotation angles.

        Returns:
            torch.Tensor: The complete unitary operator (as a torch.Tensor) representing the RZ gate
                        applied to the qudit state.
        """

        L = self.wires
        U = aux.eye(1, device=self.device, sparse=self.sparse)
        for i in range(L):
            d = self.dim if isinstance(self.dim, int) else self.dim[i]
            if i in self.index:
                j_val = self.j_map[i]
                if j_val < 0 or j_val >= d:
                    raise ValueError(
                        f"For qudit {i} with dimension {d}, the target level j={j_val} is out of range."
                    )
                angle_val = (
                    self.angle[i]
                    if param is None
                    else torch.tensor(param[i], device=self.device)
                )
                if d == 2:
                    if j_val == 0:
                        phase0 = torch.exp(1j * angle_val / 2)
                        phase1 = torch.exp(-1j * angle_val / 2)
                    else:
                        phase0 = torch.exp(-1j * angle_val / 2)
                        phase1 = torch.exp(1j * angle_val / 2)
                    M = torch.diag(
                        torch.tensor(
                            [phase0, phase1], dtype=torch.complex64, device=self.device
                        )
                    )
                else:
                    M = aux.eye(dim=d, device=self.device, sparse=self.sparse)
                    phase = torch.exp(1j * angle_val)
                    M.index_put_(
                        (
                            torch.tensor([j_val], device=self.device),
                            torch.tensor([j_val], device=self.device),
                        ),
                        torch.tensor(phase, dtype=torch.complex64, device=self.device),
                    )
                U = aux.kron(U, M, sparse=self.sparse)
            else:
                M = aux.eye(
                    dim=(d if isinstance(self.dim, int) else self.dim[i]),
                    device=self.device,
                    sparse=self.sparse,
                )
                U = aux.kron(U, M, sparse=self.sparse)

        return U


class CNOT(nn.Module):
    r"""
    Controlled-NOT (CNOT) Gate for qudits.

    The CNOT gate is a controlled gate where the target qudit is shifted based on the state of the
    control qudit. For qubits (2-level systems), the standard CNOT is recovered. For higher-dimensional
    qudits, if the control qudit is in state \(|c\rangle\) and the target qudit is in state \(|t\rangle\),
    the target is updated to \(|(t + c) \mod d_t\rangle\), where \(d_t\) is the dimension of the target qudit.

    **Arguments:**
        index (list of int): The indices of the control and target qudits, where `index[0]` is the control
                             and `index[1]` is the target. Default is [0, 1].
        wires (int): The total number of qudits in the circuit. Default is 2.
        dim (int or list of int): The dimension(s) of the qudits. If an integer, all qudits have that dimension.
                                  If a list, each element corresponds to the dimension of a qudit.
        device (str): The device on which computations are performed. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.
        inverse (bool): Whether to apply the inverse of the CNOT gate. Default is False.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (int or list of int): The dimension(s) of the qudits.
        U (torch.Tensor): The matrix representation of the CNOT gate.
        inverse (bool): Whether the matrix is inverted.

    **Examples:**
        >>> # Uniform qudit case (e.g. qubits)
        >>> gate = qf.CNOT(index=[0,1], wires=2, dim=2)
        >>> state = qf.State('0-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: first qudit (control) is a qubit (dim=2) and second (target) is a qutrit (dim=3)
        >>> gate = CNOT(index=[0,1], wires=2, dim=[2,3])
        >>> state = State('1-2', dim=[2,3])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(
        self, index=[0, 1], wires=2, dim=2, device="cpu", sparse=False, inverse=False
    ):
        super(CNOT, self).__init__()
        self.index = index
        self.wires = wires
        self.device = device
        self.sparse = sparse
        self.inverse = inverse

        # Dense matrix implementation.
        if not sparse:
            # Determine the dimensions for each qudit.
            if isinstance(dim, int):
                dims_list = [dim] * wires
            else:
                dims_list = dim
                if len(dims_list) != wires:
                    raise ValueError(
                        "For multidimensional qudits, length of dim list must equal wires."
                    )

            # Generate the full computational basis as a tensor.
            # Each row corresponds to one computational basis element.
            L = torch.tensor(list(itertools.product(*[range(d) for d in dims_list])))
            # Create a copy to modify for the new target values.
            l2ns = L.clone()
            # For the target qudit, perform modular addition:
            #   new target = (control + target) mod (target dimension)
            d_target = dims_list[index[1]]
            l2ns[:, index[1]] = (L[:, index[0]] + L[:, index[1]]) % d_target

            # Build the unitary matrix: U_{ij} = 1 if the i-th basis vector maps to the j-th basis vector.
            # Compare every row of L to every row of l2ns.
            # indices is a boolean matrix of shape (num_basis, num_basis).
            indices = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
            # Create U as a complex tensor.
            U = torch.where(
                indices,
                torch.tensor([1.0 + 0j], dtype=torch.complex64),
                torch.tensor([0.0], dtype=torch.complex64),
            ).to(device)
        # Sparse matrix implementation (if provided, user must update aux.CNOT_sparse for multidimensional support)
        else:
            # We assume aux.CNOT_sparse is updated to handle a list for dims.
            U = aux.CNOT_sparse(index[0], index[1], dim, wires, device=device)

        # Apply inverse if requested.
        if inverse:
            U = torch.conj(U).T.contiguous()

        self.register_buffer("U", U)

    def forward(self, x):
        """
        Apply the CNOT gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CNOT gate.
        """
        return self.U @ x

    def matrix(self):
        """
        Retrieve the unitary matrix representing the CNOT gate.

        Returns:
            torch.Tensor: The unitary matrix representing the CNOT gate.
        """

        return self.U


class CZ(nn.Module):
    r"""
    Controlled-Z (CZ) Gate for qudits.

    The CZ gate applies a Z operation on the target qudit if the control qudit is in a specific state.
    For qubits, it recovers the standard controlled-Z behavior. For qudits, the gate is defined such that
    if the control qudit (at index `index[0]`) is in state \(|c\rangle\) and the target qudit (at index
    `index[1]`) is in state \(|t\rangle\), then the target qudit is acted upon by a Z rotation:

    .. math::
          |c,t\rangle \to \left(|c\rangle\langle c| \otimes Z_t(c)\right)|c,t\rangle,

    where the target qudit receives a phase shift determined by \(c\). In our implementation, for each
    possible control state \(c\), we construct the operator

    .. math::
          u_c = \bigotimes_{i=0}^{wires-1} U_i,

    with
      - \(U_i =\) the projector \(|c\rangle\langle c|\) on the control qudit (if \(i = index[0]\)),
      - \(U_i =\) the Z gate \(Z(s=c)\) on the target qudit (if \(i = index[1]\)), and
      - \(U_i = I\) for all other qudits.

    The full CZ operator is then obtained by summing \(u_c\) over all control states.

    **Arguments:**
        index (list of int): The indices of the control and target qudits, where `index[0]` is the control
                             and `index[1]` is the target. Default is [0, 1].
        dim (int or list of int): The dimension of the qudits. If an integer, all qudits are assumed to have that
                                  dimension; if a list, each element specifies the dimension of the corresponding qudit.
        wires (int): The total number of qudits in the circuit. Default is 2. (If `dim` is a list, `wires` is set to len(dim).)
        device (str): The device to perform the computations on. Default is 'cpu'.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (int or list of int): The dimension(s) of the qudits.
        U (torch.Tensor): The matrix representation of the CZ gate.

    **Examples:**
        >>> # Single dimensional qudit case (e.g., qubits):
        >>> gate = qf.CZ(index=[0, 1], dim=2, wires=2)
        >>> state = qf.State('0-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits:
        >>> # For instance, first qudit is a qutrit (dim=3) and second is a qubit (dim=2)
        >>> gate = qf.CZ(index=[0, 1], dim=[3,2])
        >>> state = qf.State('1-0', dim=[3,2])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, device="cpu"):
        super(CZ, self).__init__()

        # Process dimensions: if dim is an int, create a list of that dimension.
        if isinstance(dim, int):
            dims_list = [dim] * wires
        else:
            dims_list = dim
            wires = len(dims_list)

        self.dim = dims_list
        self.index = index
        self.wires = wires

        # Total Hilbert space dimension.
        D = int(np.prod(dims_list))
        U = torch.zeros((D, D), device=device, dtype=torch.complex64)

        # Loop over all possible control states (for the control qudit).
        control_dim = dims_list[index[0]]
        for c_val in range(control_dim):
            u = torch.eye(1, device=device, dtype=torch.complex64)
            # Build the operator for each qudit.
            for i in range(wires):
                if i == index[0]:
                    # For the control qudit, use the projector onto state |c_val>.
                    # aux.base(dim, device) returns a list of basis column vectors.
                    proj = aux.base(dims_list[i], device=device)[
                        c_val
                    ]  # shape (dims_list[i], 1)
                    P = proj @ proj.T.conj()  # projector matrix
                    u = torch.kron(u, P)
                elif i == index[1]:
                    # For the target qudit, apply a Z gate with phase shift parameter s = c_val.
                    M = Z(dim=dims_list[i], device=device, s=c_val).matrix()
                    u = torch.kron(u, M)
                else:
                    # For other qudits, use the identity.
                    u = torch.kron(
                        u, torch.eye(dims_list[i], device=device, dtype=torch.complex64)
                    )
            U += u

        self.register_buffer("U", U)

    def forward(self, x):
        """
        Apply the CZ gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CZ gate.
        """
        return self.U @ x

    def matrix(self):
        """
        Retrieve the unitary matrix representing the CZ gate.

        Returns:
            torch.Tensor: The unitary matrix representing the CZ gate.
        """

        return self.U


class SWAP(nn.Module):
    r"""
    SWAP Gate for qudits.

    The SWAP gate exchanges the states of two qudits, generalizing the SWAP gate for qubits
    to higher-dimensional systems. When using multidimensional qudits, the total Hilbert space
    dimension is given by the product of the individual qudit dimensions.

    **Details:**

    For a qubit system (2-level qudits), the SWAP gate is represented as:

    .. math::

            SWAP = 
            \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
            \end{pmatrix}

    For qudits, the gate exchanges the states of the two qudits specified by their indices.

    **Arguments:**
        index (list of int): The indices of the two qudits to be swapped. Default is [0, 1].
        dim (int or list of int): The dimension of the qudits. If an integer is provided, all qudits have that dimension.
                                   If a list is provided, each element specifies the dimension of the corresponding qudit.
        wires (int): The total number of qudits in the circuit. Default is 2. (If `dim` is a list, `wires` is taken as len(dim).)
        device (str): The device on which the computations are performed. Default is 'cpu'.

    **Attributes:**
        index (list of int): The indices of the qudits to be swapped.
        dim (int or list of int): The dimension(s) of the qudits.
        U (torch.Tensor): The matrix representation of the SWAP gate.

    **Examples:**
        >>> # Single dimensional qudit case (e.g. qubits)
        >>> gate = qf.SWAP(index=[0,1], dim=2, wires=2)
        >>> state = qf.State('0-1', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: first qudit is a qubit (dim=2) and second is a qutrit (dim=3)
        >>> gate = qf.SWAP(index=[0,1], dim=[2,3])
        >>> state = qf.State('0-2', dim=[2,3])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, device="cpu"):
        super(SWAP, self).__init__()
        c = index[0]  # first qudit to swap
        t = index[1]  # second qudit to swap

        # Process dimensions: if dim is an int, assume uniform dimensions.
        if isinstance(dim, int):
            dims_list = [dim] * wires
        else:
            dims_list = dim
            wires = len(dims_list)

        # Total Hilbert space dimension is the product of individual dimensions.
        D = int(np.prod(dims_list))
        U = torch.zeros((D, D), device=device, dtype=torch.complex64)

        # Build the SWAP gate by looping over each basis state.
        # The helper functions aux.dec2den and aux.den2dec must be updated to accept dims_list.
        for k in range(D):
            # Convert from decimal to local qudit representation.
            localr = aux.dec2den(k, wires, dims_list)
            locall = localr.copy()
            # Swap the values at positions c and t.
            locall[c], locall[t] = localr[t], localr[c]
            # Convert back to a decimal index.
            globall = aux.den2dec(locall, dims_list)
            U[globall, k] = 1

        self.register_buffer("U", U)

    def forward(self, x):
        """
        Apply the SWAP gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the SWAP gate.
        """
        return self.U @ x

    def matrix(self):
        """
        Retrieve the unitary matrix representing the SWAP gate.

        Returns:
            torch.Tensor: The unitary matrix representing the SWAP gate.
        """

        return self.U


class CRX(nn.Module):
    r"""
    Controlled-RX Gate for qudits.

    The CRX gate applies an RX (rotation-X) operation on the target qudit, conditioned on the
    state of the control qudit. For qudits, if the control qudit (at index `index[0]`) is in a given state,
    the RX rotation (by an angle \(\theta\)) is applied on the target qudit (at index `index[1]`), where the rotation
    affects only the subspace spanned by the levels \(j\) and \(k\).

    **Arguments:**
        index (list of int): The indices of the control and target qudits. Default is [0, 1].
        dim (int or list of int): The dimension(s) of the qudits. If an integer, all qudits are assumed to have that dimension.
        wires (int): The total number of qudits in the circuit. Default is 2.
        j (int): The first target level for the RX rotation. Default is 0.
        k (int): The second target level for the RX rotation. Default is 1.
        device (str): The device to perform computations on. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (list of int): The list of dimensions of the qudits.
        j, k (int): The levels (0-indexed) on the target qudit that define the RX rotation subspace.
        angle (torch.nn.Parameter): A learnable rotation angle parameter.
        wires (int): Total number of qudits.
        sparse (bool): Whether a sparse matrix representation is used.

    **Examples:**
        >>> # Uniform qudit case (e.g., qubits)
        >>> gate = CRX(index=[0, 1], dim=2, wires=2, j=0, k=1)
        >>> state = State('0-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: for instance, first qudit is a qutrit (dim=3) and second is a qubit (dim=2).
        >>> # Here, we want to apply the controlled RX rotation on the target qudit's subspace between levels 0 and 1.
        >>> gate = CRX(index=[0, 1], dim=[3,2], wires=2, j=0, k=1, device='cpu')
        >>> state = State('1-0', dim=[3,2])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(
        self, index=[0, 1], dim=2, wires=2, j=0, k=1, device="cpu", sparse=False
    ):
        super(CRX, self).__init__()
        self.index = index
        self.wires = wires
        self.device = device
        self.sparse = sparse
        self.j = j
        self.k = k
        # Define angle as a learnable parameter (a single angle; you can extend this to per-wire if needed)
        self.angle = nn.Parameter(np.pi * torch.randn(1, device=device))

        # Ensure dims are given as a list.
        if isinstance(dim, int):
            self.dims_list = [dim] * wires
        else:
            self.dims_list = dim
            if len(self.dims_list) != wires:
                raise ValueError("Length of dim list must equal wires.")

    def forward(self, x, param=False):
        """
        Apply the CRX gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor (a column vector) whose dimension equals the product
                              of the individual qudit dimensions.
            param (torch.Tensor or bool): If False, use the internal angle; otherwise, use the provided angle.

        Returns:
            torch.Tensor: The resulting state after applying the CRX gate.
        """
        c = self.index[0]  # control qudit index
        t = self.index[1]  # target qudit index

        # Global Hilbert space dimension.
        D = int(np.prod(self.dims_list))
        # Build U as a dense matrix.
        U = torch.zeros((D, D), dtype=torch.complex64, device=x.device)

        # Build dims for subspace excluding the target qudit.
        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        # number of basis states for all qudits except target
        Dl = int(np.prod(dims_without_target))

        indices_list = []
        values_list = []

        # For computing the rotation angle, we need the reduced index for the control qudit.
        # In the reduced list (i.e. after removing target qudit), the index of the control qudit is:
        c_reduced = c if c < t else c - 1

        for m in range(Dl):
            # Obtain the multi-index for the subspace (all qudits except target)
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            # Determine the rotation angle:
            if self.wires == 2:
                angle_val = (local[0] * self.angle) / 2
            else:
                angle_val = (local[c_reduced] * self.angle) / 2

            # Build two versions for the target qudit: one with level j and one with level k.
            listj = local.copy()
            listj.insert(t, self.j)  # use self.j directly (assumed 0-indexed)
            intj = aux.den2dec(listj, self.dims_list)
            listk = local.copy()
            listk.insert(t, self.k)
            intk = aux.den2dec(listk, self.dims_list)

            # Create the 2x2 submatrix for the target qudit.
            # The nontrivial part: between states intj and intk.
            # We build a 2x2 block:
            # [ cos(angle)    -i sin(angle) ]
            # [ -i sin(angle)  cos(angle)    ]
            base_indices = torch.tensor(
                [[intj, intk, intj, intk], [intj, intk, intk, intj]], device=x.device
            )
            base_values = torch.zeros(4, dtype=torch.complex64, device=x.device)
            base_values[0] = torch.cos(angle_val)
            base_values[1] = torch.cos(angle_val)
            base_values[2] = -1j * torch.sin(angle_val)
            base_values[3] = -1j * torch.sin(angle_val)

            # Now, for every other level l in the target qudit not equal to self.j or self.k,
            # the operator acts as identity.
            indices = base_indices.clone()
            values = base_values.clone()
            for l in range(self.dims_list[t]):
                if l != self.j and l != self.k:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dims_list)
                    # For identity, we set the (intl, intl) element to 1.
                    new_index = torch.tensor([[intl]], device=x.device)
                    # We need to add the same index for both row and column.
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat(
                        (
                            values,
                            torch.tensor([1.0], dtype=torch.complex64, device=x.device),
                        )
                    )
            indices_list.append(indices)
            values_list.append(values)

        # Concatenate over the loop.
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        # (Optional) Filter out any negative indices.
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if not self.sparse:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=x.device)
        return U @ x

    def matrix(self, param=False):
        """
        Construct and return the full matrix representation of the CRX gate.

        Args:
            param (torch.Tensor or bool):
                If False, the method uses the internal angle parameter (`self.angle`).
                Otherwise, the provided tensor overrides the internal angle.

        Returns:
            torch.Tensor: The dense unitary matrix representing the CRX gate.
        """

        c = self.index[0]
        t = self.index[1]
        D = int(np.prod(self.dims_list))
        U = torch.zeros((D, D), dtype=torch.complex64, device=self.device)

        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        Dl = int(np.prod(dims_without_target))

        indices_list = []
        values_list = []

        c_reduced = c if c < t else c - 1

        current_angle = (
            self.angle if param is False else torch.tensor([param], device=self.device)
        )

        for m in range(Dl):
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            if self.wires == 2:
                angle_val = (local[0] * current_angle) / 2
            else:
                angle_val = (local[c_reduced] * current_angle) / 2

            listj = local.copy()
            listj.insert(t, self.j)
            intj = aux.den2dec(listj, self.dims_list)
            listk = local.copy()
            listk.insert(t, self.k)
            intk = aux.den2dec(listk, self.dims_list)

            base_indices = torch.tensor(
                [[intj, intk, intj, intk], [intj, intk, intk, intj]], device=self.device
            )
            base_values = torch.zeros(4, dtype=torch.complex64, device=self.device)
            base_values[0] = torch.cos(angle_val)
            base_values[1] = torch.cos(angle_val)
            base_values[2] = -1j * torch.sin(angle_val)
            base_values[3] = -1j * torch.sin(angle_val)

            indices = base_indices.clone()
            values = base_values.clone()
            for l in range(self.dims_list[t]):
                if l != self.j and l != self.k:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dims_list)
                    new_index = torch.tensor([[intl]], device=self.device)
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat(
                        (
                            values,
                            torch.tensor(
                                [1.0], dtype=torch.complex64, device=self.device
                            ),
                        )
                    )
            indices_list.append(indices)
            values_list.append(values)

        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)

        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if not self.sparse:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=self.device)

        return U

    def matrix(self, param=False):
        """
        Construct and return the full matrix representation of the CRY gate.

        Args:
            param (torch.Tensor or bool):
                If False, use the internal angle parameter (`self.angle`);
                otherwise, use the provided tensor as the rotation angle.

        Returns:
            torch.Tensor: The dense unitary matrix representing the CRY gate.
        """
        current_angle = (
            self.angle if param is False else torch.tensor([param], device=self.device)
        )

        c = self.index[0]
        t = self.index[1]
        D = int(np.prod(self.dims_list))
        U = torch.zeros((D, D), dtype=torch.complex64, device=self.device)

        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        Dl = int(np.prod(dims_without_target))

        c_reduced = c if c < t else c - 1

        indices_list = []
        values_list = []

        for m in range(Dl):
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            if self.wires == 2:
                angle_val = (local[0] * current_angle) / 2
            else:
                angle_val = (local[c_reduced] * current_angle) / 2

            listj = local.copy()
            listj.insert(t, self.j)
            intj = aux.den2dec(listj, self.dims_list)

            listk = local.copy()
            listk.insert(t, self.k)
            intk = aux.den2dec(listk, self.dims_list)

            base_indices = torch.tensor(
                [[intj, intk, intj, intk], [intj, intk, intk, intj]], device=self.device
            )
            base_values = torch.zeros(4, dtype=torch.complex64, device=self.device)
            base_values[0] = torch.cos(angle_val)
            base_values[1] = torch.cos(angle_val)
            base_values[2] = -torch.sin(angle_val)
            base_values[3] = -torch.sin(angle_val)

            indices = base_indices.clone()
            values = base_values.clone()
            for l in range(self.dims_list[t]):
                if l != self.j and l != self.k:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dims_list)
                    new_index = torch.tensor([[intl]], device=self.device)
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat(
                        (
                            values,
                            torch.tensor(
                                [1.0], dtype=torch.complex64, device=self.device
                            ),
                        )
                    )
            indices_list.append(indices)
            values_list.append(values)

        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if not self.sparse:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=self.device)

        return U


class CRY(nn.Module):
    r"""
    Controlled-RY Gate for qudits.

    The CRY gate applies an RY (rotation-Y) operation on the target qudit, conditioned on the state of the
    control qudit. For qudits, if the control qudit (at index `index[0]`) is in a given state, then an RY rotation
    (by angle \(\theta\)) is applied on the target qudit (at index `index[1]`). The rotation acts nontrivially only
    on the two-dimensional subspace spanned by levels \(j\) and \(k\) of the target.

    **Arguments:**
        index (list of int): The indices of the control and target qudits. Default is [0, 1].
        dim (int or list of int): The dimension(s) of the qudits. If an integer, all qudits are assumed to have that
                                  dimension; if a list, each element specifies the dimension of the corresponding qudit.
        wires (int): The total number of qudits in the circuit. Default is 2.
        j (int): The first target level for the RY rotation (0-indexed). Default is 0.
        k (int): The second target level for the RY rotation (0-indexed). Default is 1.
        device (str): The device for computations. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (list of int): The dimensions of the qudits.
        angle (torch.nn.Parameter): The learnable rotation angle.
        wires (int): The total number of qudits.
        sparse (bool): Whether a sparse matrix is used.

    **Examples:**
        >>> # Single dimensional qudit case (e.g. qubits)
        >>> gate = qf.CRY(index=[0,1], dim=2, wires=2, j=0, k=1)
        >>> state = qf.State('0-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: first qudit is a qutrit (dim=3) and second is a qubit (dim=2)
        >>> gate = qf.CRY(index=[0,1], dim=[3,2], wires=2, j=0, k=1, device='cpu')
        >>> state = qf.State('1-0', dim=[3,2])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(
        self, index=[0, 1], dim=2, wires=2, j=0, k=1, device="cpu", sparse=False
    ):
        super(CRY, self).__init__()
        self.index = index
        self.wires = wires
        self.device = device
        self.sparse = sparse
        self.j = j
        self.k = k
        # Learnable rotation angle.
        self.angle = nn.Parameter(np.pi * torch.randn(1, device=device))

        # Process dimensions: if dim is an int, create a list.
        if isinstance(dim, int):
            self.dims_list = [dim] * wires
        else:
            self.dims_list = dim
            if len(self.dims_list) != wires:
                raise ValueError("Length of dim list must equal wires.")

    def forward(self, x, param=False):
        """
        Apply the CRY gate to the qudit state.

        Args:
            x (torch.Tensor): The input state (a column vector) whose dimension equals the product
                              of the individual qudit dimensions.
            param (torch.Tensor or bool): If False, use the internal angle; otherwise, use the provided angle.

        Returns:
            torch.Tensor: The state after applying the CRY gate.
        """
        c = self.index[0]  # control qudit index
        t = self.index[1]  # target qudit index

        # Total Hilbert space dimension.
        D = int(np.prod(self.dims_list))
        U = torch.zeros((D, D), dtype=torch.complex64, device=x.device)

        # Build the reduced dimensions: remove the target qudit.
        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        Dl = int(np.prod(dims_without_target))

        # For the control qudit in the reduced space, if its original index is after the target, subtract one.
        c_reduced = c if c < t else c - 1

        indices_list = []
        values_list = []

        for m in range(Dl):
            # Obtain the multi-index (for all qudits except target) using the updated aux.dec2den.
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            # Compute rotation angle based on the control qudit's value in the reduced space.
            if self.wires == 2:
                angle_val = (local[0] * self.angle) / 2
            else:
                angle_val = (local[c_reduced] * self.angle) / 2

            # Build two versions: one where the target qudit is set to j and one where it is set to k.
            listj = local.copy()
            listj.insert(t, self.j)
            intj = aux.den2dec(listj, self.dims_list)

            listk = local.copy()
            listk.insert(t, self.k)
            intk = aux.den2dec(listk, self.dims_list)

            # Construct a 2x2 submatrix for the target qudit.
            indices = torch.tensor(
                [[intj, intk, intj, intk], [intj, intk, intk, intj]], device=x.device
            )
            values = torch.zeros(4, dtype=torch.complex64, device=x.device)
            values[0] = torch.cos(angle_val)
            values[1] = torch.cos(angle_val)
            values[2] = -torch.sin(angle_val)
            values[3] = -torch.sin(angle_val)

            # For every other level l in the target qudit not equal to self.j or self.k, the operator acts as identity.
            for l in range(self.dims_list[t]):
                if l != self.j and l != self.k:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dims_list)
                    new_index = torch.tensor([[intl]], device=x.device)
                    # Add the diagonal element 1.
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat(
                        (
                            values,
                            torch.tensor([1.0], dtype=torch.complex64, device=x.device),
                        )
                    )

            indices_list.append(indices)
            values_list.append(values)

        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if not self.sparse:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=x.device)
        return U @ x

    def matrix(self, param=False):
        """
        Construct and return the full matrix representation of the CRY gate.

        Args:
            param (torch.Tensor or bool):
                If False, use the internal angle parameter (`self.angle`);
                otherwise, use the provided tensor as the rotation angle.

        Returns:
            torch.Tensor: The dense unitary matrix representing the CRY gate.
        """
        current_angle = (
            self.angle if param is False else torch.tensor([param], device=self.device)
        )

        c = self.index[0]
        t = self.index[1]
        D = int(np.prod(self.dims_list))
        U = torch.zeros((D, D), dtype=torch.complex64, device=self.device)

        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        Dl = int(np.prod(dims_without_target))

        c_reduced = c if c < t else c - 1

        indices_list = []
        values_list = []

        for m in range(Dl):
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            if self.wires == 2:
                angle_val = (local[0] * current_angle) / 2
            else:
                angle_val = (local[c_reduced] * current_angle) / 2

            listj = local.copy()
            listj.insert(t, self.j)
            intj = aux.den2dec(listj, self.dims_list)

            listk = local.copy()
            listk.insert(t, self.k)
            intk = aux.den2dec(listk, self.dims_list)

            base_indices = torch.tensor(
                [[intj, intk, intj, intk], [intj, intk, intk, intj]], device=self.device
            )
            base_values = torch.zeros(4, dtype=torch.complex64, device=self.device)
            base_values[0] = torch.cos(angle_val)
            base_values[1] = torch.cos(angle_val)
            base_values[2] = -torch.sin(angle_val)
            base_values[3] = -torch.sin(angle_val)

            indices = base_indices.clone()
            values = base_values.clone()
            for l in range(self.dims_list[t]):
                if l != self.j and l != self.k:
                    listl = local.copy()
                    listl.insert(t, l)
                    intl = aux.den2dec(listl, self.dims_list)
                    new_index = torch.tensor([[intl]], device=self.device)
                    indices = torch.cat((indices, new_index.expand(2, -1)), dim=1)
                    values = torch.cat(
                        (
                            values,
                            torch.tensor(
                                [1.0], dtype=torch.complex64, device=self.device
                            ),
                        )
                    )
            indices_list.append(indices)
            values_list.append(values)

        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if not self.sparse:
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=self.device)

        return U


class CRZ(nn.Module):
    r"""
    Controlled-RZ Gate for qudits.

    The CRZ gate applies an RZ (rotation-Z) operation on the target qudit, conditioned on the state
    of the control qudit. For qubits, this recovers the standard controlled-RZ. For qudits, if the control
    qudit (at index `index[0]`) is in a given state, then the target qudit (at index `index[1]`) is phase-shifted.
    The phase shift is applied only to the target qudit’s basis elements according to a function of the control value.

    **Arguments:**
        index (list of int): The indices of the control and target qudits, where index[0] is the control.
        dim (int or list of int): The dimension(s) of the qudits. If an integer, all qudits have that dimension.
        wires (int): The total number of qudits in the circuit. Default is 2. (If dim is a list, wires is taken as len(dim).)
        j (int): The target level on the target qudit where a different phase is applied. Default is 1.
        device (str): The device on which computations are performed. Default is 'cpu'.
        sparse (bool): Whether to use a sparse matrix representation. Default is False.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (int or list of int): The dimension(s) of the qudits.
        angle (torch.nn.Parameter): The learnable rotation angle.
        wires (int): The total number of qudits.
        sparse (bool): Whether a sparse matrix is used.

    **Examples:**
        >>> # Single dimensional qudit case (qubits)
        >>> gate = qf.CRZ(index=[0, 1], dim=2, wires=2, j=1)
        >>> state = qf.State('0-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: e.g. first qudit has dim 3 and second dim 2.
        >>> gate = qf.CRZ(index=[0, 1], dim=[3,2], wires=2, j=1, device='cpu')
        >>> state = qf.State('0-0', dim=[3,2])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, j=1, device="cpu", sparse=False):
        super(CRZ, self).__init__()
        self.index = index
        self.j = j
        self.wires = wires
        self.device = device
        self.sparse = sparse
        # Learnable rotation angle.
        self.angle = nn.Parameter(np.pi * torch.randn(1, device=device))

        # Process dimensions: if dim is an int, convert to list.
        if isinstance(dim, int):
            self.dims_list = [dim] * wires
        else:
            self.dims_list = dim
            if len(self.dims_list) != wires:
                raise ValueError("Length of dim list must equal wires.")

    def forward(self, x):
        """
        Apply the CRZ gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor (column vector) whose dimension equals the product
                              of the individual qudit dimensions.
        Returns:
            torch.Tensor: The resulting state after applying the CRZ gate.
        """
        c = self.index[0]  # control qudit index
        t = self.index[1]  # target qudit index

        # Total Hilbert space dimension.
        D = int(np.prod(self.dims_list))

        # Build reduced dimensions (exclude target qudit).
        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        Dl = int(np.prod(dims_without_target))

        # Prepare lists to accumulate indices and values.
        indices = []
        values = []

        # Iterate over the reduced basis of all qudits except target.
        # Adjust control index in reduced space: if control index > target, then reduced index is c-1.
        c_reduced = c if c < t else c - 1

        for m in range(Dl):
            # Get the multi-index for the reduced space.
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            # For wires==2, local[0] is the control; for more, use reduced control index.
            if self.wires == 2:
                loc = local[0]
            else:
                loc = local[c_reduced]
            # Compute the rotation angle as a function of the control value.
            # The factor np.sqrt(2/(j*(j+1))) is kept from the original code.
            angle = ((loc * self.angle) / 2) * np.sqrt(2 / (self.j * (self.j + 1)))

            # Loop over the possible levels for the target qudit.
            for k_val in range(self.dims_list[t]):
                listk = local.copy()
                # Insert the target value into the reduced multi-index at position t.
                listk.insert(t, k_val)
                # Convert back to a global index using the full dims_list.
                intk = aux.den2dec(listk, self.dims_list)
                # Depending on the target level relative to j, assign phase.
                if k_val < self.j:
                    phase = torch.cos(angle) - 1j * torch.sin(angle)
                elif k_val == self.j:
                    # Multiply angle by self.j.
                    phase = torch.cos(self.j * angle) + 1j * torch.sin(self.j * angle)
                else:  # k_val > self.j
                    phase = 1.0
                indices.append([intk, intk])
                values.append(phase)

        # Convert indices and values to tensors.
        indices = torch.tensor(indices, dtype=torch.long, device=x.device).T
        values = torch.tensor(values, dtype=torch.complex64, device=x.device)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        # Build the operator U.
        if not self.sparse:
            U = torch.zeros((D, D), dtype=torch.complex64, device=x.device)
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=x.device)

        return U @ x

    def matrix(self, param=False):
        """
        Construct and return the full matrix representation of the CRZ gate.

        Args:
            param (torch.Tensor or bool):
                If False, use the internal angle parameter (`self.angle`);
                otherwise, use the provided tensor as the rotation angle.

        Returns:
            torch.Tensor: The dense unitary matrix  representing the CRZ gate.
        """
        current_angle = (
            self.angle if param is False else torch.tensor([param], device=self.device)
        )

        c = self.index[0]
        t = self.index[1]

        D = int(np.prod(self.dims_list))

        dims_without_target = self.dims_list.copy()
        del dims_without_target[t]
        Dl = int(np.prod(dims_without_target))

        indices = []
        values = []

        c_reduced = c if c < t else c - 1

        for m in range(Dl):
            local = aux.dec2den(m, self.wires - 1, dims_without_target)
            loc = local[0] if self.wires == 2 else local[c_reduced]
            angle = ((loc * current_angle) / 2) * np.sqrt(2 / (self.j * (self.j + 1)))

            for k_val in range(self.dims_list[t]):
                listk = local.copy()
                listk.insert(t, k_val)
                intk = aux.den2dec(listk, self.dims_list)
                if k_val < self.j:
                    phase = torch.cos(angle) - 1j * torch.sin(angle)
                elif k_val == self.j:
                    phase = torch.cos(self.j * angle) + 1j * torch.sin(self.j * angle)
                else:
                    phase = 1.0
                indices.append([intk, intk])
                values.append(phase)

        indices = torch.tensor(indices, dtype=torch.long, device=self.device).T
        values = torch.tensor(values, dtype=torch.complex64, device=self.device)
        mask = (indices[0] >= 0) & (indices[1] >= 0)
        indices = indices[:, mask]
        values = values[mask]

        if not self.sparse:
            U = torch.zeros((D, D), dtype=torch.complex64, device=self.device)
            U.index_put_(tuple(indices), values)
        else:
            U = torch.sparse_coo_tensor(indices, values, (D, D), device=self.device)

        return U


class CCNOT(nn.Module):
    r"""
    CCNOT (Toffoli) Gate for qudits.

    The CCNOT gate (or Toffoli gate) is a controlled-controlled NOT gate that flips the target qudit if both
    control qudits are in the specified states. For qudits, this gate is generalized to perform a modular
    arithmetic operation on the target: if the control qudits are in states \(|c_1\rangle\) and \(|c_2\rangle\),
    and the target qudit is in state \(|t\rangle\), then

    .. math::
          |c_1\, c_2\, t\rangle \to |c_1\, c_2\, ((c_1 \times c_2) + t) \mod d_t\rangle,

    where \(d_t\) is the dimension of the target qudit.

    **Arguments:**
        index (list of int): The indices of the control and target qudits, where the first two entries are the
                             control qudits and the third is the target qudit. Default is [0, 1, 2].
        dim (int or list of int): The dimension of the qudits. If an integer is provided, all qudits are assumed to have that
                                  dimension; if a list is provided, each element specifies the dimension of the corresponding qudit.
        wires (int): The total number of qudits. Default is 3.
        inverse (bool): Whether to apply the inverse of the CCNOT gate. Default is False.
        device (str): The device on which computations are performed. Default is 'cpu'.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (list of int): The dimension of each qudit.
        U (torch.Tensor): The matrix representation of the CCNOT gate.
        inverse (bool): Whether the gate is inverted.

    **Examples:**
        >>> import quforge.quforge as qf
        >>> gate = qf.CCNOT(index=[0, 1, 2], dim=2, wires=3)
        >>> state = qf.State('1-1-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # For multidimensional qudits, e.g. if the first two qudits are 3-dimensional and the target is 2-dimensional:
        >>> gate = qf.CCNOT(index=[0, 1, 2], dim=[3,3,2], wires=3)
        >>> state = qf.State('1-2-0', dim=[3,3,2])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1, 2], dim=2, wires=3, inverse=False, device="cpu"):
        super(CCNOT, self).__init__()

        # Process dimensions: if an integer is given, make a list.
        if isinstance(dim, int):
            dims_list = [dim] * wires
        else:
            dims_list = dim
            if len(dims_list) != wires:
                raise ValueError("Length of dim list must equal the number of wires.")

        self.index = index
        self.device = device
        self.dims_list = dims_list

        # Total Hilbert space dimension.
        D = int(np.prod(dims_list))
        # Generate the full computational basis.
        basis = torch.tensor(
            list(itertools.product(*[range(d) for d in dims_list]))
        ).to(device)
        # Make a copy that will be modified.
        basis_modified = basis.clone()
        # Update the target qudit entry.
        target_dim = dims_list[index[2]]
        # Here, we perform: new_target = (control1 * control2 + target) mod (target_dim)
        basis_modified[:, index[2]] = (
            basis[:, index[0]] * basis[:, index[1]] + basis[:, index[2]]
        ) % target_dim

        # Build the unitary matrix: for each pair (i,j) we set U[i,j] = 1 if basis[i] == modified_basis[j]
        eq_matrix = torch.all(basis[:, None, :] == basis_modified[None, :, :], dim=2)
        U = torch.where(
            eq_matrix,
            torch.tensor(1.0 + 0j, dtype=torch.complex64, device=device),
            torch.tensor(0.0, dtype=torch.complex64, device=device),
        )
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer("U", U)

    def forward(self, x):
        """
        Apply the CCNOT gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the CCNOT gate.
        """
        return self.U @ x

    def matrix(self):
        """
        Retrieve the full matrix representation of the CCNOT gate.

        Returns:
            torch.Tensor: The unitary matrix representing the CCNOT gate.
        """
        return self.U


class MCX(nn.Module):
    r"""
    Multi-Controlled CX Gate for qudits.

    The MCX gate applies a controlled-X operation where multiple control qudits are used to control
    a target qudit. For qubits (2-level systems) this recovers the standard controlled-X gate.
    For qudits, if the control qudits are in states \(|c_1\rangle, |c_2\rangle, \dots, |c_{n-1}\rangle\)
    and the target qudit is in state \(|t\rangle\), then the target is updated as

    .. math::
          |c_1, c_2, \dots, c_{n-1}, t\rangle \to |c_1, c_2, \dots, c_{n-1}, (c_1 \cdot c_2 \cdots c_{n-1} + t) \mod d_t\rangle,

    where \(d_t\) is the dimension of the target qudit.

    **Arguments:**
        index (list of int): The indices of the control and target qudits. The last element is the target qudit,
                             and the preceding indices are the control qudits. Default is [0, 1].
        dim (int or list of int): The dimension of the qudits. If an integer, all qudits are assumed to have that
                                  dimension; if a list, each element specifies the dimension of the corresponding qudit.
        wires (int): The total number of qudits. Default is 2.
        inverse (bool): Whether to apply the inverse of the MCX gate. Default is False.
        device (str): The device on which computations are performed. Default is 'cpu'.

    **Attributes:**
        index (list of int): The indices of the control and target qudits.
        dim (list of int): The dimension of each qudit.
        U (torch.Tensor): The matrix representation of the MCX gate.
        inverse (bool): Whether the gate is inverted.

    **Examples:**
        >>> import quforge.quforge as qf
        >>> # Single dimensional qudit case (e.g. qubits)
        >>> gate = qf.MCX(index=[0, 1], dim=2, wires=2)
        >>> state = qf.State('0-0', dim=2)
        >>> result = gate(state)
        >>> print(result)
        >>>
        >>> # Multidimensional qudits: for example, first qudit is 3-dimensional and second is 2-dimensional.
        >>> gate = qf.MCX(index=[0, 1], dim=[3, 2], wires=2, device='cpu')
        >>> state = qf.State('1-0', dim=[3, 2])
        >>> result = gate(state)
        >>> print(result)
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, inverse=False, device="cpu"):
        super(MCX, self).__init__()

        # If dim is an integer, create a list of equal dimensions.
        if isinstance(dim, int):
            dims_list = [dim] * wires
        else:
            dims_list = dim
            if len(dims_list) != wires:
                raise ValueError("Length of dim list must equal the number of wires.")

        self.index = index
        self.dims_list = dims_list
        self.device = device

        # Total Hilbert space dimension.
        # D = int(np.prod(dims_list))

        # Generate the full computational basis.
        # Each row is a tuple representing a basis state.
        basis = torch.tensor(
            list(itertools.product(*[range(d) for d in dims_list]))
        ).to(device)
        # Make a copy that will be modified.
        basis_modified = basis.clone()

        # Compute the product of the control qudit values.
        control_value = 1
        for i in range(len(index) - 1):
            control_value *= basis_modified[:, index[i]]
        # For the target qudit (index[-1]), update its value:
        # new_target = (control_value + original_target) mod (target qudit dimension)
        target_dim = dims_list[index[-1]]
        basis_modified[:, index[-1]] = (
            control_value + basis_modified[:, index[-1]]
        ) % target_dim

        # Build the unitary matrix.
        eq_matrix = torch.all(basis[:, None, :] == basis_modified[None, :, :], dim=2)
        U = torch.where(
            eq_matrix,
            torch.tensor(1.0 + 0j, dtype=torch.complex64, device=device),
            torch.tensor(0.0, dtype=torch.complex64, device=device),
        )
        if inverse:
            U = torch.conj(U).T.contiguous()
        self.register_buffer("U", U)

    def forward(self, x):
        """
        Apply the MCX gate to the qudit state.

        Args:
            x (torch.Tensor): The input state tensor of the qudits.

        Returns:
            torch.Tensor: The resulting state after applying the MCX gate.
        """
        return self.U @ x

    def matrix(self):
        """
        Retrieve the full matrix representation of the MCX gate.

        Returns:
            torch.Tensor: The unitary matrix representing the MCX gate.
        """
        return self.U


class U(nn.Module):
    r"""
    Random or Custom Unitary Gate for qudits.

    This gate can either apply a user-provided unitary matrix or, if no matrix is provided,
    generate a random unitary operator on the Hilbert space of a set of qudits. The gate can be
    applied on the entire system or on a specified subset of qudits via the `index` flag.

    **Arguments:**
        matrix (torch.Tensor or None): The custom unitary matrix to be applied as the gate.
            If None (default), a random unitary is generated (via a random parameter that is
            exponentiated to produce a unitary).
        dim (int or list of int): The dimension(s) of the qudits. If an integer is provided, all qudits
            are assumed to have that dimension; if a list is provided, each element specifies the dimension
            of a corresponding qudit.
        wires (int): The total number of qudits in the circuit (used only when `dim` is an integer).
        device (str): The device on which computations are performed (e.g. 'cpu' or 'cuda').
        index (int, list of int, or None): The index/indices of the qudit(s) on which to apply the gate.
            If None, the gate acts on the entire system. For a single index, an integer can be provided.
            For a subset, provide a list of indices.

    **Examples:**
        >>> # Full-system custom gate (acting on 3 qudits, each with dimension 2)
        >>> custom_matrix = torch.tensor(np.eye(2**3), dtype=torch.complex64)
        >>> gate = U(matrix=custom_matrix, dim=2, wires=3, device='cpu')
        >>> state = ...  # some state of dimension 2**3
        >>> result = gate(state)
        >>>
        >>> # Full-system random gate (acting on 3 qudits, each with dimension 2)
        >>> gate = U(matrix=None, dim=2, wires=3, device='cpu')
        >>>
        >>> # Custom gate acting only on qudits 0 and 2 of a 3-qudit system
        >>> custom_matrix = torch.tensor(np.eye(2*2), dtype=torch.complex64)  # 4x4 matrix
        >>> gate = U(matrix=custom_matrix, dim=2, wires=3, index=[0,2], device='cpu')
        >>>
        >>> # Random gate acting only on qudits 1 and 2 of a 3-qudit system
        >>> gate = U(matrix=None, dim=2, wires=3, index=[1,2], device='cpu')
    """

    def __init__(self, matrix=None, dim=2, wires=1, device="cpu", index=None):
        super(U, self).__init__()
        self.device = device

        # Process dimensions: if dim is an int, assume all qudits share that dimension.
        if isinstance(dim, int):
            self.dims_list = [dim] * wires
            self.wires = wires
        else:
            self.dims_list = dim
            self.wires = len(dim)

        # Process the index flag.
        if index is None:
            self.indices = None  # acts on the full Hilbert space
        elif isinstance(index, int):
            self.indices = [index]
        else:
            self.indices = list(index)
            self.indices.sort()

        # Compute full Hilbert space dimension.
        self.total_dim = int(np.prod(self.dims_list))

        # If acting on a subspace, determine its dimension.
        if self.indices is None:
            self.sub_dim = self.total_dim
        else:
            sub_dims = [self.dims_list[i] for i in self.indices]
            self.sub_dim = int(np.prod(sub_dims))

        # Flag to denote whether we are using a random (parametrized) unitary.
        self.random = matrix is None

        if self.random:
            # If no matrix is provided, build a random gate.
            if self.indices is None:
                # Full system: parameter is a total_dim x total_dim complex matrix.
                U_rand = (torch.eye(self.total_dim, device=device, dtype=torch.complex64) +
                          torch.randn((self.total_dim, self.total_dim), device=device, dtype=torch.complex64))
                self.U_param = nn.Parameter(U_rand)
            else:
                # Subsystem: parameter is a sub_dim x sub_dim complex matrix.
                U_rand = (torch.eye(self.sub_dim, device=device, dtype=torch.complex64) +
                          torch.randn((self.sub_dim, self.sub_dim), device=device, dtype=torch.complex64))
                self.U_param = nn.Parameter(U_rand)
        else:
            # Use the provided unitary matrix.
            # Ensure that the provided matrix is a torch tensor on the correct device and in complex64.
            self.M = torch.tensor(matrix, device=device, dtype=torch.complex64)
            # Verify that the matrix has the expected dimensions.
            if self.indices is None:
                if self.M.shape[0] != self.total_dim or self.M.shape[1] != self.total_dim:
                    raise ValueError("Provided matrix dimensions do not match the full Hilbert space dimension.")
            else:
                if self.M.shape[0] != self.sub_dim or self.M.shape[1] != self.sub_dim:
                    raise ValueError("Provided matrix dimensions do not match the product of the targeted qudits' dimensions.")

    def forward(self, x):
        """
        Apply the unitary gate to the state x.

        Args:
            x (torch.Tensor): The input state vector (as a column vector) with dimension equal to
                the product of the individual qudit dimensions.
        Returns:
            torch.Tensor: The transformed state vector.
        """
        if self.indices is None:
            # The unitary acts on the full Hilbert space.
            if self.random:
                # Unitarize the parameter: make it skew-Hermitian then exponentiate.
                U_param = 0.5 * (self.U_param - torch.conj(self.U_param.T))
                U_final = torch.matrix_exp(U_param)
            else:
                U_final = self.M
            return U_final @ x

        # Otherwise, the unitary acts on a subsystem.
        # Permute the state so that the targeted qudits are in the front.
        all_indices = list(range(self.wires))
        target_indices = self.indices
        remaining_indices = [i for i in all_indices if i not in target_indices]
        new_order = target_indices + remaining_indices
        # Inverse permutation for later.
        inv_order = [new_order.index(i) for i in range(self.wires)]

        # Reshape the input state vector into a tensor with shape given by dims_list.
        psi = x.view(*self.dims_list)
        psi_perm = psi.permute(*new_order).contiguous()

        # Flatten into (sub_dim, rest_dim) where sub_dim is for targeted qudits.
        d_rest = self.total_dim // self.sub_dim
        psi_flat = psi_perm.reshape(self.sub_dim, d_rest)

        # Get the unitary acting on the subspace.
        if self.random:
            U_param = self.U_param - torch.conj(self.U_param.T)
            U_sub = torch.matrix_exp(U_param)
        else:
            U_sub = self.M

        # Apply the subspace unitary.
        psi_transformed = U_sub @ psi_flat

        # Reshape back to the permuted tensor shape.
        new_shape = [self.dims_list[i] for i in new_order]
        psi_perm_transformed = psi_transformed.reshape(*new_shape)

        # Inverse permute to restore the original ordering.
        psi_final = psi_perm_transformed.permute(*inv_order).contiguous()

        return psi_final.view(self.total_dim, 1)

    def matrix(self):
        """
        Construct and return the full unitary matrix representation of the gate.

        Returns:
            torch.Tensor: The full unitary matrix.
        """
        if self.indices is None:
            # Full Hilbert space.
            if self.random:
                U_param = 0.5 * (self.U_param - torch.conj(self.U_param.T))
                U_full = torch.matrix_exp(U_param)
            else:
                U_full = self.M
            return U_full

        # For a gate that acts on a subset, we embed the subspace unitary.
        # Determine the unitary acting on the targeted qudits.
        if self.random:
            U_param = self.U_param - torch.conj(self.U_param.T)
            U_sub = torch.matrix_exp(U_param)
        else:
            U_sub = self.M

        # Build the permutation matrix P that maps basis states to a permuted ordering
        # with the targeted qudits first.
        total_dim = self.total_dim
        dims = self.dims_list

        # Create the permutation by mapping each multi-index.
        basis_indices = list(itertools.product(*[range(d) for d in dims]))
        perm = []
        # Define new order: target indices first, then the rest.
        all_indices = list(range(self.wires))
        target_indices = self.indices
        remaining_indices = [i for i in all_indices if i not in target_indices]
        new_order = target_indices + remaining_indices
        for m in basis_indices:
            m = list(m)
            permuted = [m[i] for i in new_order]
            # Convert multi-index back to flat index (using dims of permuted order).
            new_dec = 0
            for idx, d in zip(permuted, [dims[i] for i in new_order]):
                new_dec = new_dec * d + idx
            perm.append(new_dec)
        perm = torch.tensor(perm, dtype=torch.long, device=self.device)

        # Build the permutation matrix P.
        P = torch.zeros((total_dim, total_dim), dtype=torch.complex64, device=self.device)
        for i in range(total_dim):
            P[i, perm[i]] = 1.0

        # Dimensions for the targeted subspace and its complement.
        d_sub = self.sub_dim
        d_rest = total_dim // d_sub
        I_rest = torch.eye(d_rest, dtype=torch.complex64, device=self.device)
        U_embedded = torch.kron(U_sub, I_rest)

        # Embed the unitary: U_full = P^T @ U_embedded @ P.
        U_full = P.T @ U_embedded @ P
        return U_full


class CU(nn.Module):
    r"""
    Controlled-Unitary (CU) Gate for qudits with configurable control branches.

    This gate applies a different unitary on a target subsystem depending on the state of a control qudit.
    The qudits affected by this gate are specified by a single list `index`, where the first element is the
    control qudit and the remaining elements are the target qudits. For a control qudit with dimension \(d_c\),
    the ideal operation is:
    
    .. math::
       \mathrm{CU} = \sum_{k=0}^{d_c-1} |k\rangle\langle k| \otimes U_k.

    With the new flag `control_dim`, the user specifies which control state(s) trigger a nontrivial unitary
    on the target subsystem. For control states *not* in `control_dim` the gate acts as the identity.
    
    There are two modes:
    
    1. **Random (Learnable) Mode (`matrix=None`):**  
       For each control state in `control_dim`, a learnable parameter (of shape \((d_{target}, d_{target})\))
       is defined. Its skew-Hermitian part is exponentiated to yield a unitary. Control states not in
       `control_dim` use the identity block.
       
       For example, with qubits and:
       
       - `index=[0,1]` and `control_dim=[1]`:  
         \(\mathrm{CU} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U.\)
       - `index=[0,1]` and `control_dim=[0]`:  
         \(\mathrm{CU} = |0\rangle\langle 0| \otimes U + |1\rangle\langle 1| \otimes I.\)
         
    2. **Custom Matrix Mode (when `matrix` is provided):**  
       The user must supply the active unitary blocks for the control states in `control_dim`. These can be
       provided as either a list of matrices, a 3D tensor (with first dimension equal to len(`control_dim`)),
       or a 2D tensor that packs the blocks. Blocks for control states not in `control_dim` are taken as identity.

    **Arguments:**
        dim (int or list of int): The dimension of each qudit. If an integer, all qudits are assumed to have that
            dimension; if a list, each element specifies the dimension of the corresponding qudit.
        wires (int): The total number of qudits in the system (used only when `dim` is an integer).
        device (str): The device on which computations are performed.
        index (list of int): A list specifying which qudits the gate acts on. The first element is the control qudit,
            and the remaining elements are the target qudits.
        matrix (torch.Tensor or list or None): If provided, a custom unitary is used for the active branches.
            See discussion above.
        control_dim (int or list of int or None): Specifies the control state(s) for which the gate applies the active
            unitary. For control states not in this list, the identity is used. If not provided, a default is chosen:
            for qubits, the default is `[1]`; for higher dimensions, the default is `[d_control - 1]`.

    **Examples:**
        >>> # Error: index must contain at least one control and one target.
        >>> U = CU(dim=2, wires=3, index=[0])  # Raises an error.
        >>>
        >>> # For a 3-qudit system with control qudit 0 and target qudit 1,
        >>> # using random parameters and default control_dim (for qubits, [1]):
        >>> U = CU(dim=2, wires=3, index=[0,1])
        >>>
        >>> # For a 3-qudit system with control qudit 0 and target qudit 1,
        >>> # but with identity applied on control state |1> (i.e. active unitary on state |0>):
        >>> U = CU(dim=2, wires=3, index=[0,1], control_dim=[0])
        >>>
        >>> # For a 3-qudit system with control qudit 1 and target qudit 0,
        >>> # with custom unitary blocks provided as a list (for control states in control_dim).
        >>> custom_blocks = [torch.eye(2, dtype=torch.complex64), 0.5*torch.eye(2, dtype=torch.complex64)]
        >>> U = CU(dim=2, wires=3, index=[1,0], control_dim=[0,1], matrix=custom_blocks)
    """

    def __init__(self, dim=2, wires=2, device="cpu", index=[0, 1], matrix=None, control_dim=None):
        super(CU, self).__init__()
        # Validate the index list.
        if index is None or len(index) < 2:
            raise ValueError("The 'index' parameter must be a list with at least two elements: one control and at least one target.")
        
        # First element is control; remaining are targets.
        self.index = index
        self.control_index = index[0]
        self.target_index = index[1:]
        
        # Process qudit dimensions.
        if isinstance(dim, int):
            self.dims_list = [dim] * wires
            self.wires = wires
        else:
            self.dims_list = dim
            self.wires = len(dim)
        self.device = device

        # Dimensions of control and target subsystems.
        self.d_control = self.dims_list[self.control_index]
        self.d_target = int(np.prod([self.dims_list[i] for i in self.target_index]))
        self.sub_dim = self.d_control * self.d_target

        # Process control_dim.
        if control_dim is None:
            # Default: for qubits, default to [1]; otherwise, use [d_control - 1].
            self.control_dim = [1] if self.d_control == 2 else [self.d_control - 1]
        elif isinstance(control_dim, int):
            self.control_dim = [control_dim]
        else:
            self.control_dim = list(control_dim)
        for ctrl_state in self.control_dim:
            if ctrl_state < 0 or ctrl_state >= self.d_control:
                raise ValueError("Each element in control_dim must lie in range [0, d_control-1].")
        
        # Determine mode: random (learnable) if matrix is None; otherwise custom.
        if matrix is None:
            self.random = True
            # Learn parameters only for control states in icontrol_dim.
            num_active = len(self.control_dim)
            self.U_blocks_param = nn.Parameter(
                torch.randn(num_active, self.d_target, self.d_target,
                            device=device, dtype=torch.complex64)
            )
        else:
            self.random = False
            # Process the provided custom matrix/blocks.
            # The user must provide one block for each control state in control_dim.
            if isinstance(matrix, list):
                if len(matrix) != len(self.control_dim):
                    raise ValueError("When providing a list, the number of matrices must equal len(control_dim).")
                self.custom_blocks = {}
                for i, ctrl_state in enumerate(self.control_dim):
                    M = torch.tensor(matrix[i], device=device, dtype=torch.complex64)
                    if M.shape != (self.d_target, self.d_target):
                        raise ValueError("Each provided matrix must be of shape (d_target, d_target).")
                    self.custom_blocks[ctrl_state] = M
            elif isinstance(matrix, torch.Tensor):
                if matrix.ndim == 3:
                    if matrix.shape[0] != len(self.control_dim):
                        raise ValueError("For a 3D tensor, the first dimension must equal len(control_dim).")
                    self.custom_blocks = {}
                    for i, ctrl_state in enumerate(self.control_dim):
                        block = matrix[i]
                        if block.shape != (self.d_target, self.d_target):
                            raise ValueError("Each provided block must have shape (d_target, d_target).")
                        self.custom_blocks[ctrl_state] = block
                elif matrix.ndim == 2:
                    expected = len(self.control_dim) * self.d_target
                    if matrix.shape[0] != expected or matrix.shape[1] != expected:
                        raise ValueError("For a 2D tensor, the shape must be (len(control_dim)*d_target, len(control_dim)*d_target).")
                    self.custom_blocks = {}
                    reshaped = matrix.view(len(self.control_dim), self.d_target, self.d_target)
                    for i, ctrl_state in enumerate(self.control_dim):
                        self.custom_blocks[ctrl_state] = reshaped[i]
                else:
                    raise ValueError("Provided matrix must be either a list of matrices, a 3D tensor, or a 2D tensor.")
            else:
                raise ValueError("Provided matrix must be either a list or a torch.Tensor.")
    
    def forward(self, x):
        # Permute qudits so that the controlled subsystem (control and targets) comes first.
        all_indices = list(range(self.wires))
        sub_indices = self.index  # control followed by targets
        remaining = [i for i in all_indices if i not in sub_indices]
        new_order = sub_indices + remaining
        inv_order = [new_order.index(i) for i in range(self.wires)]
        
        # Reshape state into tensor of shape given by dims_list.
        state_tensor = x.view(*self.dims_list)
        psi_perm = state_tensor.permute(*new_order).contiguous()
        d_sub = int(np.prod([self.dims_list[i] for i in sub_indices]))
        d_rem = int(np.prod([self.dims_list[i] for i in remaining])) if remaining else 1
        A = psi_perm.view(d_sub, d_rem)
        # Reshape controlled subsystem: control dimension and target.
        A = A.view(self.d_control, self.d_target, d_rem)
        
        # Build the controlled unitary for the subspace.
        blocks = []
        for k in range(self.d_control):
            if k in self.control_dim:
                # For active control states, use the learned/custom block.
                if self.random:
                    idx = self.control_dim.index(k)
                    param = self.U_blocks_param[idx]
                    U_k = torch.matrix_exp(param - torch.conj(param).T)
                else:
                    U_k = self.custom_blocks[k]
            else:
                # For inactive control states, use identity.
                U_k = torch.eye(self.d_target, dtype=torch.complex64, device=self.device)
            blocks.append(U_k)
        # Assemble block-diagonal matrix with blocks in order for k=0,..., d_control-1.
        U_sub = torch.block_diag(*blocks)  # shape: (d_control * d_target, d_control * d_target)
        
        # Apply the controlled unitary.
        A_sub = A.view(self.sub_dim, d_rem)
        A_new = U_sub @ A_sub
        # Reshape and inverse-permute to original ordering.
        new_shape = ([self.d_control] + [self.dims_list[i] for i in self.target_index] +
                     ([self.dims_list[i] for i in remaining] if remaining else []))
        psi_perm_new = A_new.view(*new_shape)
        psi_final = psi_perm_new.permute(*inv_order).contiguous().view(-1, 1)
        return psi_final

    def matrix(self):
        """
        Retrieve the full unitary matrix representation of the CU gate.
        """
        # Construct the controlled subspace unitary.
        blocks = []
        for k in range(self.d_control):
            if k in self.control_dim:
                if self.random:
                    idx = self.control_dim.index(k)
                    param = self.U_blocks_param[idx]
                    U_k = torch.matrix_exp(param - torch.conj(param).T)
                else:
                    U_k = self.custom_blocks[k]
            else:
                U_k = torch.eye(self.d_target, dtype=torch.complex64, device=self.device)
            blocks.append(U_k)
        U_sub = torch.block_diag(*blocks)
        
        # Determine the remaining subsystem.
        all_indices = list(range(self.wires))
        remaining = [i for i in all_indices if i not in self.index]
        d_rem = int(np.prod([self.dims_list[i] for i in remaining])) if remaining else 1
        I_rem = torch.eye(d_rem, dtype=torch.complex64, device=self.device)
        U_embedded = torch.kron(U_sub, I_rem)
        
        # Build permutation matrix P to restore original qudit ordering.
        total_dim = int(np.prod(self.dims_list))
        new_order = self.index + [i for i in all_indices if i not in self.index]
        new_dims = [self.dims_list[i] for i in new_order]
        basis_indices = list(itertools.product(*[range(d) for d in self.dims_list]))
        perm = []
        for m in basis_indices:
            m = list(m)
            permuted = [m[i] for i in new_order]
            new_dec = 0
            for idx, d in zip(permuted, new_dims):
                new_dec = new_dec * d + idx
            perm.append(new_dec)
        perm = torch.tensor(perm, dtype=torch.long, device=self.device)
        P = torch.zeros((total_dim, total_dim), dtype=torch.complex64, device=self.device)
        for i in range(total_dim):
            P[i, perm[i]] = 1.0
        
        U_full = P.T @ U_embedded @ P
        return U_full
