import numpy as np
import torch
from torch import nn


def kron(matrix1, matrix2, sparse=False):
    """
    Tensor product of dense or sparse matrix
    Input:
        matrix1: first matrix
        matrix2: second matrix
    Output:
        matrix: matrix tensor product
    """

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
        matrix = torch.sparse_coo_tensor(pos, val, size=(D1 * D2, D1 * D2)).to(
            matrix1.device
        )

    elif sparse is False:
        matrix = torch.kron(matrix1, matrix2)

    return matrix


def fidelity(state1, state2):
    F = abs(torch.matmul(torch.conj(state1).T, state2)) ** 2
    return F.real


def delta(i, j):
    if i == j:
        return 1
    return 0


def Sx(j, k, D=2, device="cpu"):
    # 0 <= j < k < D
    S = torch.zeros((D, D), device=device)
    S[j][k] = 1.0
    S[k][j] = 1.0
    return S


def Sy(j, k, D=2, device="cpu"):
    # 0 <= j < k < D
    S = torch.zeros((D, D), device=device, dtype=torch.complex64)
    S[j][k] = -1j
    S[k][j] = 1j
    return S


def Sz(j, D=2, device="cpu"):
    # 1 <= j < D
    f = (2.0 / (j * (j + 1))) ** 0.5
    S = torch.zeros((D, D), device=device)
    for k in range(0, j + 1):
        S[k][k] = f * (-j) ** delta(j, k)
    return S


sigma = [Sx, Sy, Sz]


def base(dim, device="cpu"):
    base = torch.eye(dim, device=device, dtype=torch.complex64).reshape((dim, dim, 1))
    return base


def argmax(x):
    return torch.argmax(x)


def mean(x):
    return torch.mean(x)


def dec2den(j, N, d):
    """
    Convert a global (decimal) index j into its local multi-index representation.

    Args:
        j (int): The global index.
        N (int): The number of qudits.
        d (int or list of int): The dimension(s) of the qudits. If an integer, all qudits have that
            dimension; if a list, each element specifies the dimension of the corresponding qudit.

    Returns:
        list: A list of length N representing the multi-index (local computational basis state).
    """
    den = [0 for _ in range(N)]
    jv = j
    if isinstance(d, int):
        # Uniform dimension: same as before.
        for k in range(N):
            base = d ** (N - 1 - k)
            if jv >= base:
                den[k] = jv // base
                jv = jv - den[k] * base
    else:
        # d is a list: each qudit has its own dimension.
        # Compute the product of dimensions for qudits k+1 through N-1.
        for k in range(N):
            prod = 1
            for dim in d[k + 1 :]:
                prod *= dim
            if prod == 0:
                prod = 1  # avoid division by zero if no subsequent digits
            den[k] = jv // prod
            jv = jv % prod
    return den


def den2dec(local, d):
    """
    Convert a local multi-index (denary representation) into a global (decimal) index.

    Args:
        local (list of int): The local computational basis state as a list of digits.
        d (int or list of int): The dimension(s) of the qudits. If an integer, all qudits have that
            dimension; if a list, each element specifies the dimension of the corresponding qudit.

    Returns:
        int: The global computational basis index.
    """
    N = len(local)
    j = 0
    if isinstance(d, int):
        for k in range(N):
            j += local[k] * (d ** (N - 1 - k))
    else:
        # Compute the place value for each digit.
        for k in range(N):
            prod = 1
            for dim in d[k + 1 :]:
                prod *= dim
            j += local[k] * prod
    return j


def projector(index, dim):
    P = torch.zeros((dim, dim), dtype=torch.complex64)
    P[index][index] = 1.0

    return P


def eye(dim, device="cpu", sparse=False):
    """
    Create a sparse identity matrix
    Input:
        -D: qudit dimension
        -device: cpu or cuda
    Output:
        -eye_sparse: sparse identity matrix
    """
    if sparse is True:
        indices = torch.arange(dim, device=device).repeat(2, 1)
        values = torch.ones(dim, dtype=torch.complex64, device=device)
        M = torch.sparse_coo_tensor(indices, values, (dim, dim))
    else:
        M = torch.eye(dim, dtype=torch.complex64, device=device)

    return M


def zeros(m, n, device="cpu"):
    M = torch.zeros((m, n), device=device)
    return M


def ones(m, n, device="cpu"):
    M = torch.ones((m, n), device=device)
    return M


def cnot_qudits_Position(c, t, n, d, device="cpu"):
    """
    Compute the positions (row, column indices) for the sparse CNOT matrix for multidimensional
    qudits.

    Args:
        c (int): Index of the control qudit.
        t (int): Index of the target qudit.
        n (int): Total number of qudits.
        d (int or list of int): The dimension of each qudit. If an integer, all qudits are assumed
            to have that dimension; if a list, each element specifies the dimension for the
            corresponding qudit.
        device (str): The device for tensor allocation.

    Returns:
        torch.Tensor: A tensor of shape (D, 2) where D is the total Hilbert space dimension. Each
            row contains a pair (row_index, col_index) for a nonzero element of the CNOT matrix.
    """
    # If d is a single integer, assume all qudits have the same dimension.
    if isinstance(d, int):
        dims = [d] * n
    else:
        dims = d
        if len(dims) != n:
            raise ValueError(
                "Length of dimension list must equal the number of qudits (n)."
            )

    # Build the computational basis for n qudits.
    # Create a list of 1D tensors for each qudit's possible values.
    grid = [torch.arange(dim, dtype=torch.float, device=device) for dim in dims]
    # Use torch.meshgrid to create the full grid.
    meshes = torch.meshgrid(*grid, indexing="ij")
    # Stack the meshgrid outputs to form a matrix L of shape (D, n),
    # where D is the total number of basis states.
    L = torch.stack(meshes, dim=-1).reshape(-1, n)

    # Update the target qudit's value: new_target = (old_target + control) mod (dimension of target)
    L[:, t] = (L[:, t] + L[:, c]) % dims[t]

    # Compute the place values for each qudit.
    # For example, for dims = [d0, d1, ..., d_{n-1}], the linear index is:
    #    index = L[0]* (d1*d2*...*d_{n-1}) + L[1]* (d2*...*d_{n-1}) + ... + L[n-1]
    place = []
    prod = 1
    for dim in dims[::-1]:
        place.insert(0, prod)
        prod *= dim
    tt = torch.tensor(place, dtype=torch.float, device=device).reshape(n, 1)

    # Compute the new linear indices for the modified basis states.
    lin = torch.matmul(L, tt)
    D = int(prod)  # total Hilbert space dimension
    col = torch.arange(D, dtype=torch.float, device=device).reshape(D, 1)
    return torch.cat((lin, col), dim=1)


def CNOT_sparse(c, t, d, n, device="cpu"):
    """
    Constructs the sparse matrix representation of the CNOT gate for multidimensional qudits.

    Args:
        c (int): The control qudit index.
        t (int): The target qudit index.
        d (int or list of int): The dimension(s) of the qudits.
        n (int): The total number of qudits.
        device (str): The device for tensor allocation.

    Returns:
        torch.sparse_coo_tensor: The sparse matrix representation of the CNOT gate.
    """
    # If d is a single integer, convert to list.
    if isinstance(d, int):
        dims = [d] * n
    else:
        dims = d
        if len(dims) != n:
            raise ValueError(
                "Length of dimension list must equal the number of qudits (n)."
            )
    # Total Hilbert space dimension: product of all individual dimensions.
    D = int(np.prod(dims))
    indices = cnot_qudits_Position(c, t, n, dims, device=device)
    values = torch.ones(D, device=device)
    eye_sparse = torch.sparse_coo_tensor(
        indices.t(), values, (D, D), dtype=torch.complex64, device=device
    )
    return eye_sparse


def sparse_index_put(M, indices, values, device):
    """
    Updates a sparse tensor M by replacing values at given indices.

    Args:
        M (torch.sparse_coo_tensor): The original sparse matrix.
        indices (torch.Tensor): Indices to update (shape: [2, N] for 2D matrix).
        values (torch.Tensor): Values to set at the given indices (shape: [N]).

    Returns:
        torch.sparse_coo_tensor: Updated sparse tensor.
    """
    assert M.is_sparse, "Input matrix must be a sparse tensor"

    M = M.coalesce()

    # Get old indices and values
    old_indices = M.indices()
    old_values = M.values()

    # Concatenate old and new data
    all_indices = torch.cat([old_indices, indices], dim=1)
    all_values = torch.cat([old_values, values])

    # Convert indices to 1D linear indices for efficient deduplication
    flat_indices = all_indices[0] * M.shape[1] + all_indices[1]  # Works for 2D matrices

    # Keep only the latest occurrences of each index (i.e., remove duplicates)
    unique, inverse = torch.unique(flat_indices, return_inverse=True, sorted=False)
    last_occurrences = torch.zeros_like(unique, dtype=torch.long)
    last_occurrences[inverse] = torch.arange(
        len(flat_indices), device=device
    )  # Stores last occurrence index

    # Gather the latest indices and values
    new_indices = all_indices[:, last_occurrences]
    new_values = all_values[last_occurrences]

    # Create the updated sparse tensor
    return torch.sparse_coo_tensor(new_indices, new_values, M.shape, device=device)
