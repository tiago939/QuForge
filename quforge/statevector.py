import itertools
import functools
import operator
import math
import numpy as np
import sympy as sp
import torch
from math import prod

def State(dits: str, dim=2, device="cpu", sparse: bool = False) -> torch.Tensor:
    """
    Constructs a pure state vector for a system of qudits based on a string input, with optional sparse representation.

    Args:
        dits (str):
            A string of digits separated by non-digits (e.g., '-') indicating each qudit's index.
            Example: '0-2-1' for a three-qudit state |0,2,1>.
        dim (int or list of int):
            Dimension(s) of the qudits. If int, all qudits have that dimension.
            If list, length must match number of qudits.
        device (str):
            'cpu' or 'cuda'.
        sparse (bool):
            If True, returns a sparse COO tensor with only one nonzero entry.

    Returns:
        torch.Tensor: A column vector of shape (N,1), where N = prod(dimensions).
                     Sparse COO if requested, else a dense complex64 tensor.
    """

    # 1. Parse input string into list of qudit indices
    qudit_strs = []
    curr = ""
    for c in dits:
        if c.isdigit():
            curr += c
        elif curr:
            qudit_strs.append(curr);
            curr = ""
    if curr:
        qudit_strs.append(curr)
    # convert to ints
    indices = [int(s) for s in qudit_strs]
    num_qudits = len(indices)

    # 2. Determine dimensions per qudit
    if isinstance(dim, int):
        dims = [dim] * num_qudits
    else:
        dims = list(dim)
        if len(dims) != num_qudits:
            raise ValueError("Length of dim list must equal the number of qudits in input.")

    # 3. Compute linear index in the state vector
    # strides[i] = product of dims[i+1:]
    strides = [prod(dims[i+1:]) for i in range(num_qudits)] + [1]
    # trim the last extra stride
    strides = strides[:num_qudits]
    lin_idx = sum(idx * stride for idx, stride in zip(indices, strides))

    # 4. Total size
    N = prod(dims)

    # 5. Build state vector
    if sparse:
        # single nonzero at (lin_idx,0)
        idx = torch.tensor([[lin_idx], [0]], device=device)
        vals = torch.tensor([1.0], dtype=torch.complex64, device=device)
        state = torch.sparse_coo_tensor(
            idx, vals, (N, 1), dtype=torch.complex64, device=device
        )
        return state.coalesce()
    else:
        state = torch.zeros((N, 1), dtype=torch.complex64, device=device)
        state[lin_idx, 0] = 1.0
        return state


def density_matrix(state, normalize=False):
    """
    Computes the density matrix for a pure state vector.

    Parameters:
        state : torch.Tensor
            A state vector representing a pure state. The state can be given as either a 1D tensor
            of shape (N,) or a 2D column vector of shape (N, 1), where N is the dimension of the Hilbert space.
        normalize : bool, optional
            If True, the function will normalize the state before computing the density matrix.
            Default is False, assuming the state is already normalized.

    Returns:
        rho : torch.Tensor
            The density matrix of the state computed as:

                rho = state * state†

            The output is a 2D tensor of shape (N, N).
    """
    # Ensure state is a column vector.
    if state.ndim == 1:
        state = state.unsqueeze(1)

    # Optionally normalize the state.
    if normalize:
        norm = torch.sum(torch.abs(state) ** 2) ** 0.5
        if norm > 0:
            state = state / norm

    # Compute the density matrix.
    rho = torch.matmul(state, torch.conj(state).T)
    return rho


def partial_trace(state, index=[0], dim=2, wires=None):
    """
    Computes the partial trace over the complementary subsystem.

    Parameters:
      state : torch.Tensor
          The state vector representing the pure state.
      index : list of int
          The indices corresponding to the subsystem to keep.
      dims : int or list of int
          If int, all qudits are assumed to have the same dimension.
          If list, each element is the dimension of the corresponding qudit.
      wires : int, optional
          Total number of qudits. Required if dims is an int.

    Returns:
      rho : torch.Tensor
          The reduced density matrix on the subsystem specified by 'index'.
    """
    # Determine dimensions per qudit.
    if isinstance(dim, int):
        if wires is None:
            raise ValueError("wires must be specified when dims is an int")
        dims_list = [dim] * wires
    else:
        dims_list = dim
        if wires is None:
            wires = len(dims_list)
        elif wires != len(dims_list):
            raise ValueError("wires parameter does not match length of dims list")

    # Validate indices.
    all_indices = torch.arange(wires, device=state.device)
    index_tensor = torch.tensor(index, device=state.device)
    if torch.any(index_tensor < 0) or torch.any(index_tensor >= wires):
        raise ValueError("Invalid index found in the index list.")

    # Identify complementary indices.
    complementary_indices = all_indices[~torch.isin(all_indices, index_tensor)]

    # Sort indices.
    index_sorted = torch.sort(index_tensor).values
    comp_sorted = torch.sort(complementary_indices).values
    new_order = torch.cat((index_sorted, comp_sorted), dim=0)

    # Reshape state to a tensor with one index per qudit.
    state_tensor = state.view(*dims_list)
    # Permute so that the indices we wish to keep come first.
    permuted_state = state_tensor.permute(*new_order).contiguous()

    # Compute the product of dimensions for the subsystems.
    # 'da' corresponds to the subsystem specified by index (kept),
    # 'db' corresponds to the complementary subsystem (to be traced out).
    dims_keep = [dims_list[i] for i in index_sorted.tolist()]
    dims_trace = [dims_list[i] for i in comp_sorted.tolist()]
    da = functools.reduce(operator.mul, dims_keep, 1)
    db = functools.reduce(operator.mul, dims_trace, 1)

    # Flatten the tensor into a matrix with shape (da, db).
    reshaped_state = permuted_state.view(da, db)

    # Compute the reduced density matrix by tracing out the complementary subsystem.
    # For a pure state |psi>, this is given by:
    #    ρ = A A†,  where A is the reshaped state.
    state_conj = reshaped_state.conj()
    rho = torch.matmul(reshaped_state, state_conj.transpose(0, 1))

    return rho


def measure(state=None, index=[0], shots=1, dim=2, wires=None):
    """
    Measures the state on the specified qudits (subsystems) and returns a histogram of the outcomes
    along with the probability distribution of those outcomes.

    Parameters:
        state : torch.Tensor
            The state (a pure state vector) to be measured.
        index : list of int
            The list of qudit indices (wires) on which to perform the measurement.
        shots : int
            The number of measurement repetitions.
        dim : int or list of int
            The dimension(s) of each qudit. If an integer is provided, every qudit is assumed to have
            that same dimension. If a list is provided, its length should equal the number of wires.
        wires : int, optional
            Total number of qudits in the state. Required if 'dim' is provided as an integer.

    Returns:
        histogram : dict
            A dictionary whose keys are the measurement outcomes (as concatenated strings) and
            values are the counts of occurrences over the specified number of shots.
        p : torch.Tensor
            A tensor representing the probability distribution of the measurement outcomes.

    """
    # Determine dimensions per qudit.
    if isinstance(dim, int):
        if wires is None:
            raise ValueError("wires must be specified when dim is an integer")
        dims_list = [dim] * wires
    else:
        dims_list = dim
        if wires is None:
            wires = len(dims_list)
        elif wires != len(dims_list):
            raise ValueError(
                "The wires parameter must match the length of the dim list."
            )

    # Compute the reduced density matrix (partial trace over the complementary subsystem).
    rho = partial_trace(state, index, dim=dims_list, wires=wires)

    # Extract probabilities from the diagonal of the density matrix.
    p = abs(torch.diag(rho))
    p = p / torch.sum(p)

    # Sample measurement outcomes.
    num_outcomes = p.shape[0]
    outcomes = np.arange(num_outcomes)
    positions = np.random.choice(outcomes, p=p.detach().cpu().numpy(), size=shots)

    # For each measured qudit, determine its dimension.
    # Use the sorted order for the indices as in the partial trace.
    measured_indices = sorted(index)
    measured_dims = [dims_list[i] for i in measured_indices]

    # Create keys corresponding to each computational basis state of the measured subsystem.
    # For example, if measured_dims=[2,3], then keys correspond to the product of range(2) and range(3)
    basis_states = list(itertools.product(*[range(d) for d in measured_dims]))
    keys = ["".join(str(s) for s in state_tuple) for state_tuple in basis_states]

    # Initialize histogram for each basis state.
    histogram = {key: 0 for key in keys}
    for pos in positions:
        histogram[keys[pos]] += 1

    return histogram, p


def projector(k, d):
    """
    Returns the projection operator for the k-th computational basis state in a d-dimensional Hilbert space.

    Parameters:
        k : int
            The index of the computational basis state.
        d : int
            The dimension of the Hilbert space.

    Returns:
        P : torch.Tensor
            A (d x d) projection matrix with a 1 at the (k,k) position and 0 elsewhere.
    """
    P = torch.zeros((d, d), dtype=torch.complex64)
    P[k, k] = 1.0
    return P


def project(state, index=[0], dim=2):
    """
    Projects the given pure state onto one of its computational basis components according
    to the measurement on a subset of qudits.

    Parameters:
        state : torch.Tensor
            A pure state vector (1D tensor) of dimension equal to the product of the dimensions
            of the individual qudits.
        index : list of int, optional
            The list of qudit indices (wires) on which the projection (measurement) is performed.
        dim : int or list of int, optional
            If an integer, every qudit is assumed to have the same dimension.
            If a list, it specifies the dimension of each qudit. The product of these dimensions
            must equal state.shape[0].

    Returns:
        new_state : torch.Tensor
            The post-measurement state (normalized) after applying the projection.
        L : tuple
            The multi-index (a tuple of outcomes for each qudit) corresponding to the computational basis
            state that was projected.

    """
    # Determine the dimensions for each qudit.
    if isinstance(dim, int):
        # Compute the number of qudits by taking the logarithm base `dim` of the state size.
        wires = int(round(np.log(state.shape[0]) / np.log(dim)))
        dims_list = [dim] * wires
    else:
        dims_list = dim
        wires = len(dims_list)

    total_dim = np.prod(dims_list)
    if total_dim != state.shape[0]:
        raise ValueError(
            "The product of the individual qudit dimensions must equal the state dimension."
        )

    # Compute the probability distribution from the state amplitudes.
    # (Assumes state is a pure state vector.)
    p_list = [(abs(state[i]) ** 2).item() for i in range(len(state))]
    p_array = np.array(p_list)
    p_array = p_array / np.sum(p_array)

    # Sample one outcome from the probability distribution.
    flat_index = np.random.choice(np.arange(len(state)), p=p_array, size=1)[0]

    # Convert the flat index into a multi-index according to dims_list.
    # This multi-index corresponds to the computational basis state.
    L = np.unravel_index(flat_index, dims_list)

    # Build the projection operator U as a tensor product.
    # For each qudit, if its index is in `index`, apply the corresponding projector, otherwise identity.
    U = torch.eye(1, device=state.device, dtype=state.dtype)
    for i in range(wires):
        if i in index:
            # Use the measured outcome for this qudit.
            proj_op = projector(L[i], dims_list[i]).to(state.device)
            U = torch.kron(U, proj_op)
        else:
            # Use the identity operator for this qudit.
            U = torch.kron(
                U, torch.eye(dims_list[i], device=state.device, dtype=state.dtype)
            )

    # Apply the projection operator and normalize the new state.
    new_state = torch.matmul(U, state)
    norm = torch.sum(torch.abs(new_state) ** 2) ** 0.5
    if norm > 0:
        new_state = new_state / norm

    return new_state, L


def exp_value(state, observable="Z", index=0, dim=2, separate=False):
    """
    Computes the expectation value of an observable on a pure state.
    Supports multidimensional qudits by allowing a single integer or a list of dimensions.

    Parameters:
        state : torch.Tensor
            A pure state vector (shape (N,) or (N,1)), where N = prod(dimensions).
        observable : str, np.ndarray, or torch.Tensor
            If a string, currently supports 'Z' for the generalized Z operator.
            Otherwise, a matrix representation of the observable.
        index : int or list of int
            The qudit(s) on which the observable acts.
        dim : int or list of int
            Dimension(s) of the qudits.
        separate : bool, optional
            If False (default), returns the expectation on the *specified* indices
            (or all combined if index is a list). If True, ignores `index` and returns
            a list of expectation values for each individual wire.

    Returns:
        torch.Tensor or list of torch.Tensor
    """
    # Determine dimensions list and number of wires
    if isinstance(dim, int):
        wires = int(round(np.log(state.numel()) / np.log(dim)))
        dims_list = [dim] * wires
    else:
        dims_list = list(dim)
        wires = len(dims_list)

    # Quick sanity check
    if np.prod(dims_list) != state.numel():
        raise ValueError(
            "Product of dims_list must equal state dimension."
        )

    # If requested, compute each wire separately
    if separate:
        # For each wire i, compute ⟨ψ| Z_i ⊗ I ⊗ … |ψ⟩
        return [
            exp_value(state, observable=observable, index=i, dim=dims_list, separate=False)
            for i in range(wires)
        ]

    # Normalize index argument to a sorted list
    if isinstance(index, int):
        indices = [index]
    else:
        indices = sorted(index)

    # Helper: generalized Z operator
    def generalized_Z(d):
        ω = np.exp(2j * np.pi / d)
        diag = [ω**k for k in range(d)]
        return torch.diag(torch.tensor(diag, dtype=torch.complex64))

    # Build a map of {wire: operator}
    op_measured = {}
    if isinstance(observable, str):
        if observable != "Z":
            raise ValueError(f"Unknown observable '{observable}'")
        for i in indices:
            d = dims_list[i]
            op_measured[i] = generalized_Z(d).to(state.device)
    else:
        M = (
            torch.tensor(observable, dtype=torch.complex64, device=state.device)
            if isinstance(observable, np.ndarray)
            else observable.to(state.device)
        )
        for i in indices:
            op_measured[i] = M

    # Build the full operator via tensor products
    full_op = torch.eye(1, dtype=torch.complex64, device=state.device)
    for wire in range(wires):
        op = op_measured.get(wire, torch.eye(dims_list[wire], dtype=torch.complex64, device=state.device))
        full_op = torch.kron(full_op, op)

    # Ensure column-vector shape
    ψ = state.unsqueeze(1) if state.ndim == 1 else state

    # ⟨ψ| full_op |ψ⟩
    val = ψ.conj().transpose(-2, -1) @ (full_op @ ψ)
    return val.squeeze()


def show(
    state,
    dim=2,
    wires=1,
    tol=1e-12,
    use_floats=False,
    float_precision=4,
    emphasize_index=None,
    suppress_inner=False,
):
    r"""
    Convert a state vector into a LaTeX formatted string, supporting both uniform and multidimensional qudits.

    Parameters:
        state (np.ndarray or torch.Tensor):
            The state vector as a 1D array (or a tensor that can be converted to a 1D array).
            (If using a tensor library, you may need to detach and convert it as shown.)
        dim (int or list of int):
            The dimension of each qudit. If an integer, every qudit is assumed to have that dimension.
            If a list is provided, each element specifies the dimension for the corresponding qudit.
        wires (int):
            The number of qudits in the circuit. If `dim` is a list, wires should equal len(dim).
        tol (float):
            Tolerance below which amplitudes are considered zero.
        use_floats (bool):
            If True, display amplitudes as floating point numbers.
        float_precision (int):
            The number of significant digits for floats.
        emphasize_index (int or None):
            If an integer is provided, the specified qudit (0-indexed) is factored out and emphasized.
            If None, no emphasis is applied.
        suppress_inner (bool):
            If True (and emphasize_index is not None), the non-emphasized part is replaced by a placeholder ket
            (e.g. :math:`|\phi_1\rangle`, :math:`|\phi_2\rangle`, etc.) instead of showing its full expansion.

    Returns:
        str: A string containing the LaTeX representation of the state.

    """

    # Convert state to a 1D numpy array.
    try:
        state = state.detach().cpu().numpy().flatten()
    except AttributeError:
        state = np.array(state).flatten()

    # Determine the dimensions list for each qudit.
    if isinstance(dim, int):
        dims_list = [dim] * wires
    else:
        dims_list = dim
        if wires != len(dims_list):
            raise ValueError(
                "For multidimensional qudits, wires must equal the length of the dim list."
            )

    # total_wires = len(dims_list)

    # Helper function: Convert a flat index to a list of digits given dims_list.
    def index_to_ket(idx, dims):
        digits = []
        for i in range(len(dims)):
            # Compute the product of dimensions for subsequent qudits.
            if i < len(dims) - 1:
                prod = math.prod(dims[i + 1 :])
            else:
                prod = 1
            digit = (idx // prod) % dims[i]
            digits.append(str(digit))
        return digits

    # If no emphasis is requested, do the normal conversion.
    if emphasize_index is None:
        terms = []
        for idx, amp in enumerate(state):
            if np.abs(amp) < tol:
                continue

            # Format amplitude.
            if use_floats:
                re = amp.real
                im = amp.imag
                if np.abs(im) < tol:
                    amp_str = f"{re:.{float_precision}g}"
                elif np.abs(re) < tol:
                    amp_str = f"{im:.{float_precision}g}i"
                else:
                    sign = "+" if im >= 0 else "-"
                    amp_str = (
                        f"{re:.{float_precision}g}{sign}{abs(im):.{float_precision}g}i"
                    )
            else:
                # Use sympy for a simplified symbolic representation.
                if np.abs(amp.imag) < tol:
                    amp_val = amp.real
                else:
                    amp_val = amp
                simplified_amp = sp.nsimplify(amp_val, [sp.sqrt(2)], tolerance=1e-10)
                if isinstance(simplified_amp, sp.Rational):
                    simplified_amp = simplified_amp.limit_denominator(1000)
                amp_str = sp.latex(simplified_amp)
                if amp_str == "1":
                    amp_str = ""
                elif amp_str == "-1":
                    amp_str = "-"

            # Convert the flat index into a ket string using dims_list.
            ket_digits = index_to_ket(idx, dims_list)
            ket_str = "".join(ket_digits)

            terms.append(f"{amp_str}|{ket_str}\\rangle")

        latex_state = " + ".join(terms)
        latex_state = latex_state.replace("+ -", "- ")
        return latex_state

    # If emphasis is requested, factor out the specified qudit.
    else:
        groups = {}  # Group terms by the emphasized digit.
        for idx, amp in enumerate(state):
            if np.abs(amp) < tol:
                continue

            # Format amplitude.
            if use_floats:
                re = amp.real
                im = amp.imag
                if np.abs(im) < tol:
                    amp_str = f"{re:.{float_precision}g}"
                elif np.abs(re) < tol:
                    amp_str = f"{im:.{float_precision}g}i"
                else:
                    sign = "+" if im >= 0 else "-"
                    amp_str = (
                        f"{re:.{float_precision}g}{sign}{abs(im):.{float_precision}g}i"
                    )
            else:
                if np.abs(amp.imag) < tol:
                    amp_val = amp.real
                else:
                    amp_val = amp
                simplified_amp = sp.nsimplify(amp_val, [sp.sqrt(2)], tolerance=1e-10)
                if isinstance(simplified_amp, sp.Rational):
                    simplified_amp = simplified_amp.limit_denominator(1000)
                amp_str = sp.latex(simplified_amp)
                if amp_str == "1":
                    amp_str = ""
                elif amp_str == "-1":
                    amp_str = "-"

            # Build the full ket as a list of digits.
            digits = index_to_ket(idx, dims_list)

            # Extract the digit to emphasize.
            try:
                emph_digit = digits[emphasize_index]
            except IndexError as exc:
                raise ValueError(
                    "emphasize_index is out of range for the number of qudits."
                ) from exc

            # Build the remaining ket (removing the emphasized digit).
            remaining_digits = digits[:emphasize_index] + digits[emphasize_index + 1 :]
            remaining_ket = "".join(remaining_digits)

            # Group terms by the emphasized digit.
            if emph_digit not in groups:
                groups[emph_digit] = []
            groups[emph_digit].append((amp_str, remaining_ket))

        # Build the grouped LaTeX string.
        group_terms = []
        placeholder_counter = 0
        for digit, term_list in groups.items():
            if suppress_inner:
                # Replace the detailed inner sum with a placeholder ket.
                inner = f"|\\phi_{{{placeholder_counter}}}\\rangle"
                placeholder_counter += 1
            else:
                inner_terms = []
                for amp_str, rem_ket in term_list:
                    if rem_ket == "":
                        inner_terms.append(f"{amp_str}")
                    else:
                        inner_terms.append(f"{amp_str}|{rem_ket}\\rangle")
                inner = "(" + " + ".join(inner_terms).replace("+ -", "- ") + ")"
            group_terms.append(f"|{digit}\\rangle\\,{inner}")

        latex_state = " + ".join(group_terms)
        latex_state = latex_state.replace("+ -", "- ")
        return latex_state


def fidelity(x, y):
    F = abs(x.conj().T @ y)**2
    return F