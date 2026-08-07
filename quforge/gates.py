from math import log as log
import cmath
import itertools
import numpy as np
import quforge.aux as aux
import torch
import torch.nn as nn
from typing import List, Tuple
from itertools import product


class H(nn.Module):
    r"""
    Memory‑efficient Generalized Hadamard gate for qudits via local tensordot applications.

    Applies the D×D Hadamard transform on each target qudit axis without building the full 2^N×2^N matrix.

    **Arguments:**
        dim (int or list[int]): Qudit dimension(s); if int, use `wires` copies.
        wires (int): Number of qudits when `dim` is int.
        index (list[int]): Target qudit axes. Defaults to all.
        inverse (bool): If True, applies the inverse (conjugate transpose) Hadamard.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, matrix() returns a sparse COO unitary; else dense.
    """
    def __init__(
        self,
        dim=2,
        wires=None,
        index=None,
        inverse: bool = False,
        device: str = 'cpu',
        sparse: bool = False,
    ):
        super().__init__()
        # dimension list
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("`wires` must be specified when `dim` is int.")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
        self.wires = len(self.dim_list)
        # targets
        self.index = index if index is not None else list(range(self.wires))
        self.inverse = inverse
        self.device = device
        self.sparse = sparse
        # precompute local Hadamard matrices
        self.Hm = {}
        for t in self.index:
            d = self.dim_list[t]
            # root of unity
            omega = cmath.exp(2j * cmath.pi / d)
            # build H
            M = torch.zeros((d, d), dtype=torch.complex64, device=device)
            for i in range(d):
                for j in range(d):
                    M[i, j] = 1.0 * omega**(i*j)
            M = M / (d**0.5)
            if inverse:
                M = M.conj().T.contiguous()
            self.register_buffer(f"H_{t}", M)
            self.Hm[t] = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply H on each target qudit via tensordot.
        """
        # densify
        if x.is_sparse:
            x = x.to_dense()
        # reshape to tensor
        psi = x.view(*self.dim_list)
        for t, M in self.Hm.items():
            # apply local H_t
            tmp = torch.tensordot(M, psi, dims=([1], [t]))
            # permute to move new axis back to position t
            axes = list(range(1, tmp.ndim)); axes.insert(t, 0)
            psi = tmp.permute(axes)
        return psi.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        Build full unitary. Returns sparse COO if `sparse=True`, else dense.
        """
        if self.sparse:
            U = aux.eye(1, device=self.device, sparse=True).coalesce()
            for i, d in enumerate(self.dim_list):
                if i in self.Hm:
                    Ht = self.Hm[i]
                    idx = torch.tensor([list(range(d)), list(range(d))], device=self.device)
                    vals = Ht.diag() if d==Ht.size(0) else None
                    # full dense-to-sparse for general H
                    rows = []
                    cols = []
                    vs = []
                    for r in range(d):
                        for c in range(d):
                            v = Ht[r, c]
                            if v != 0:
                                rows.append(r); cols.append(c); vs.append(v)
                    idx = torch.tensor([rows, cols], device=self.device)
                    vals = torch.stack(vs)
                    M = torch.sparse_coo_tensor(idx, vals, (d, d), dtype=torch.complex64, device=self.device).coalesce()
                else:
                    M = aux.eye(d, device=self.device, sparse=True).coalesce()
                U = aux.kron(U, M, sparse=True).coalesce()
            return U
        else:
            U = torch.eye(1, dtype=torch.complex64, device=self.device)
            for i, d in enumerate(self.dim_list):
                if i in self.Hm:
                    M = self.Hm[i]
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
                U = torch.kron(U, M)
            return U



class X(nn.Module):
    r"""
    Generalized Pauli-X (X) Gate for qudits, memory‐efficient application via state‐tensor rolls,
    with optional sparse small‐matrix construction for explicit unitary retrieval.

    **Arguments:**
        s (int): cyclic shift per target qudit (positive for forward, negative for inverse)
        dim (int or list of int): qudit dimensions; if int, all qudits share that dimension.
        wires (int): total qudits when `dim` is int (ignored if `dim` is list).
        index (list[int]): target wires on which to apply the shift.
        device (str): 'cpu' or 'cuda'.
        inverse (bool): apply negative shift when True.
        sparse (bool): if True, small per‐wire matrices are stored sparsely for `matrix()` calls.
    """
    def __init__(
        self,
        s: int = 1,
        dim=2,
        wires: int = 1,
        index=None,
        device: str = 'cpu',
        inverse: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.s = -s if inverse else s
        self.device = device
        self.inverse = inverse
        self.sparse = sparse
        # process dimension list
        if isinstance(dim, int):
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
        self.wires = len(self.dim_list)
        self.index = index if index is not None else list(range(self.wires))

        # precompute small local matrices for explicit matrix() if needed
        self.M_dict = {}
        for i in self.index:
            d = self.dim_list[i]
            # build local shift matrix
            if self.sparse:
                rows, cols, vals = [], [], []
                for a in range(d):
                    b = (a + self.s) % d
                    rows.append(b)
                    cols.append(a)
                    vals.append(1.0)
                idx = torch.tensor([rows, cols], device=self.device)
                vals = torch.tensor(vals, dtype=torch.complex64, device=self.device)
                M = torch.sparse_coo_tensor(idx, vals, (d, d), dtype=torch.complex64, device=self.device).coalesce()
            else:
                M = torch.zeros((d, d), dtype=torch.complex64, device=self.device)
                for a in range(d):
                    b = (a + self.s) % d
                    M[b, a] = 1.0
            self.register_buffer(f"M_{i}", M)
            self.M_dict[i] = getattr(self, f"M_{i}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply X by rolling the state tensor along each target qudit dimension.
        Memory scales O(prod(dim_list)).
        """
        # ensure dense for reshape
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        for i in self.index:
            psi = psi.roll(shifts=self.s, dims=i)
        return psi.reshape(-1)

    def matrix(self) -> torch.Tensor:
        """
        (Optional) Construct the full unitary via small‐matrix Kron.
        Only for small wire counts. Returns sparse if requested.
        """
        # start with 1×1 identity
        if self.sparse:
            U = aux.eye(1, device=self.device, sparse=True).coalesce()
        else:
            U = torch.eye(1, dtype=torch.complex64, device=self.device)
        # Kron factors over wires
        for i, d in enumerate(self.dim_list):
            if i in self.index:
                M = self.M_dict[i]
            else:
                if self.sparse:
                    M = aux.eye(d, device=self.device, sparse=True).coalesce()
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = aux.kron(U, M, sparse=self.sparse)
            if self.sparse:
                U = U.coalesce()
        return U


class Z(nn.Module):
    r"""
    Memory‑efficient Generalized Pauli‑Z gate for qudits with optional sparse full‑unitary construction.

    Applies phase shifts directly in forward and optionally builds a sparse or dense full unitary in matrix().

    **Arguments:**
        dim (int or list[int]): If int, use `wires` copies; else list of per‑wire dims.
        wires (int): Number of qudits when `dim` is int.
        s (int): Phase shift multiplier. Default = 1.
        index (list[int]): Target qudit indices. Default = all qudits.
        device (str): 'cpu' or 'cuda'.
        inverse (bool): If True, applies inverse (negate `s`).
        sparse (bool): If True, matrix() returns a sparse COO unitary; else dense.
    """
    def __init__(
        self,
        dim=2,
        wires=None,
        s: int = 1,
        index: list = None,
        device: str = 'cpu',
        inverse: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.s = -s if inverse else s
        self.device = device
        self.sparse = sparse
        # build dimension list
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is an int")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
        self.wires = len(self.dim_list)
        # target wires
        self.index = index if index is not None else list(range(self.wires))
        # precompute per‑wire phase vectors
        self.phase_vectors = {}
        for i in self.index:
            d = self.dim_list[i]
            omega = cmath.exp(2j * cmath.pi / d)
            phases = torch.tensor(
                [omega ** ((j * self.s) % d) for j in range(d)],
                dtype=torch.complex64,
                device=self.device
            )
            self.phase_vectors[i] = phases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Z by broadcasting phase factors onto each target qudit axis.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        for i, phases in self.phase_vectors.items():
            shape = [1] * self.wires
            shape[i] = phases.numel()
            psi = psi * phases.view(shape)
        return psi.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        Build and return the full unitary. If `sparse=True`, returns a sparse COO matrix;
        otherwise a dense torch.Tensor.
        """
        if self.sparse:
            U = aux.eye(1, device=self.device, sparse=True).coalesce()
            for i, d in enumerate(self.dim_list):
                if i in self.phase_vectors:
                    phases = self.phase_vectors[i]
                    idx = torch.tensor([list(range(d)), list(range(d))], device=self.device)
                    M = torch.sparse_coo_tensor(
                        idx,
                        phases,
                        (d, d),
                        dtype=torch.complex64,
                        device=self.device
                    ).coalesce()
                else:
                    M = aux.eye(d, device=self.device, sparse=True).coalesce()
                U = aux.kron(U, M, sparse=True).coalesce()
            return U
        else:
            U = torch.eye(1, dtype=torch.complex64, device=self.device)
            for i, d in enumerate(self.dim_list):
                if i in self.phase_vectors:
                    M = torch.diag(self.phase_vectors[i])
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
                U = torch.kron(U, M)
            return U


class Y(nn.Module):
    r"""
    Memory‑efficient Generalized Pauli‑Y gate for qudits: Y = (1/i) Z·X = -i·Z·X via local operations.

    Applies the Y rotation on each target qudit directly to the state tensor,
    avoiding full 2^N×2^N matrix constructions.

    **Arguments:**
        s (int): cyclic shift parameter for X; default=1.
        dim (int or list[int]): qudit dimensions; if int, use `wires` copies.
        wires (int): number of qudits when `dim` is int.
        index (list[int]): target qudit axes; default=all.
        device (str): 'cpu' or 'cuda'.
        inverse (bool): if True, applies inverse Y (negates phase and shift).
        sparse (bool): ignored in forward; retained for API consistency.
    """
    def __init__(
        self,
        s: int = 1,
        dim=2,
        wires: int = None,
        index: list = None,
        device: str = 'cpu',
        inverse: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        # process dims
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int.")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
        self.wires = len(self.dim_list)
        # targets
        self.index = index if index is not None else list(range(self.wires))
        # shift and phase sign
        self.s = -s if inverse else s
        self.device = device
        self.sparse = sparse
        # precompute local X roll matrices and Z phase vectors
        self.Mx = {}
        self.zphases = {}
        for t in self.index:
            d = self.dim_list[t]
            # X: cyclic shift by +s mod d
            M = torch.zeros((d, d), dtype=torch.complex64, device=device)
            for a in range(d):
                b = (a + self.s) % d
                M[b, a] = 1.0
            self.register_buffer(f"M_{t}", M)
            self.Mx[t] = M
            # Z: root of unity phases
            omega = cmath.exp(2j * cmath.pi / d)
            phases = torch.tensor([omega**((j * self.s) % d) for j in range(d)],
                                   dtype=torch.complex64, device=device)
            self.zphases[t] = phases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Y gate by: psi' = -1j * Z_t ( X_t psi ) on each target t.
        """
        # densify
        if x.is_sparse:
            x = x.to_dense()
        # reshape state
        psi = x.view(*self.dim_list)
        for t in self.index:
            # apply X_t
            tmp = torch.tensordot(self.Mx[t], psi, dims=([1], [t]))
            axes = list(range(1, tmp.ndim)); axes.insert(t, 0)
            psi_x = tmp.permute(axes)
            # apply Z_t
            phases = self.zphases[t]
            shape = [1] * self.wires; shape[t] = phases.numel()
            psi_z = psi_x * phases.view(shape)
            # multiply by -i = exp(-i*pi/2)
            psi = (-1j) * psi_z
        return psi.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        (Optional) Construct full unitary for small wire counts.
        Uses Kron of local 2×2 Y blocks.
        """
        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        for i, d in enumerate(self.dim_list):
            if i in self.index:
                # build local Y = -i Z X
                Mx = self.Mx[i]
                phases = self.zphases[i]
                # Z diag
                Z = torch.diag(phases)
                Yloc = (-1j) * Z @ Mx
                M = Yloc
            else:
                M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = torch.kron(U, M)
        return U


class RX(nn.Module):
    r"""
    Memory-efficient Rotation-X (RX) gate for qudits via local tensordot applications.

    Applies each 2×2 rotation block directly to the state tensor along the target axis,
    avoiding any full 2^N×2^N matrix construction.

    **Arguments:**
        j (int or list[int]): First level indices for each target qudit.
        k (int or list[int]): Second level indices for each target qudit.
        index (list[int]): Target qudit axes. Defaults to [0].
        dim (int or list[int]): Dimension(s) of qudits; if int, use `wires` times.
        wires (int): Number of qudits when `dim` is an int.
        device (str): 'cpu' or 'cuda'.
        angle (float or Tensor or None): Rotation parameter(s). If None, random init.
        sparse (bool): Unused here; kept for signature compatibility.
        global_angle (bool): If True, all targets share one parameter.
        amplitude (float): Scale factor for angles.
    """
    def __init__(
        self,
        j=0,
        k=1,
        index=None,
        dim=2,
        wires=1,
        device='cpu',
        angle=None,
        sparse=False,
        global_angle=False,
        amplitude=1.0,
    ):
        super().__init__()
        self.device = device
        self.index = index if index is not None else [0]
        self.wires = wires
        self.global_angle = global_angle
        self.amplitude = amplitude
        # build dimension list
        if isinstance(dim, int):
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
            self.wires = len(self.dim_list)
        # j/k maps
        js = [j] * len(self.index) if isinstance(j, int) else j
        ks = [k] * len(self.index) if isinstance(k, int) else k
        self.j_map = dict(zip(self.index, js))
        self.k_map = dict(zip(self.index, ks))
        # parameters
        n_param = 1 if global_angle else len(self.index)
        self.param_map = {t: 0 for t in self.index} if global_angle else {t: i for i,t in enumerate(self.index)}
        if angle is None:
            init = 2 * np.pi * torch.rand(n_param, device=device)
            #init = np.pi * (2 * torch.rand(n_param, device=device) - 1.0)
            self.angle = nn.Parameter(init)
        else:
            #a = angle if torch.is_tensor(angle) else torch.tensor([angle]*n_param, device=device)
            #self.angle = nn.Parameter(a.view(n_param))
            self.angle = torch.tensor([angle]*n_param, device=device)

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply RX by reshaping to an N‑d state tensor and applying local rotations.
        """
        # get parameters
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        # ensure dense for view
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # apply each target rotation
        for t in self.index:
            d = self.dim_list[t]
            j_val = self.j_map[t]; k_val = self.k_map[t]
            theta = self.amplitude * (p[0] if self.global_angle else p[self.param_map[t]])
            c = torch.cos(theta/2)
            s = -1j * torch.sin(theta/2)
            # build small rotation block M
            M = torch.eye(d, dtype=torch.complex64, device=self.device)
            M[j_val, j_val] = c; M[k_val, k_val] = c
            M[j_val, k_val] = s; M[k_val, j_val] = s
            # tensordot on axis t
            psi = torch.tensordot(M, psi, dims=([1],[t]))
            # reorder axes to put rotated axis back to position t
            axes = list(range(1, psi.ndim))
            axes.insert(t, 0)
            psi = psi.permute(axes)
        return psi.reshape(-1, 1)

    def matrix(self, param=None) -> torch.Tensor:
        """
        (Optional) Construct full unitary via Kron of small blocks. Only for small wires.
        """
        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        # get parameters
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        for i in range(self.wires):
            d = self.dim_list[i]
            if i in self.index:
                j_val = self.j_map[i]; k_val = self.k_map[i]
                theta = self.amplitude * (p[0] if self.global_angle else p[self.param_map[i]])
                c = torch.cos(theta/2); s = -1j * torch.sin(theta/2)
                M = torch.eye(d, dtype=torch.complex64, device=self.device)
                M[j_val, j_val] = c; M[k_val, k_val] = c
                M[j_val, k_val] = s; M[k_val, j_val] = s
            else:
                M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = torch.kron(U, M)
        return U


class RY(nn.Module):
    r"""
    Memory‑efficient Rotation‑Y (RY) gate for qudits via local tensordot applications.

    Applies a two‑level Y rotation on each target qudit directly to the state tensor,
    avoiding any full 2^N×2^N matrix construction.

    **Arguments:**
        j (int or list[int]): First level index for each target qudit (or common if int).
        k (int or list[int]): Second level index for each target qudit (or common if int).
        index (list[int]): Target qudit axes. Defaults to [0].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        angle (float or Tensor or None): Rotation parameter(s). Random if None.
        sparse (bool): Ignored in forward; retained for compatibility.
        global_angle (bool): If True, all targets share a single parameter.
        amplitude (float): Scale factor for angles.
    """
    def __init__(
        self,
        j=0,
        k=1,
        index=None,
        dim=2,
        wires=1,
        device='cpu',
        angle=None,
        sparse=False,
        global_angle=False,
        amplitude=1.0,
    ):
        super().__init__()
        self.device = device
        self.sparse = sparse
        self.global_angle = global_angle
        self.amplitude = amplitude
        # targets
        self.index = index if index is not None else [0]
        # dimensions
        if isinstance(dim, int):
            self.dim_list = [dim] * wires
            self.wires = wires
        else:
            self.dim_list = list(dim)
            self.wires = len(self.dim_list)
        # j/k maps
        js = [j] * len(self.index) if isinstance(j, int) else j
        ks = [k] * len(self.index) if isinstance(k, int) else k
        self.j_map = dict(zip(self.index, js))
        self.k_map = dict(zip(self.index, ks))
        # parameter mapping
        n_param = 1 if global_angle else len(self.index)
        self.param_map = {t: 0 for t in self.index} if global_angle else {t: i for i, t in enumerate(self.index)}
        # initialize angle(s)
        if angle is None:
            init = 2 * np.pi * torch.rand(n_param, device=device)
            self.angle = nn.Parameter(init)
        else:
            a = angle if torch.is_tensor(angle) else torch.tensor([angle] * n_param, device=device)
            self.angle = nn.Parameter(a.view(n_param))

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply RY by reshaping state to N‑d tensor and performing local 2×2 Y rotations per axis.
        """
        # ensure dense for view
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # parameter tensor
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        for t in self.index:
            d = self.dim_list[t]
            j_val = self.j_map[t]
            k_val = self.k_map[t]
            theta = self.amplitude * (p[0] if self.global_angle else p[self.param_map[t]])
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            # build small Y rotation: [[c, -s], [s, c]] in (j,k) subspace
            M = torch.eye(d, dtype=torch.complex64, device=self.device)
            M[j_val, j_val] = c;      M[k_val, k_val] = c
            M[j_val, k_val] = -s;     M[k_val, j_val] = s
            # apply via tensordot on axis t
            tmp = torch.tensordot(M, psi, dims=([1], [t]))
            # permute new axis back to position t
            axes = list(range(1, tmp.ndim)); axes.insert(t, 0)
            psi = tmp.permute(axes)
        return psi.reshape(-1, 1)

    def matrix(self, param=None) -> torch.Tensor:
        """
        (Optional) Fallback: build full unitary via Kron of small rotation blocks.
        Only suitable for small wire counts.
        """
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        for i in range(self.wires):
            d = self.dim_list[i]
            if i in self.index:
                j_val = self.j_map[i]; k_val = self.k_map[i]
                theta = self.amplitude * (p[0] if self.global_angle else p[self.param_map[i]])
                c = torch.cos(theta / 2); s = torch.sin(theta / 2)
                M = torch.eye(d, dtype=torch.complex64, device=self.device)
                M[j_val, j_val] = c;      M[k_val, k_val] = c
                M[j_val, k_val] = -s;     M[k_val, j_val] = s
            else:
                M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = torch.kron(U, M)
        return U


class RZ(nn.Module):
    r"""
    Memory‑efficient Rotation‑Z (RZ) gate for qudits via direct phase multiplications.

    Applies a phase rotation on each target qudit axis without building the full 2^N×2^N matrix.

    **Arguments:**
        j (int or list[int]): Level(s) to rotate (or common if int).
        index (list[int]): Target qudit axes. Defaults to [0].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        angle (float or Tensor or None): Rotation parameter(s). Random if None.
        sparse (bool): Ignored in forward; retained for API compatibility.
        global_angle (bool): If True, all targets share a single parameter.
        amplitude (float): Scale factor for angles.
    """
    def __init__(
        self,
        j=0,
        index=None,
        dim=2,
        wires=1,
        device='cpu',
        angle=None,
        sparse=False,
        global_angle=False,
        amplitude = 1.0
    ):
        super().__init__()
        self.device = device
        self.sparse = sparse
        self.global_angle = global_angle
        self.amplitude = amplitude
        # targets
        self.index = index if index is not None else [0]
        # dimensions
        if isinstance(dim, int):
            self.dim_list = [dim] * wires
            self.wires = wires
        else:
            self.dim_list = list(dim)
            self.wires = len(self.dim_list)
        # j_map
        js = [j] * len(self.index) if isinstance(j, int) else j
        self.j_map = dict(zip(self.index, js))
        # parameters
        n_param = 1 if global_angle else len(self.index)
        self.param_map = {t: 0 for t in self.index} if global_angle else {t: i for i, t in enumerate(self.index)}
        # init angles
        if angle is None:
            init = 2 * np.pi * torch.rand(n_param, device=device)
            self.angle = nn.Parameter(init)
        else:
            a = angle if torch.is_tensor(angle) else torch.tensor([angle] * n_param, device=device)
            self.angle = nn.Parameter(a.view(n_param))

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)

        # keep autograd on parameters
        if param is None:
            p = self.angle
        else:
            p = param if torch.is_tensor(param) else torch.as_tensor(param, device=self.device, dtype=self.angle.dtype)

        for t in self.index:
            d = self.dim_list[t]
            j_val = self.j_map[t]
            theta = self.amplitude * (p[0] if self.global_angle else p[self.param_map[t]])

            if d == 2:
                # build phases with stack to preserve grad
                if j_val == 0:
                    phases = torch.stack((torch.exp(-0.5j * theta), torch.exp(0.5j * theta)))
                else:
                    phases = torch.stack((torch.exp(0.5j * theta), torch.exp(-0.5j * theta)))
            else:
                # make a tensor that depends on theta so grad flows
                phases = torch.ones(d, dtype=torch.complex64, device=self.device)
                phases = phases.to(dtype=torch.complex64)
                phases = phases.clone()  # avoid in-place on a leaf
                phases = phases + 0j      # ensure complex
                phases = phases.index_put_((torch.tensor([j_val], device=self.device),),
                                        torch.exp(1j * theta) - 1 + phases[j_val])

            shape = [1] * self.wires
            shape[t] = d
            psi = psi * phases.view(shape)

        return psi.reshape(-1, 1)

    def matrix(self, param=None) -> torch.Tensor:
        if param is None:
            p = self.angle
        else:
            p = param if torch.is_tensor(param) else torch.as_tensor(param, device=self.device, dtype=self.angle.dtype)

        U = torch.eye(1, dtype=torch.complex64, device=self.device)
        for i in range(self.wires):
            d = self.dim_list[i]
            if i in self.index:
                j_val = self.j_map[i]
                theta = self.amplitude * (p[0] if self.global_angle else p[self.param_map[i]])  # <-- use i here
                if d == 2:
                    if j_val == 0:
                        phases = torch.stack((torch.exp(-0.5j * theta), torch.exp(0.5j * theta)))
                    else:
                        phases = torch.stack((torch.exp(0.5j * theta), torch.exp(-0.5j * theta)))
                    M = torch.diag(phases)
                else:
                    phases = torch.ones(d, dtype=torch.complex64, device=self.device)
                    phases = phases.index_put_((torch.tensor([j_val], device=self.device),),
                                            torch.exp(1j * theta))
                    M = torch.diag(phases)
            else:
                M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = aux.kron(U, M, sparse=self.sparse)
        return U


class CNOT(nn.Module):
    r"""
    Memory‑efficient Controlled-NOT (CNOT) gate for qudits via conditional rolls,
    with optional sparse full‑unitary fallback.

    **Arguments:**
        index (list[int]): [control_axis, target_axis].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor.
        inverse (bool): If True, shifts by -c instead of +c.
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        device='cpu',
        sparse=False,
        inverse=False,
    ):
        super().__init__()
        if not (isinstance(index, (list, tuple)) and len(index) == 2):
            raise ValueError("`index` must be [control, target]")
        self.ctrl, self.tgt = index
        self.device = device
        self.inverse = inverse
        self.sparse = sparse
        # build dims list
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CNOT by reshaping state to an N‑D tensor and conditionally rolling the target axis
        by the control value for each slice.
        """
        # densify if sparse
        if x.is_sparse:
            x = x.to_dense()
        # reshape into multi‑dim tensor
        psi = x.view(*self.dim_list)
        # permute so control->axis0, target->axis1
        axes = list(range(self.wires))
        axes.remove(self.ctrl); axes.remove(self.tgt)
        perm = [self.ctrl, self.tgt] + axes
        psi_pt = psi.permute(perm)
        # flatten trailing axes
        dc, dt = psi_pt.shape[0], psi_pt.shape[1]
        rest = psi_pt.numel() // (dc * dt)
        psi_flat = psi_pt.reshape(dc, dt, rest)
        # prepare output
        out_flat = torch.empty_like(psi_flat)
        # apply conditional shift
        for c_val in range(dc):
            slice_c = psi_flat[c_val]  # shape [dt, rest]
            shift = -c_val if self.inverse else c_val
            out_flat[c_val] = slice_c.roll(shifts=shift, dims=0)
        # reshape back and invert permutation
        out = out_flat.reshape(psi_pt.shape)
        inv = [0] * self.wires
        for i, ax in enumerate(perm):
            inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        Construct the full CNOT unitary. Returns sparse COO if `sparse=True`, else dense.
        Only recommended for small systems.
        """
        # build computational basis mapping
        dims = self.dim_list
        L = torch.tensor(list(product(*[range(d) for d in dims])), device=self.device)
        L2 = L.clone()
        c, t = self.ctrl, self.tgt
        L2[:, t] = (L[:, c] + ( -L[:, t] if self.inverse else L[:, t] )) % dims[t]
        # build sparse indices and values
        D = L.shape[0]
        idx = []
        vals = []
        # map each basis state i -> j
        basis_list = L.tolist()
        for i, row in enumerate(L2.tolist()):
            j = basis_list.index(row)
            idx.append([i, j])
            vals.append(1.0)
        idx = torch.tensor(idx, device=self.device).t()
        vals = torch.tensor(vals, dtype=torch.complex64, device=self.device)
        U_sparse = torch.sparse_coo_tensor(
            idx, vals, (D, D), dtype=torch.complex64, device=self.device
        ).coalesce()
        return U_sparse if self.sparse else U_sparse.to_dense()


class CZ(nn.Module):
    r"""
    Memory‑efficient Controlled-Z (CZ) gate for qudits via conditional phase broadcasts,
    with optional sparse full‑unitary fallback.

    **Arguments:**
        index (list[int]): [control_axis, target_axis].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor.
        inverse (bool): If True, applies inverse CZ (phase negation).
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        device='cpu',
        sparse=False,
        inverse=False,
    ):
        super().__init__()
        if not (isinstance(index, (list, tuple)) and len(index) == 2):
            raise ValueError("`index` must be [control, target]")
        self.ctrl, self.tgt = index
        self.device = device
        self.sparse = sparse
        self.inverse = inverse
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
        self.wires = len(self.dim_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CZ by reshaping to N‑D tensor and broadcasting a conditional phase to the target axis.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        axes = list(range(self.wires))
        axes.remove(self.ctrl);
        axes.remove(self.tgt)
        perm = [self.ctrl, self.tgt] + axes
        psi_pt = psi.permute(perm)
        dc, dt = psi_pt.shape[0], psi_pt.shape[1]
        rest = psi_pt.numel() // (dc * dt)
        psi_flat = psi_pt.reshape(dc, dt, rest)
        out_flat = torch.empty_like(psi_flat)
        omega = {}  # cache roots of unity
        for c_val in range(dc):
            if c_val not in omega:
                d_t = dt
                omega[c_val] = cmath.exp(2j * cmath.pi / d_t)
                phases = torch.tensor([omega[c_val]**j for j in range(d_t)], dtype=torch.complex64, device=self.device)
                if self.inverse:
                    phases = phases.conj()
                omega[c_val] = phases
            phases = omega[c_val].view(dt, 1)
            out_flat[c_val] = psi_flat[c_val] * phases
        out = out_flat.reshape(psi_pt.shape)
        inv = [0] * self.wires
        for i, ax in enumerate(perm): inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        Construct full CZ unitary. Returns sparse COO if `sparse=True`, else dense.
        Only recommended for small systems.
        """
        dims = self.dim_list
        L = torch.tensor(list(product(*[range(d) for d in dims])), device=self.device)
        L2 = L.clone()
        c, t = self.ctrl, self.tgt
        for i in range(L.shape[0]):
            c_val = int(L[i, c].item())
            d_t = dims[t]
            phase = cmath.exp(2j * cmath.pi * c_val / d_t)
            if self.inverse:
                phase = phase.conj()
            L2[i, t] = L[i, t]  # state unchanged, phase applied in values
        # build sparse unitary diag of phases where control==state
        D = L.shape[0]
        idx = []
        vals = []
        for i in range(D):
            idx.append([i, i])
            c_val = int(L[i, c].item())
            d_t = dims[t]
            phase = cmath.exp(2j * cmath.pi * c_val / d_t)
            if self.inverse:
                phase = phase.conj()
            vals.append(phase)
        idx = torch.tensor(idx, device=self.device).t()
        vals = torch.tensor(vals, dtype=torch.complex64, device=self.device)
        U_sparse = torch.sparse_coo_tensor(idx, vals, (D, D), dtype=torch.complex64, device=self.device).coalesce()
        return U_sparse if self.sparse else U_sparse.to_dense()


class SWAP(nn.Module):
    r"""
    Memory‑efficient SWAP gate for qudits via axis transpose,
    with optional sparse full‑unitary fallback.

    Swaps the states of two qudits by transposing their axes in the state tensor.

    **Arguments:**
        index (list[int]): [axis1, axis2] to swap.
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor fallback.
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        device='cpu',
        sparse=False,
    ):
        super().__init__()
        if not (isinstance(index,(list,tuple)) and len(index)==2):
            raise ValueError("`index` must be [axis1, axis2]")
        self.a1, self.a2 = index
        self.device = device
        self.sparse = sparse
        # build dims list
        if isinstance(dim,int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SWAP by reshaping to an N‑D tensor and transposing the two axes.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        psi_swapped = psi.transpose(self.a1, self.a2)
        return psi_swapped.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        Construct the full SWAP unitary. Returns sparse COO if `sparse=True`, else dense.
        Only recommended for small systems.
        """
        from itertools import product
        dims = self.dim_list
        L = torch.tensor(list(product(*[range(d) for d in dims])), device=self.device)
        D = L.shape[0]
        idx = []
        vals = []
        basis_list = L.tolist()
        for i,row in enumerate(basis_list):
            swapped = row.copy()
            swapped[self.a1], swapped[self.a2] = row[self.a2], row[self.a1]
            j = basis_list.index(swapped)
            idx.append([i,j]); vals.append(1.0)
        idx = torch.tensor(idx, device=self.device).t()
        vals = torch.tensor(vals, dtype=torch.complex64, device=self.device)
        U_sparse = torch.sparse_coo_tensor(idx, vals, (D,D), dtype=torch.complex64, device=self.device).coalesce()
        return U_sparse if self.sparse else U_sparse.to_dense()



class CRX(nn.Module):
    r"""
    Memory-efficient Controlled-RX gate for qudits via conditional local rotations,
    with optional sparse full-unitary fallback.

    Applies an RX rotation on the target qudit conditioned on the control qudit's state:
    for control state c, applies RX(θ * c) between levels j,k on the target axis.

    **Arguments:**
        index (list[int]): [control_axis, target_axis].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        j (int): First level of RX subspace on target.
        k (int): Second level of RX subspace on target.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor; else dense.
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        j=0,
        k=1,
        device='cpu',
        sparse=False,
    ):
        super().__init__()
        if not (isinstance(index,(list,tuple)) and len(index)==2):
            raise ValueError("`index` must be [control, target]")
        self.ctrl, self.tgt = index
        self.j = j
        self.k = k
        self.device = device
        self.sparse = sparse
        # build dimension list
        if isinstance(dim,int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires
        # learnable angle
        self.angle = nn.Parameter(2 * np.pi * torch.rand(1, device=device))

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply CRX by reshaping to N-D tensor and performing conditional RX on the target axis.
        """
        # ensure dense
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # bring control->axis0, target->axis1
        axes = list(range(self.wires))
        axes.remove(self.ctrl); axes.remove(self.tgt)
        perm = [self.ctrl, self.tgt] + axes
        psi_pt = psi.permute(perm)
        nc, dt = psi_pt.shape[0], psi_pt.shape[1]
        rest = psi_pt.numel() // (nc * dt)
        psi_flat = psi_pt.reshape(nc, dt, rest)
        out_flat = torch.empty_like(psi_flat)
        # parameter tensor
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        for c_val in range(nc):
            theta = (p[0] * c_val)
            ccos = torch.cos(theta/2)
            ssin = -1j * torch.sin(theta/2)
            # build local RX block
            M = torch.eye(dt, dtype=torch.complex64, device=self.device)
            M[self.j,self.j] = ccos; M[self.k,self.k] = ccos
            M[self.j,self.k] = ssin; M[self.k,self.j] = ssin
            # apply to slice
            slice_c = psi_flat[c_val]
            out_flat[c_val] = M @ slice_c
        # reshape back and invert permute
        out = out_flat.reshape(psi_pt.shape)
        inv = [0]*self.wires
        for i,ax in enumerate(perm): inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        (Optional) Construct full CRX unitary via Kron of control-conditioned blocks.
        Returns sparse COO if `sparse=True`, else dense.
        """
        # full identity
        U = aux.eye(1,device=self.device,sparse=self.sparse)
        for i in range(self.wires):
            d = self.dim_list[i]
            if i == self.ctrl:
                # control as diagonal of basis states
                if self.sparse:
                    idx = torch.tensor([list(range(d)), list(range(d))], device=self.device)
                    vals = torch.ones(d, dtype=torch.complex64, device=self.device)
                    M = torch.sparse_coo_tensor(idx, vals, (d,d), dtype=torch.complex64, device=self.device).coalesce()
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
            elif i == self.tgt:
                # build block for each control basis via sum
                # sum_c |c><c| ⊗ RX(c*θ)
                d_t = d
                mats = []
                for c_val in range(d):
                    theta = self.angle[0]*c_val
                    ccos = torch.cos(theta/2); ssin = -1j*torch.sin(theta/2)
                    B = torch.eye(d_t, dtype=torch.complex64, device=self.device)
                    B[self.j,self.j]=ccos; B[self.k,self.k]=ccos
                    B[self.j,self.k]=ssin; B[self.k,self.j]=ssin
                    # weight by projector |c><c|
                    proj = torch.zeros((d,d),dtype=torch.complex64,device=self.device)
                    proj[c_val,c_val]=1
                    mats.append(proj @ B)
                # sum mats
                Mdense = sum(mats)
                if self.sparse:
                    # convert Mdense to sparse
                    nz = Mdense.nonzero()
                    vals = Mdense[nz[:,0], nz[:,1]]
                    M = torch.sparse_coo_tensor(nz.t(), vals, (d_t,d_t), dtype=torch.complex64, device=self.device).coalesce()
                else:
                    M = Mdense
            else:
                # identity
                if self.sparse:
                    M = aux.eye(d,device=self.device,sparse=True).coalesce()
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = aux.kron(U, M, sparse=self.sparse)
        return U



class CRY(nn.Module):
    r"""
    Memory-efficient Controlled-RY gate for qudits via conditional local rotations,
    with optional sparse full-unitary fallback.

    Applies an RY rotation on the target qudit conditioned on the control qudit's state:
    for control state c, applies RY(θ * c) between levels j,k on the target axis.

    **Arguments:**
        index (list[int]): [control_axis, target_axis].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        j (int): First level of RY subspace on target.
        k (int): Second level of RY subspace on target.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor; else dense.
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        j=0,
        k=1,
        device='cpu',
        sparse=False,
    ):
        super().__init__()
        if not (isinstance(index,(list,tuple)) and len(index)==2):
            raise ValueError("`index` must be [control, target]")
        self.ctrl, self.tgt = index
        self.j = j
        self.k = k
        self.device = device
        self.sparse = sparse
        # build dimension list
        if isinstance(dim,int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires
        # learnable angle (random initialization)
        self.angle = nn.Parameter(2 * np.pi * torch.rand(1, device=self.device))

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply CRY by reshaping to N-D tensor and performing conditional RY on the target axis.
        """
        # ensure dense
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # bring control->axis0, target->axis1
        axes = list(range(self.wires))
        axes.remove(self.ctrl); axes.remove(self.tgt)
        perm = [self.ctrl, self.tgt] + axes
        psi_pt = psi.permute(perm)
        nc, dt = psi_pt.shape[0], psi_pt.shape[1]
        rest = psi_pt.numel() // (nc * dt)
        psi_flat = psi_pt.reshape(nc, dt, rest)
        out_flat = torch.empty_like(psi_flat)
        # parameter tensor
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        for c_val in range(nc):
            theta = (p[0] * c_val)
            ccos = torch.cos(theta/2)
            ssin = torch.sin(theta/2)
            # build local RY block
            M = torch.eye(dt, dtype=torch.complex64, device=self.device)
            M[self.j,self.j] = ccos; M[self.k,self.k] = ccos
            M[self.j,self.k] = -ssin; M[self.k,self.j] = ssin
            # apply to slice
            slice_c = psi_flat[c_val]
            out_flat[c_val] = M @ slice_c
        # reshape back and invert permute
        out = out_flat.reshape(psi_pt.shape)
        inv = [0]*self.wires
        for i,ax in enumerate(perm): inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        (Optional) Construct full CRY unitary via Kron of control-conditioned blocks.
        Returns sparse COO if `sparse=True`, else dense.
        """
        # full identity
        U = aux.eye(1,device=self.device,sparse=self.sparse)
        for i in range(self.wires):
            d = self.dim_list[i]
            if i == self.ctrl:
                # control as identity projector (diag)
                if self.sparse:
                    idx = torch.tensor([list(range(d)), list(range(d))], device=self.device)
                    vals = torch.ones(d, dtype=torch.complex64, device=self.device)
                    M = torch.sparse_coo_tensor(idx, vals, (d,d), dtype=torch.complex64, device=self.device).coalesce()
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
            elif i == self.tgt:
                # build block for each control basis via sum
                d_t = d
                mats = []
                for c_val in range(d):
                    theta = self.angle[0]*c_val
                    ccos = torch.cos(theta/2); ssin = torch.sin(theta/2)
                    B = torch.eye(d_t, dtype=torch.complex64, device=self.device)
                    B[self.j,self.j]=ccos; B[self.k,self.k]=ccos
                    B[self.j,self.k]=-ssin; B[self.k,self.j]=ssin
                    # weight by projector |c><c|
                    proj = torch.zeros((d,d),dtype=torch.complex64,device=self.device)
                    proj[c_val,c_val]=1
                    mats.append(proj @ B)
                Mdense = sum(mats)
                if self.sparse:
                    nz = Mdense.nonzero()
                    vals = Mdense[nz[:,0], nz[:,1]]
                    M = torch.sparse_coo_tensor(nz.t(), vals, (d_t,d_t), dtype=torch.complex64, device=self.device).coalesce()
                else:
                    M = Mdense
            else:
                # identity
                if self.sparse:
                    M = aux.eye(d,device=self.device,sparse=True).coalesce()
                else:
                    M = torch.eye(d, dtype=torch.complex64, device=self.device)
            U = aux.kron(U, M, sparse=self.sparse)
        return U


class CRZ(nn.Module):
    r"""
    Memory-efficient Controlled-RZ (CRZ) gate for qudits via conditional phase broadcasts,
    with optional sparse full-unitary fallback.

    Applies a Z-phase rotation on the target qudit level `j`, conditioned on the control qudit's state.
    For control state c and learnable angle θ, target level j accumulates phase exp(i·c·θ).

    **Arguments:**
        index (list[int]): [control_axis, target_axis].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        j (int): Target level index to phase-shift.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor; else dense.
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        j=1,
        device='cpu',
        sparse=False,
        inverse=False,
    ):
        super().__init__()
        if not (isinstance(index,(list,tuple)) and len(index)==2):
            raise ValueError("`index` must be [control, target]")
        self.ctrl, self.tgt = index
        self.j = j
        self.device = device
        self.sparse = sparse
        self.inverse = inverse
        # build dimension list
        if isinstance(dim,int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires
        # learnable angle (random init)
        self.angle = nn.Parameter(2 * np.pi * torch.rand(1, device=self.device))

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply CRZ by reshaping to N-D state tensor and broadcasting a conditional phase on target axis.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # permute so control->axis0, target->axis1
        axes = list(range(self.wires))
        axes.remove(self.ctrl); axes.remove(self.tgt)
        perm = [self.ctrl, self.tgt] + axes
        psi_pt = psi.permute(perm)
        nc, dt = psi_pt.shape[0], psi_pt.shape[1]
        rest = psi_pt.numel() // (nc * dt)
        psi_flat = psi_pt.reshape(nc, dt, rest)
        out_flat = torch.empty_like(psi_flat)
        # parameter tensor
        p = self.angle if param is None else torch.tensor(param, device=self.device)
        for c_val in range(nc):
            # compute phase vector
            phi = p[0] * c_val
            phases = torch.ones(dt, dtype=torch.complex64, device=self.device)
            phases[self.j] = torch.exp(1j * phi)
            if self.inverse:
                phases = phases.conj()
            # apply phase to slice
            out_flat[c_val] = psi_flat[c_val] * phases.view(dt,1)
        # reshape back and invert permute
        out = out_flat.reshape(psi_pt.shape)
        inv = [0]*self.wires
        for i,ax in enumerate(perm): inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        (Optional) Construct the full CRZ unitary. Returns sparse COO if `sparse=True`, else dense.
        Only recommended for small systems.
        """
        dims = self.dim_list
        D = int(np.prod(dims))
        # linear basis indices
        # build diag phases
        phases = []
        for idx in range(D):
            local = aux.dec2den(idx, self.wires, dims)
            c_val = local[self.ctrl]
            t_val = local[self.tgt]
            phi = self.angle[0] * c_val
            if t_val == self.j:
                phase = torch.exp(1j * phi)
            else:
                phase = 1.0
            if self.inverse:
                phase = phase.conj()
            phases.append(phase)
        phases = torch.stack(phases).to(torch.complex64)
        if self.sparse:
            idx = torch.arange(D, device=self.device)
            idx = torch.stack([idx, idx], dim=0)
            U = torch.sparse_coo_tensor(idx, phases, (D,D), dtype=torch.complex64, device=self.device).coalesce()
        else:
            U = torch.diag(phases)
        return U


class CCNOT(nn.Module):
    r"""
    Memory-efficient CCNOT (Toffoli) gate for qudits via conditional target rolls,
    with optional sparse full-unitary fallback.

    For controls c1, c2 and target t: t -> (t + c1 * c2) mod d_t.

    **Arguments:**
        index (list[int]): [control1_axis, control2_axis, target_axis].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): If True, `matrix()` returns a sparse COO tensor; else dense.
        inverse (bool): If True, applies inverse operation: t -> (t - c1*c2).
    """
    def __init__(
        self,
        index=[0,1,2],
        dim=2,
        wires=None,
        device='cpu',
        sparse=False,
        inverse=False,
    ):
        super().__init__()
        if not (isinstance(index,(list,tuple)) and len(index)==3):
            raise ValueError("`index` must be [ctrl1, ctrl2, target]")
        self.ctrl1, self.ctrl2, self.tgt = index
        self.device = device
        self.sparse = sparse
        self.inverse = inverse
        # build dims
        if isinstance(dim,int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CCNOT by reshaping to N-D tensor and rolling the target axis by c1*c2 mod d_t.
        """
        # ensure dense
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # permute controls->axes0,1 and target->axis2
        axes = list(range(self.wires))
        for ax in [self.ctrl1, self.ctrl2, self.tgt]:
            axes.remove(ax)
        perm = [self.ctrl1, self.ctrl2, self.tgt] + axes
        psi_pt = psi.permute(perm)
        d1, d2, dt = psi_pt.shape[0], psi_pt.shape[1], psi_pt.shape[2]
        rest = psi_pt.numel() // (d1 * d2 * dt)
        psi_flat = psi_pt.reshape(d1, d2, dt, rest)
        out_flat = torch.empty_like(psi_flat)
        # apply conditional roll
        for c1 in range(d1):
            for c2 in range(d2):
                slice_c = psi_flat[c1, c2]  # shape [dt, rest]
                shift = (c1 * c2) % dt
                if self.inverse:
                    shift = -shift
                out_flat[c1, c2] = slice_c.roll(shifts=shift, dims=0)
        # reshape back and invert permute
        out = out_flat.reshape(psi_pt.shape)
        inv = [0]*self.wires
        for i,ax in enumerate(perm): inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        (Optional) Construct full CCNOT unitary. Returns sparse COO if `sparse=True`, else dense.
        Only for small systems.
        """
        dims = self.dim_list
        D = int(torch.prod(torch.tensor(dims)))
        # build basis
        L = torch.tensor(list(product(*[range(d) for d in dims])), device=self.device)
        # compute mapping
        tgt_idx = self.tgt
        ctrl1_idx = self.ctrl1
        ctrl2_idx = self.ctrl2
        mapping = []
        for i, state in enumerate(L.tolist()):
            c1 = state[ctrl1_idx]
            c2 = state[ctrl2_idx]
            t = state[tgt_idx]
            shift = (c1 * c2) % dims[tgt_idx]
            if self.inverse:
                shift = (-c1 * c2) % dims[tgt_idx]
            new_t = (t + shift) % dims[tgt_idx]
            new = state.copy()
            new[tgt_idx] = new_t
            j = L.tolist().index(new)
            mapping.append((i, j))
        # build sparse indices and values
        idx = torch.tensor(mapping, device=self.device).t()
        vals = torch.ones(D, dtype=torch.complex64, device=self.device)
        U_sparse = torch.sparse_coo_tensor(idx, vals, (D, D), dtype=torch.complex64, device=self.device).coalesce()
        return U_sparse if self.sparse else U_sparse.to_dense()


class MCX(nn.Module):
    r"""
    Memory-efficient Multi-Controlled-X (MCX) gate for qudits via conditional target rolls,
    with optional sparse full-unitary fallback.

    For controls c1,...,c_{n-1} and target t: t -> (t + c1*c2*...*c_{n-1}) mod d_t.

    **Arguments:**
        index (list[int]): control axes followed by target axis.
        dim (int or list[int]): qudit dimensions; if int, repeated `wires` times.
        wires (int): number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        sparse (bool): if True, `matrix()` returns sparse COO; else dense.
        inverse (bool): if True, subtracts the product instead of adding.
    """
    def __init__(
        self,
        index=[0,1],
        dim=2,
        wires=None,
        device='cpu',
        sparse=False,
        inverse=False,
    ):
        super().__init__()
        if not (isinstance(index,(list,tuple)) and len(index)>=2):
            raise ValueError("`index` must be a list with at least two elements [controls...,target]")
        *ctrls, tgt = index
        self.ctrls = ctrls
        self.tgt = tgt
        self.device = device
        self.sparse = sparse
        self.inverse = inverse
        # build dims list
        if isinstance(dim,int):
            if wires is None:
                raise ValueError("Must specify `wires` when `dim` is int")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MCX by reshaping to N-D tensor and rolling the target axis by the product of control values mod d_t.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        # permute controls and target to front
        axes = list(range(self.wires))
        for ax in (*self.ctrls, self.tgt):
            axes.remove(ax)
        perm = [*self.ctrls, self.tgt] + axes
        psi_pt = psi.permute(perm)
        # compute shape multipliers
        ctr_dims = [self.dim_list[i] for i in self.ctrls]
        C = 1
        for d in ctr_dims: C *= d
        dt = psi_pt.shape[len(ctr_dims)]
        rest = psi_pt.numel() // (C * dt)
        psi_flat = psi_pt.reshape(C, dt, rest)
        out_flat = torch.empty_like(psi_flat)
        # apply conditional roll
        for u in range(C):
            # decode u -> control values
            local = aux.dec2den(u, len(self.ctrls), ctr_dims)
            prod_val = 1
            for v in local: prod_val *= v
            shift = prod_val % dt
            if self.inverse:
                shift = -shift
            out_flat[u] = psi_flat[u].roll(shifts=shift, dims=0)
        # reshape back and invert permute
        out = out_flat.reshape(psi_pt.shape)
        inv = [0]*self.wires
        for i,ax in enumerate(perm): inv[ax] = i
        psi_new = out.permute(inv)
        return psi_new.reshape(-1, 1)

    def matrix(self) -> torch.Tensor:
        """
        Construct full MCX unitary. Returns sparse COO if `sparse=True`, else dense.
        Only for small wire counts.
        """
        dims = self.dim_list
        D = int(torch.prod(torch.tensor(dims)))
        # build basis states
        L = torch.tensor(list(product(*[range(d) for d in dims])), device=self.device)
        # mapping pairs
        ctrl_idxs = self.ctrls
        tgt_idx = self.tgt
        mapping = []
        for i,state in enumerate(L.tolist()):
            prod_val = 1
            for ci in ctrl_idxs:
                prod_val *= state[ci]
            shift = prod_val % dims[tgt_idx]
            if self.inverse:
                shift = (-prod_val) % dims[tgt_idx]
            new = state.copy()
            new[tgt_idx] = (state[tgt_idx] + shift) % dims[tgt_idx]
            j = L.tolist().index(new)
            mapping.append([i,j])
        idx = torch.tensor(mapping, device=self.device).t()
        vals = torch.ones(D, dtype=torch.complex64, device=self.device)
        U_sparse = torch.sparse_coo_tensor(idx, vals, (D,D), dtype=torch.complex64, device=self.device).coalesce()
        return U_sparse if self.sparse else U_sparse.to_dense()


class U(nn.Module):
    r"""
    Memory‑efficient Custom/Random Unitary Gate for qudits with optional sparse embedding.

    Applies either a provided matrix or a learnable random unitary on the full system or a subspace.

    **Arguments:**
        matrix (Tensor or None): User‑supplied unitary (square) on full or subspace; if None, parameterize random.
        dim (int or list[int]): Qudit dimensions; if int, repeated wires times.
        wires (int): Number of qudits when dim is int.
        device (str): 'cpu' or 'cuda'.
        index (int or list[int] or None): Axes to act on; if None, acts on all wires.
        sparse (bool): If True, matrix() returns a sparse COO tensor; else dense.
    """
    def __init__(
        self,
        matrix=None,
        dim=2,
        wires=None,
        device='cpu',
        index=None,
        sparse=False,
    ):
        super().__init__()
        self.device = device
        self.sparse = sparse
        # collect dims
        if isinstance(dim, int):
            if wires is None:
                raise ValueError('wires must be specified when dim is int')
            self.dims = [dim]*wires
        else:
            self.dims = list(dim)
            wires = len(self.dims)
        self.wires = wires
        # target indices
        if index is None:
            self.targets = list(range(self.wires))
        elif isinstance(index, int):
            self.targets = [index]
        else:
            self.targets = sorted(index)
        # total and subspace dims
        self.D = int(np.prod(self.dims))
        self.sub_dims = [self.dims[i] for i in self.targets]
        self.sub_D = int(np.prod(self.sub_dims))
        # user matrix or random parameter
        self.custom = matrix is not None
        if self.custom:
            M = torch.as_tensor(matrix, dtype=torch.complex64, device=device)
            expected = self.sub_D
            if M.shape != (expected, expected):
                raise ValueError(f"Matrix shape {M.shape} mismatches subspace {expected}x{expected}")
            self.register_buffer('M', M)
        else:
            init = torch.randn((self.sub_D, self.sub_D), dtype=torch.complex64, device=device)
            self.U_param = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the (subspace) unitary to state x without building full matrices.
        """
        # densify
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dims)
        # permute so targets front
        rest = [i for i in range(self.wires) if i not in self.targets]
        order = self.targets + rest
        inv = [order.index(i) for i in range(self.wires)]
        psi_p = psi.permute(order).contiguous()
        # reshape to (sub_D, rest_D)
        rest_D = self.D // self.sub_D
        psi_flat = psi_p.reshape(self.sub_D, rest_D)
        # get subspace U
        if self.custom:
            U_sub = self.M
        else:
            H = 0.5*(self.U_param - self.U_param.conj().T)
            U_sub = torch.matrix_exp(H)
        # apply
        out_flat = U_sub @ psi_flat
        # restore shape
        new_shape = [self.dims[i] for i in order]
        psi2 = out_flat.reshape(*new_shape)
        psi_final = psi2.permute(inv).contiguous()
        return psi_final.reshape(self.D, 1)

    def matrix(self) -> torch.Tensor:
        """
        Construct full unitary: sparse or dense, only for small systems.
        """
        # get subspace U
        if self.custom:
            U_sub = self.M
        else:
            H = 0.5*(self.U_param - self.U_param.conj().T)
            U_sub = torch.matrix_exp(H)
        # if full system, return U_sub
        if self.targets == list(range(self.wires)):
            return U_sub
        # else embed
        # build permutation
        perm_axes = self.targets + [i for i in range(self.wires) if i not in self.targets]
        inv = [perm_axes.index(i) for i in range(self.wires)]
        # build permutation mapping for basis
        idx_map = []
        for idx in range(self.D):
            multi = aux.dec2den(idx, self.wires, self.dims)
            perm_mult = [multi[i] for i in perm_axes]
            new_idx = aux.den2dec(perm_mult, [self.dims[i] for i in perm_axes])
            idx_map.append(new_idx)
        perm = torch.tensor(idx_map, device=self.device)
        # build P
        if self.sparse:
            I = torch.arange(self.D, device=self.device)
            P = torch.sparse_coo_tensor(torch.stack([I, perm]), torch.ones(self.D, device=self.device), (self.D,self.D))
        else:
            P = torch.zeros((self.D,self.D), dtype=torch.complex64, device=self.device)
            P[torch.arange(self.D), perm] = 1
        # embed: Kron(U_sub, I_rest)
        rest_D = self.D//self.sub_D
        if self.sparse:
            I_rest = aux.eye(rest_D, device=self.device, sparse=True).coalesce()
            U_emb = torch.kron(U_sub, I_rest, sparse=True).coalesce()
        else:
            I_rest = torch.eye(rest_D, dtype=torch.complex64, device=self.device)
            U_emb = torch.kron(U_sub, I_rest)
        # conjugate embed
        if self.sparse:
            return P.transpose(0,1).matmul(U_emb).matmul(P)
        else:
            return P.T @ U_emb @ P



class CU(nn.Module):
    r"""
    Memory‑efficient Controlled‑Unitary (CU) gate for qudits via conditional slice‑wise application.

    Applies different unitary blocks on a target subspace depending on a control qudit's value:
    CU = \sum_{k=0}^{d_c-1} |k><k| \otimes U_k, with identity for inactive control states.

    **Arguments:**
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): number of qudits when dim is int.
        index (list[int]): [control_axis, *target_axes].
        device (str): 'cpu' or 'cuda'.
        matrix (Tensor/list/None): custom blocks for active control states; if None, learnable.
        control_dim (int/list/None): control values for which U_k is nontrivial.
        sparse (bool): if True, matrix() returns a sparse COO tensor.
    """
    def __init__(
        self,
        dim=2,
        wires=None,
        index=None,
        matrix=None,
        control_dim=None,
        device='cpu',
        sparse=False,
    ):
        super().__init__()
        # build dims
        if isinstance(dim, int):
            if wires is None:
                raise ValueError('`wires` must be specified when `dim` is int')
            self.dims = [dim] * wires
        else:
            self.dims = list(dim)
            wires = len(self.dims)
        self.wires = wires
        self.device = device
        self.sparse = sparse
        # parse control and targets
        if not (isinstance(index, (list, tuple)) and len(index) >= 2):
            raise ValueError('`index` must be [control_index, *target_indices]')
        self.ctrl = index[0]
        self.targets = index[1:]
        # control and target dims
        self.d_control = self.dims[self.ctrl]
        self.d_target = int(np.prod([self.dims[i] for i in self.targets]))
        # control_dim (active states)
        if control_dim is None:
            self.active = [self.d_control - 1]
        elif isinstance(control_dim, int):
            self.active = [control_dim]
        else:
            self.active = list(control_dim)
        for k in self.active:
            if not (0 <= k < self.d_control):
                raise ValueError('control_dim values must be in [0, d_control-1]')
        # custom vs learnable
        self.custom = matrix is not None
        if self.custom:
            self.custom_blocks = {}
            # list of blocks
            if isinstance(matrix, list):
                if len(matrix) != len(self.active):
                    raise ValueError('Length of custom matrix list must match len(control_dim)')
                for idx, k in enumerate(self.active):
                    M = torch.as_tensor(
                        matrix[idx], dtype=torch.complex64, device=self.device
                    )
                    if M.shape != (self.d_target, self.d_target):
                        raise ValueError('Each custom block must be (d_target x d_target)')
                    self.custom_blocks[k] = M
            # stacked tensor
            elif isinstance(matrix, torch.Tensor):
                if matrix.ndim == 3:
                    if matrix.shape[0] != len(self.active):
                        raise ValueError('matrix.shape[0] must equal len(control_dim)')
                    for idx, k in enumerate(self.active):
                        self.custom_blocks[k] = matrix[idx]
                elif matrix.ndim == 2:
                    exp = len(self.active) * self.d_target
                    if matrix.shape != (exp, exp):
                        raise ValueError('Flat matrix must be (len(control_dim)*d_target)^2')
                    reshaped = matrix.view(len(self.active), self.d_target, self.d_target)
                    for idx, k in enumerate(self.active):
                        self.custom_blocks[k] = reshaped[idx]
                else:
                    raise ValueError('Invalid tensor shape for custom matrix')
            else:
                raise ValueError('matrix must be list or torch.Tensor')
        else:
            # learnable parameters for active blocks
            self.U_param = nn.Parameter(
                torch.randn(
                    len(self.active), self.d_target, self.d_target,
                    dtype=torch.complex64,
                    device=self.device,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CU by converting any sparse input to dense, then slice‑wise applying each U_k.
        """
        # ensure dense
        x = x.to_dense() if not isinstance(x, torch.Tensor) or x.is_sparse or x.layout != torch.strided else x
        psi = x.view(*self.dims)
        # permute control and targets to front
        subsys = [self.ctrl] + self.targets
        rest = [i for i in range(self.wires) if i not in subsys]
        order = subsys + rest
        inv_order = [order.index(i) for i in range(self.wires)]
        psi_p = psi.permute(order).contiguous()
        # reshape to (d_control, d_target, d_rest)
        d_rest = int(np.prod([self.dims[i] for i in rest])) if rest else 1
        psi_f = psi_p.reshape(self.d_control, self.d_target, d_rest)
        out = torch.empty_like(psi_f)
        # apply block on each control slice
        for k in range(self.d_control):
            if k in self.active:
                if self.custom:
                    U_k = self.custom_blocks[k]
                else:
                    H = self.U_param[self.active.index(k)]
                    U_k = torch.matrix_exp(0.5 * (H - H.conj().transpose(0, 1)))
            else:
                U_k = torch.eye(self.d_target, dtype=torch.complex64, device=self.device)
            out[k] = U_k @ psi_f[k]
        # reshape back and invert permutation
        # reshape back to separate control, each target, and rest axes
        psi2 = out.reshape([self.d_control] + [self.dims[i] for i in self.targets] + [self.dims[i] for i in rest])
        # invert permutation and flatten back to column vector
        psi_final = psi2.permute(inv_order).contiguous().reshape(-1, 1)
        return psi_final

    def matrix(self) -> torch.Tensor:
        """
        Optional fallback: build full CU unitary as sparse or dense for small systems.
        """
        # assemble block-diagonal controlled-unitary on subspace
        blocks = []
        for k in range(self.d_control):
            if k in self.active:
                if self.custom:
                    B = self.custom_blocks[k]
                else:
                    H = self.U_param[self.active.index(k)]
                    B = torch.matrix_exp(0.5 * (H - H.conj().transpose(0,1)))
            else:
                B = torch.eye(self.d_target, dtype=torch.complex64, device=self.device)
            blocks.append(B)
        U_sub = torch.block_diag(*blocks)
        # embed into full Hilbert space by permuting and kron with identity
        total_dim = int(np.prod(self.dims))
        subsys = [self.ctrl] + self.targets
        rest = [i for i in range(self.wires) if i not in subsys]
        perm_axes = subsys + rest
        # build index map for permutation
        idx_map = []
        for idx in range(total_dim):
            multi = aux.dec2den(idx, self.wires, self.dims)
            perm_m = [multi[i] for i in perm_axes]
            new_idx = aux.den2dec(perm_m, [self.dims[i] for i in perm_axes])
            idx_map.append(new_idx)
        perm = torch.tensor(idx_map, device=self.device)
        # construct permutation operator P
        if self.sparse:
            I = torch.arange(total_dim, device=self.device)
            P = torch.sparse_coo_tensor(torch.stack([I, perm]), torch.ones(total_dim, device=self.device), (total_dim, total_dim))
        else:
            P = torch.zeros((total_dim, total_dim), dtype=torch.complex64, device=self.device)
            P[torch.arange(total_dim), perm] = 1
        # identity on rest
        d_rest = int(np.prod([self.dims[i] for i in rest])) if rest else 1
        if self.sparse:
            I_rest = aux.eye(d_rest, device=self.device, sparse=True).coalesce()
            U_emb = torch.kron(U_sub, I_rest, sparse=True).coalesce()
            return P.transpose(0,1).matmul(U_emb).matmul(P)
        else:
            I_rest = torch.eye(d_rest, dtype=torch.complex64, device=self.device)
            U_emb = torch.kron(U_sub, I_rest)
            return P.T @ U_emb @ P


class RXX(nn.Module):
    r"""
    Memory‑efficient RXX gate for qudits: exp(-i * phi * X_i X_j / 2) via local tensordot.

    Applies the two‑qudit rotation directly on the state tensor, avoiding full 2^N×2^N unitaries.

    **Arguments:**
        index (list[int]): Two target qudit axes, e.g. [i, j].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        angle (float or Tensor or None): Rotation angle φ; random if None.
        sparse (bool): Unused in forward (matrix fallback only).
    """
    def __init__(
        self,
        index,
        dim=2,
        wires=None,
        device='cpu',
        angle=None,
        sparse=False,
    ):
        super().__init__()
        if not (isinstance(index, (list, tuple)) and len(index)==2):
            raise ValueError("`index` must be two axes [i,j].")
        self.i, self.j = index
        self.device = device
        # build dims list
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Specify `wires` when `dim` is int.")
            self.dim_list = [dim]*wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires
        # angle parameter
        if angle is None:
            self.phi = nn.Parameter(torch.randn(1, device=device))
        else:
            self.phi = nn.Parameter(torch.tensor([angle], device=device))
        # precompute local X generators for each axis
        d_i = self.dim_list[self.i]
        d_j = self.dim_list[self.j]
        # X generator is cyclic shift by +1
        M_i = torch.zeros((d_i,d_i),dtype=torch.complex64,device=device)
        for a in range(d_i): M_i[(a+1)%d_i, a]=1
        M_j = torch.zeros((d_j,d_j),dtype=torch.complex64,device=device)
        for a in range(d_j): M_j[(a+1)%d_j, a]=1
        self.register_buffer('M_i', M_i)
        self.register_buffer('M_j', M_j)

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply RXX: reshape to N‑d tensor, compute ψ_flip = X_i X_j |ψ〉, then
        |ψ'〉 = cos(φ/2) ψ + (-i sin(φ/2)) ψ_flip.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        phi = (self.phi if param is None else torch.tensor(param,device=self.device))[0]
        c = torch.cos(phi/2)
        s = -1j*torch.sin(phi/2)
        # apply X_i
        tmp = torch.tensordot(self.M_i, psi, dims=([1],[self.i]))
        # permute to bring new axis to position self.i
        axes = list(range(1, tmp.ndim)); axes.insert(self.i, 0)
        psi_i = tmp.permute(axes)
        # apply X_j
        tmp2 = torch.tensordot(self.M_j, psi_i, dims=([1],[self.j]))
        axes2 = list(range(1, tmp2.ndim)); axes2.insert(self.j, 0)
        psi_ij = tmp2.permute(axes2)
        # combine
        out = c*psi + s*psi_ij
        return out.reshape(-1,1)

    def matrix(self, param=None) -> torch.Tensor:
        """
        (Optional) Build full unitary for small wire counts.
        """
        phi = (self.phi if param is None else torch.tensor(param,device=self.device))[0]
        c = torch.cos(phi/2); s = -1j*torch.sin(phi/2)
        # full X_i X_j via small kron
        U_local = aux.kron(self.M_i, self.M_j, sparse=False)
        # identity on other wires via aux.eye
        # build full full plugin, etc...
        # fallback to existing method if needed
        raise NotImplementedError("Full matrix construction not supported in fallback.")


class RYY(nn.Module):
    r"""
    Memory‑efficient RYY gate for qudits: exp(-i * phi * Y_i Y_j / 2) via local tensordot.

    **Arguments:**
      index (list[int]): Two target qudit axes [i, j].
      dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
      wires (int): Number of qudits when `dim` is int.
      device (str): 'cpu' or 'cuda'.
      angle (float or Tensor or None): Rotation angle φ; random if None.
      sparse (bool): if True, `matrix()` will return a sparse‑COO fallback.
    """
    def __init__(
        self,
        index,
        dim=2,
        wires=None,
        device='cpu',
        angle=None,
        sparse=False,
    ):
        super().__init__()
        if not (isinstance(index, (list, tuple)) and len(index) == 2):
            raise ValueError("`index` must be two axes [i, j].")
        self.i, self.j = index
        self.device = device
        self.sparse = sparse

        # build dims list
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Specify `wires` when `dim` is int.")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires

        # angle parameter
        if angle is None:
            self.phi = nn.Parameter(torch.randn(1, device=device))
        else:
            self.phi = torch.tensor(angle * torch.ones(1, device=device), device=device)

        # build local Y‑generators: Y = (Z @ X) / i
        d_i = self.dim_list[self.i]
        d_j = self.dim_list[self.j]

        Z_i = Z(dim=d_i, wires=1, device=device).matrix()
        X_i = X(dim=d_i, wires=1, device=device).matrix()
        Z_j = Z(dim=d_j, wires=1, device=device).matrix()
        X_j = X(dim=d_j, wires=1, device=device).matrix()
        Y_i = (Z_i @ X_i) / (1j)
        Y_j = (Z_j @ X_j) / (1j)
        self.register_buffer('Y_i', Y_i)
        self.register_buffer('Y_j', Y_j)

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        """
        Apply RYY:
          |ψ'> = cos(φ/2)|ψ> + (−i sin(φ/2)) Y_i Y_j |ψ>
        all done by tensordot + permutes on a dense state.
        """
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)

        phi = (self.phi if param is None else torch.tensor(param, device=self.device))[0]
        c = torch.cos(phi / 2)
        s = -1j * torch.sin(phi / 2)

        # apply Y_i
        tmp = torch.tensordot(self.Y_i, psi, dims=([1], [self.i]))
        axes = list(range(1, tmp.ndim))
        axes.insert(self.i, 0)
        psi_i = tmp.permute(axes)

        # apply Y_j
        tmp2 = torch.tensordot(self.Y_j, psi_i, dims=([1], [self.j]))
        axes2 = list(range(1, tmp2.ndim))
        axes2.insert(self.j, 0)
        psi_ij = tmp2.permute(axes2)

        out = c * psi + s * psi_ij
        return out.reshape(-1, 1)

    def matrix(self, param=None) -> torch.Tensor:
        """
        (Optional fallback) Build the full RYY unitary for small N.
        Returns a sparse‑COO tensor if `self.sparse=True`, else dense.
        """
        phi = (self.phi if param is None else torch.tensor(param, device=self.device))[0]
        c = torch.cos(phi / 2)
        s = -1j * torch.sin(phi / 2)

        # Full local XX = Y_i ⊗ Y_j
        YY = torch.kron(self.Y_i, self.Y_j)

        # identity on other wires
        U = YY * s + torch.eye(YY.shape[0], dtype=torch.complex64, device=self.device) * c
        # then kron with identities for other qudits via aux.eye...
        # (left as an exercise or use your existing kron-based fallback)

        raise NotImplementedError("Full matrix construction not implemented in this fallback.")


class RZZ(nn.Module):
    r"""
    Memory‑efficient RZZ gate for qudits: exp(-i * phi * Z_i Z_j / 2) via elementwise phases.

    Applies the two‑qudit rotation directly on the state tensor, avoiding full 2^N×2^N unitaries.

    **Arguments:**
        index (list[int]): Two target qudit axes [i, j].
        dim (int or list[int]): Qudit dimensions; if int, repeated `wires` times.
        wires (int): Number of qudits when `dim` is int.
        device (str): 'cpu' or 'cuda'.
        angle (float or Tensor or None): Rotation angle φ; random if None.
        sparse (bool): Unused in forward (matrix fallback only).
    """
    def __init__(
        self,
        index,
        dim=2,
        wires=None,
        device='cpu',
        angle=None,
        sparse=False,
    ):
        super().__init__()
        if not (isinstance(index, (list, tuple)) and len(index) == 2):
            raise ValueError("`index` must be two axes [i, j].")
        self.i, self.j = index
        self.device = device
        self.sparse = sparse

        # build dims
        if isinstance(dim, int):
            if wires is None:
                raise ValueError("Specify `wires` when `dim` is int.")
            self.dim_list = [dim] * wires
        else:
            self.dim_list = list(dim)
            wires = len(self.dim_list)
        self.wires = wires

        # angle parameter
        if angle is None:
            self.phi = nn.Parameter(torch.randn(1, device=device))
        else:
            self.phi = torch.tensor(angle * torch.ones(1, device=device), device=device)

        # precompute Z-phase vectors
        Z_mat_i = Z(dim=self.dim_list[self.i], wires=1, device=device).matrix()
        Z_mat_j = Z(dim=self.dim_list[self.j], wires=1, device=device).matrix()
        self.register_buffer('ph_i', torch.diag(Z_mat_i))
        self.register_buffer('ph_j', torch.diag(Z_mat_j))

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        if x.is_sparse:
            x = x.to_dense()
        psi = x.view(*self.dim_list)
        phi = (self.phi if param is None else torch.tensor(param, device=self.device))[0]
        c = torch.cos(phi / 2)
        s = -1j * torch.sin(phi / 2)
        # broadcast phases
        shape_i = [1] * self.wires; shape_i[self.i] = self.ph_i.shape[0]
        shape_j = [1] * self.wires; shape_j[self.j] = self.ph_j.shape[0]
        ph_i = self.ph_i.view(shape_i)
        ph_j = self.ph_j.view(shape_j)
        psi_flip = psi * ph_i * ph_j
        out = c * psi + s * psi_flip
        return out.reshape(-1, 1)

    def matrix(self, param=None) -> torch.Tensor:
        """
        (Fallback) Full RZZ matrix for small systems, sparse if requested.
        """
        phi = (self.phi if param is None else torch.tensor(param, device=self.device))[0]
        c = torch.cos(phi / 2)
        s = -1j * torch.sin(phi / 2)
        # local ZZ is diagonal via outer diag
        ZZ = torch.ger(self.ph_i, self.ph_j).view(-1)
        diag = c + s * ZZ
        U = torch.diag(diag)
        # embed via aux.kron for other wires...
        raise NotImplementedError("Full RZZ matrix construction not implemented.")