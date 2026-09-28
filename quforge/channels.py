"""
Quantum noise channels for qudit systems.

Provides Kraus-operator representations of common noise channels generalized
to arbitrary qudit dimension d.  Each factory function returns a
``QuantumChannel`` object that holds the Kraus operators and, when the
channel is composed of (scaled) unitaries, precomputed state-independent
sampling probabilities for Monte-Carlo trajectory simulation.

Channels
--------
depolarizing   – d-dimensional depolarizing via Heisenberg–Weyl operators
dephasing      – pure dephasing (kills off-diagonal elements)
amplitude_damping – cascaded decay  |k⟩ → |k-1⟩
custom_channel – user-supplied list of Kraus matrices

Helpers
-------
heisenberg_weyl(a, b, d)  – single Heisenberg–Weyl operator  X^a Z^b
shift_matrix(d)           – generalised Pauli-X  (shift / clock)
clock_matrix(d)           – generalised Pauli-Z  (phase / clock)
"""

import torch
import numpy as np
import cmath
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# QuantumChannel container
# ---------------------------------------------------------------------------

class QuantumChannel:
    """Container for a quantum noise channel in Kraus form.

    Parameters
    ----------
    kraus_ops : list[Tensor]
        Kraus operators  K_i  of shape ``(d, d)`` (single-qudit) or
        ``(d^n, d^n)`` (multi-qudit) satisfying  Σ K_i† K_i = I.
    probabilities : Tensor or None
        If the channel is a probabilistic mixture of unitaries the
        probabilities ``p_i`` are state-independent and can be
        pre-computed.  Shape ``(len(kraus_ops),)``.  When ``None`` the
        trajectory engine must compute  ``p_i = ‖K_i|ψ⟩‖²``  per state.
    device : str
        ``'cpu'`` or ``'cuda'``.
    num_qudits : int
        How many qudits this channel acts on (1 for single-qudit gates,
        2 for two-qudit gates, etc.).
    dim : int
        Local qudit dimension *d*.
    """

    def __init__(
        self,
        kraus_ops: List[torch.Tensor],
        probabilities: Optional[torch.Tensor] = None,
        device: str = "cpu",
        num_qudits: int = 1,
        dim: int = 2,
    ):
        self.kraus_ops = [K.to(device) for K in kraus_ops]
        self.probabilities = probabilities.to(device) if probabilities is not None else None
        self.device = device
        self.num_qudits = num_qudits
        self.dim = dim
        self.num_ops = len(kraus_ops)
        self.size = kraus_ops[0].shape[0]  # d  or  d^n

    # -- validation ----------------------------------------------------------
    def validate(self, atol: float = 1e-5) -> bool:
        """Check the completeness relation  Σ K_i† K_i ≈ I."""
        acc = torch.zeros(
            (self.size, self.size), dtype=torch.complex64, device=self.device
        )
        for K in self.kraus_ops:
            acc += K.conj().T @ K
        identity = torch.eye(self.size, dtype=torch.complex64, device=self.device)
        return torch.allclose(acc, identity, atol=atol)

    # -- pretty-print --------------------------------------------------------
    def __repr__(self) -> str:
        tag = "state-indep" if self.probabilities is not None else "state-dep"
        return (
            f"QuantumChannel(num_ops={self.num_ops}, size={self.size}, "
            f"dim={self.dim}, num_qudits={self.num_qudits}, probs={tag})"
        )


# ---------------------------------------------------------------------------
# Primitive matrices
# ---------------------------------------------------------------------------
def shift_matrix(d: int, power: int = 1, device: str = "cpu") -> torch.Tensor:
    r"""Generalised Pauli-X (shift) operator  X^a.

    ``X |j⟩ = |j+1  mod d⟩``

    Parameters
    ----------
    d : int
        Qudit dimension.
    power : int
        Exponent  *a*  in  X^a.
    device : str
        Tensor device.

    Returns
    -------
    Tensor of shape ``(d, d)``, complex64.
    """
    X = torch.zeros((d, d), dtype=torch.complex64, device=device)
    for j in range(d):
        X[(j + power) % d, j] = 1.0
    return X


def clock_matrix(d: int, power: int = 1, device: str = "cpu") -> torch.Tensor:
    r"""Generalised Pauli-Z (clock) operator  Z^b.

    ``Z |j⟩ = ω^j |j⟩``   with  ``ω = exp(2πi/d)``

    Parameters
    ----------
    d : int
        Qudit dimension.
    power : int
        Exponent  *b*  in  Z^b.
    device : str
        Tensor device.

    Returns
    -------
    Tensor of shape ``(d, d)``, complex64.
    """
    omega = cmath.exp(2j * cmath.pi / d)
    phases = torch.tensor(
        [omega ** ((j * power) % d) for j in range(d)],
        dtype=torch.complex64,
        device=device,
    )
    return torch.diag(phases)


def heisenberg_weyl(a: int, b: int, d: int, device: str = "cpu") -> torch.Tensor:
    r"""Single Heisenberg–Weyl displacement operator  W_{a,b} = X^a Z^b.

    The set  {W_{a,b} : a,b ∈ {0,…,d-1}}  forms a unitary operator basis
    for the space of  d×d  matrices and satisfies

    .. math::
        \frac{1}{d}\sum_{a,b} W_{a,b}\,\rho\,W_{a,b}^\dagger = \frac{I}{d}

    Parameters
    ----------
    a, b : int
        Shift and clock exponents.
    d : int
        Qudit dimension.
    device : str
        Tensor device.

    Returns
    -------
    Tensor of shape ``(d, d)``, complex64.
    """
    return shift_matrix(d, power=a, device=device) @ clock_matrix(d, power=b, device=device)


# ---------------------------------------------------------------------------
# Channel factory functions
# ---------------------------------------------------------------------------
def depolarizing(
    p: float,
    d: int = 2,
    num_qudits: int = 1,
    device: str = "cpu",
) -> QuantumChannel:
    r"""*d*-dimensional depolarizing channel.

    .. math::
        \mathcal{E}(\rho) = (1-p)\rho + p\frac{I}{d^n}

    where  n = ``num_qudits``  and  D = d^n.

    Kraus decomposition using Heisenberg–Weyl operators:

    * ``K_0 = sqrt(1 - p + p/D²) · I``             (identity)
    * ``K_{a,b} = sqrt(p/D²) · W_{a,b}``            for  (a,b) ≠ (0,0)

    All Kraus operators are proportional to unitaries so the sampling
    probabilities are state-independent.

    Parameters
    ----------
    p : float
        Depolarizing probability  (0 ≤ p ≤ 1).
    d : int
        Local qudit dimension.
    num_qudits : int
        Number of qudits the channel acts on.
    device : str
        Tensor device.
    """
    D = d ** num_qudits          # total Hilbert-space dimension
    D2 = D * D                   # number of HW operators

    p_id = 1.0 - p + p / D2     # probability of identity
    p_hw = p / D2                # probability of each non-identity HW op

    kraus_ops = []
    probs = []

    # Build all D² Heisenberg–Weyl operators  W_{a1,b1} ⊗ W_{a2,b2} ⊗ …
    # For a single qudit this is just W_{a,b}; for n qudits we take the
    # tensor product of per-qudit HW operators.
    indices = _hw_index_grid(d, num_qudits)

    for idx, (ab_tuples) in enumerate(indices):
        # Construct the (possibly multi-qudit) HW operator
        W = _multi_qudit_hw(ab_tuples, d, device)

        is_identity = all(a == 0 and b == 0 for a, b in ab_tuples)
        coeff = p_id ** 0.5 if is_identity else p_hw ** 0.5
        prob = p_id if is_identity else p_hw

        kraus_ops.append(coeff * W)
        probs.append(prob)

    probs_t = torch.tensor(probs, dtype=torch.float64, device=device)

    return QuantumChannel(
        kraus_ops=kraus_ops,
        probabilities=probs_t,
        device=device,
        num_qudits=num_qudits,
        dim=d,
    )


def dephasing(
    p: float,
    d: int = 2,
    device: str = "cpu",
) -> QuantumChannel:
    r"""*d*-dimensional pure dephasing channel (single qudit).

    .. math::
        \mathcal{E}(\rho) = (1-p)\rho + p\sum_{k=0}^{d-1}|k\rangle\langle k|\rho\,|k\rangle\langle k|

    Off-diagonal elements are damped:  \rho_{ij} → (1-p) \rho_{ij}  for  i ≠ j.

    Kraus operators:

    * ``K_0 = sqrt(1-p) · I``
    * ``K_k = sqrt(p) · |k⟩⟨k|``   for k = 0, …, d-1

    Probabilities are state-dependent (projective Kraus operators).

    Parameters
    ----------
    p : float
        Dephasing probability  (0 ≤ p ≤ 1).
    d : int
        Qudit dimension.
    device : str
        Tensor device.
    """
    kraus_ops = []

    # K_0 = sqrt(1-p) I
    K0 = (1 - p) ** 0.5 * torch.eye(d, dtype=torch.complex64, device=device)
    kraus_ops.append(K0)

    # K_k = sqrt(p) |k><k|
    for k in range(d):
        Kk = torch.zeros((d, d), dtype=torch.complex64, device=device)
        Kk[k, k] = p ** 0.5
        kraus_ops.append(Kk)

    return QuantumChannel(
        kraus_ops=kraus_ops,
        probabilities=None,  # state-dependent
        device=device,
        num_qudits=1,
        dim=d,
    )


def amplitude_damping(
    gamma: float,
    d: int = 2,
    device: str = "cpu",
) -> QuantumChannel:
    r"""*d*-dimensional amplitude damping channel (cascaded decay, single qudit).

    Models energy relaxation where each excited level  |k⟩  decays to
    |k-1⟩  with probability  \gamma  (uniform rate across all transitions).

    Kraus operators:

    * ``K_0 = |0⟩⟨0| + Σ_{k=1}^{d-1} sqrt(1-\gamma) |k⟩⟨k|``
    * ``K_k = sqrt(\gamma) |k-1⟩⟨k|``   for  k = 1, …, d-1

    Probabilities are state-dependent.

    Parameters
    ----------
    gamma : float
        Decay probability per level  (0 ≤ γ ≤ 1).
    d : int
        Qudit dimension.
    device : str
        Tensor device.
    """
    kraus_ops = []

    # K_0: ground state unchanged, excited states survive with sqrt(1-γ)
    K0 = torch.zeros((d, d), dtype=torch.complex64, device=device)
    K0[0, 0] = 1.0
    for k in range(1, d):
        K0[k, k] = (1 - gamma) ** 0.5
    kraus_ops.append(K0)

    # K_k: decay  |k⟩ → |k-1⟩
    for k in range(1, d):
        Kk = torch.zeros((d, d), dtype=torch.complex64, device=device)
        Kk[k - 1, k] = gamma ** 0.5
        kraus_ops.append(Kk)

    return QuantumChannel(
        kraus_ops=kraus_ops,
        probabilities=None,  # state-dependent
        device=device,
        num_qudits=1,
        dim=d,
    )


def custom_channel(
    kraus_ops: List[torch.Tensor],
    dim: int = 2,
    num_qudits: int = 1,
    probabilities: Optional[torch.Tensor] = None,
    device: str = "cpu",
    validate: bool = True,
) -> QuantumChannel:
    """Create a channel from user-supplied Kraus operators.

    Parameters
    ----------
    kraus_ops : list[Tensor]
        List of  (D, D)  Kraus matrices where  D = dim^num_qudits.
    dim : int
        Local qudit dimension.
    num_qudits : int
        Number of qudits the channel acts on.
    probabilities : Tensor or None
        State-independent probabilities (if applicable).
    device : str
        Tensor device.
    validate : bool
        If ``True``, check the completeness relation on construction.
    """
    ch = QuantumChannel(
        kraus_ops=kraus_ops,
        probabilities=probabilities,
        device=device,
        num_qudits=num_qudits,
        dim=dim,
    )
    if validate:
        if not ch.validate():
            raise ValueError(
                "Kraus operators do not satisfy the completeness relation "
                "Σ K_i† K_i = I  (atol=1e-5)."
            )
    return ch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _hw_index_grid(d: int, n: int):
    """Generate all  d^(2n)  Heisenberg–Weyl index tuples for  n  qudits.

    Each element is a list of  n  tuples  [(a1,b1), (a2,b2), …]  with
    a_i, b_i ∈ {0, …, d-1}.

    Returns
    -------
    list[list[tuple[int,int]]]
    """
    import itertools
    single = list(itertools.product(range(d), range(d)))  # d² pairs
    return list(itertools.product(single, repeat=n))


def _multi_qudit_hw(
    ab_tuples: tuple,
    d: int,
    device: str,
) -> torch.Tensor:
    """Build the tensor-product HW operator  W_{a1,b1} ⊗ W_{a2,b2} ⊗ … .

    Parameters
    ----------
    ab_tuples : tuple of (a, b) pairs
        One pair per qudit.
    d : int
        Local qudit dimension.
    device : str
        Tensor device.

    Returns
    -------
    Tensor of shape  (d^n, d^n).
    """
    W = torch.tensor([[1.0]], dtype=torch.complex64, device=device)
    for a, b in ab_tuples:
        Wi = heisenberg_weyl(a, b, d, device=device)
        W = torch.kron(W, Wi)
    return W


# ---------------------------------------------------------------------------
# Convenience: thermal relaxation (T1/T2)
# ---------------------------------------------------------------------------
def thermal_relaxation(
    t1: float,
    t2: float,
    time: float,
    d: int = 2,
    device: str = "cpu",
) -> QuantumChannel:
    r"""Thermal relaxation channel combining T1 (energy decay) and T2 (dephasing).

    Composes amplitude damping (T1 process) with pure dephasing

    * Amplitude damping parameter:  \gamma = 1 - exp(-t / T1)
    * Pure dephasing parameter:     p = 1 - exp(-t / T_\phi)
      where  1/T_\phi = 1/T2 - 1/(2 T1).

    Generalised to arbitrary qudit dimension *d* via the *d*-level
    ``amplitude_damping`` and ``dephasing`` channels.

    Parameters
    ----------
    t1 : float
        T1 relaxation time (energy decay).
    t2 : float
        T2 dephasing time.  Must satisfy  T2 ≤ 2·T1.
    time : float
        Gate duration (same units as *t1* and *t2*).
    d : int
        Qudit dimension.
    device : str
        Tensor device.
    """
    import math

    if t2 > 2 * t1 + 1e-10:
        raise ValueError(f"T2 ({t2}) must be <= 2*T1 ({2*t1}).")

    # amplitude damping parameter
    gamma = 1.0 - math.exp(-time / t1)

    # pure dephasing parameter
    if t2 < 2 * t1 and t2 > 0:
        rate_phi = 1.0 / t2 - 1.0 / (2.0 * t1)
        p_deph = 1.0 - math.exp(-time * rate_phi)
    else:
        p_deph = 0.0

    ch_ad = amplitude_damping(gamma, d=d, device=device)

    if p_deph > 1e-15:
        ch_deph = dephasing(p_deph, d=d, device=device)
        return compose(ch_ad, ch_deph)
    else:
        return ch_ad


# ---------------------------------------------------------------------------
# Convenience: compose channels
# ---------------------------------------------------------------------------
def compose(ch1: QuantumChannel, ch2: QuantumChannel) -> QuantumChannel:
    """Compose two channels  ch2 ∘ ch1  (ch1 applied first).

    The resulting Kraus set is  { K2_j  K1_i }  for all pairs  (i, j).
    State-independent probabilities are preserved only if both inputs
    have them.

    Parameters
    ----------
    ch1, ch2 : QuantumChannel
        Channels acting on the same Hilbert space.
    """
    if ch1.size != ch2.size:
        raise ValueError("Cannot compose channels of different sizes.")

    new_ops = []
    new_probs = []
    has_probs = ch1.probabilities is not None and ch2.probabilities is not None

    for i, K1 in enumerate(ch1.kraus_ops):
        for j, K2 in enumerate(ch2.kraus_ops):
            new_ops.append(K2 @ K1)
            if has_probs:
                new_probs.append(
                    ch1.probabilities[i].item() * ch2.probabilities[j].item()
                )

    probs_t = torch.tensor(new_probs, dtype=torch.float64, device=ch1.device) if has_probs else None

    return QuantumChannel(
        kraus_ops=new_ops,
        probabilities=probs_t,
        device=ch1.device,
        num_qudits=ch1.num_qudits,
        dim=ch1.dim,
    )