"""
Monte-Carlo trajectory engine for noisy qudit simulation.

Provides the helper functions that the ``Circuit`` class calls when a
``NoiseModel`` is attached:

* applying a gate across a batch of trajectories  (via ``torch.vmap``
  or a plain loop fallback);
* stochastically applying Kraus operators from a ``QuantumChannel``,
  with a fast path for state-independent channels (depolarizing) and a
  general path for state-dependent channels (amplitude damping);
* extracting target-qudit indices from gate modules;
* noisy measurement with optional readout-error confusion matrices.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Optional, Tuple, Callable
from functools import partial
from math import prod

from quforge.channels import QuantumChannel
from quforge.noise import NoiseModel


# ======================================================================
# Gate-target extraction
# ======================================================================
def get_gate_targets(gate) -> List[int]:
    """Extract the qudit indices a gate acts on.

    Uses duck-typing to cover all QuForge gate classes:

    ============== =========================
    Gate style     Attribute(s) inspected
    ============== =========================
    RX, RY, H, …  ``gate.index``  (list)
    CRX, CRY, CZ  ``gate.ctrl``, ``gate.tgt``
    SWAP           ``gate.a1``,   ``gate.a2``
    RXX, RYY, RZZ ``gate.i``,    ``gate.j``
    ============== =========================

    Falls back to ``range(gate.wires)`` if nothing else matches.
    """
    # single/multi-qudit gates with an explicit index list
    if hasattr(gate, "index") and isinstance(getattr(gate, "index"), (list, tuple)):
        return list(gate.index)

    # controlled gates
    if hasattr(gate, "ctrl") and hasattr(gate, "tgt"):
        return [gate.ctrl, gate.tgt]

    # SWAP
    if hasattr(gate, "a1") and hasattr(gate, "a2"):
        return [gate.a1, gate.a2]

    # RXX / RYY / RZZ
    if hasattr(gate, "i") and hasattr(gate, "j"):
        ii = gate.i
        jj = gate.j
        if isinstance(ii, int) and isinstance(jj, int):
            return [ii, jj]

    # last resort: assume all wires
    if hasattr(gate, "wires"):
        return list(range(gate.wires))

    return []


# ======================================================================
# Local operator application
# ======================================================================

def apply_local_op(
    psi: torch.Tensor,
    op: torch.Tensor,
    targets: List[int],
    dim_list: List[int],
) -> torch.Tensor:
    """Apply a local operator to specific qudits of a state vector.

    Uses the permute → reshape → matmul → unpermute pattern.

    Parameters
    ----------
    psi : Tensor
        State vector of shape ``(D, 1)`` or ``(D,)``.
    op : Tensor
        Local operator of shape ``(d_sub, d_sub)`` where
        ``d_sub = prod(dim_list[t] for t in targets)``.
    targets : list[int]
        Qudit indices the operator acts on.
    dim_list : list[int]
        Dimensions of all qudits.

    Returns
    -------
    Tensor of shape ``(D, 1)``.
    """
    D = prod(dim_list)
    n = len(dim_list)
    squeeze = psi.dim() == 1
    psi = psi.view(*dim_list)

    targets_sorted = sorted(targets)
    rest = [i for i in range(n) if i not in targets_sorted]
    order = targets_sorted + rest
    inverse = [0] * n
    for i, ax in enumerate(order):
        inverse[ax] = i

    psi = psi.permute(order).contiguous()
    d_sub = prod(dim_list[t] for t in targets_sorted)
    d_rest = D // d_sub
    psi = psi.reshape(d_sub, d_rest)

    psi = op @ psi

    out_shape = [dim_list[i] for i in order]
    psi = psi.reshape(*out_shape)
    psi = psi.permute(inverse).contiguous()
    return psi.reshape(D) if squeeze else psi.reshape(D, 1)


# ======================================================================
# Gate application across trajectories
# ======================================================================

def apply_gate_batched(
    gate,
    psi_batch: torch.Tensor,
    use_vmap: bool = True,
) -> torch.Tensor:
    """Apply a gate to every trajectory in the batch.

    Parameters
    ----------
    gate : nn.Module
        A QuForge gate with ``forward(x) -> (D, 1)`` signature.
    psi_batch : Tensor
        Shape ``(n_traj, D, 1)``.
    use_vmap : bool
        If ``True``, uses ``torch.vmap`` for parallelism.
        If ``False``, falls back to a sequential loop (safer for gates
        that may not be vmap-compatible, e.g. sparse-matrix gates).

    Returns
    -------
    Tensor of shape ``(n_traj, D, 1)``.
    """
    if use_vmap:
        try:
            return torch.vmap(gate)(psi_batch)
        except Exception:
            # fall back silently if vmap chokes on this gate
            pass
    # sequential fallback
    return torch.stack([gate(psi_batch[i]) for i in range(psi_batch.shape[0])])


# ======================================================================
# Stochastic Kraus-operator application
# ======================================================================

def apply_channel_batched(
    psi_batch: torch.Tensor,
    channel: QuantumChannel,
    target_qudits: List[int],
    dim_list: List[int],
) -> torch.Tensor:
    """Stochastically apply a quantum channel to a batch of trajectories.

    If the channel is single-qudit (``channel.num_qudits == 1``) but the
    gate targeted multiple qudits, the channel is applied independently
    to each target qudit.

    Parameters
    ----------
    psi_batch : Tensor   ``(n_traj, D, 1)``
    channel : QuantumChannel
    target_qudits : list[int]
    dim_list : list[int]

    Returns
    -------
    Tensor ``(n_traj, D, 1)``
    """
    if channel.num_qudits == 1 and len(target_qudits) > 1:
        # independent single-qudit noise on each target
        for t in target_qudits:
            psi_batch = _apply_channel_core(psi_batch, channel, [t], dim_list)
        return psi_batch

    if channel.num_qudits != 1 and channel.num_qudits != len(target_qudits):
        raise ValueError(
            f"Channel acts on {channel.num_qudits} qudits but gate targets "
            f"{len(target_qudits)} qudits — mismatch."
        )

    return _apply_channel_core(psi_batch, channel, target_qudits, dim_list)


# ------------------------------------------------------------------
# Core dispatcher
# ------------------------------------------------------------------

def _apply_channel_core(
    psi_batch: torch.Tensor,
    channel: QuantumChannel,
    targets: List[int],
    dim_list: List[int],
) -> torch.Tensor:
    """Route to the state-independent or state-dependent path."""
    if channel.probabilities is not None:
        return _apply_state_indep(psi_batch, channel, targets, dim_list)
    else:
        return _apply_state_dep(psi_batch, channel, targets, dim_list)


# ------------------------------------------------------------------
# Fast path: state-independent probabilities
# ------------------------------------------------------------------

def _apply_state_indep(
    psi_batch: torch.Tensor,
    channel: QuantumChannel,
    targets: List[int],
    dim_list: List[int],
) -> torch.Tensor:
    """Apply a channel whose Kraus ops are proportional to unitaries.

    Sampling is a single ``torch.multinomial`` call.  The selected
    unitary (K_i / \sqrt{p_i}) is applied via ``apply_local_op`` vmapped
    over the trajectory batch.
    """
    n_traj = psi_batch.shape[0]
    device = psi_batch.device

    # precompute unitaries  U_i = K_i / sqrt(p_i)
    unitaries = []
    for K, p in zip(channel.kraus_ops, channel.probabilities):
        if p > 1e-15:
            unitaries.append(K / p.sqrt().to(K.dtype))
        else:
            unitaries.append(torch.zeros_like(K))
    U_stack = torch.stack(unitaries)  # (num_ops, d_sub, d_sub)

    # sample one op index per trajectory
    probs = channel.probabilities.float()
    indices = torch.multinomial(
        probs.unsqueeze(0).expand(n_traj, -1), 1, replacement=True
    ).squeeze(1)  # (n_traj,)

    # gather selected operators  →  (n_traj, d_sub, d_sub)
    U_selected = U_stack[indices]

    # apply in parallel
    def _apply_one(psi, op):
        return apply_local_op(psi, op, targets, dim_list)

    try:
        return torch.vmap(_apply_one)(psi_batch, U_selected)
    except Exception:
        return torch.stack(
            [_apply_one(psi_batch[i], U_selected[i]) for i in range(n_traj)]
        )


# ------------------------------------------------------------------
# General path: state-dependent probabilities
# ------------------------------------------------------------------

def _apply_state_dep(
    psi_batch: torch.Tensor,
    channel: QuantumChannel,
    targets: List[int],
    dim_list: List[int],
) -> torch.Tensor:
    """Apply a channel with state-dependent sampling probabilities.

    For each trajectory *and* each Kraus operator the result
    ``K_i|ψ⟩`` is computed; probabilities ``p_i = ‖K_i|ψ⟩‖²`` are
    evaluated, an index is sampled, and the corresponding result is
    normalised.
    """
    n_traj = psi_batch.shape[0]
    device = psi_batch.device
    num_ops = channel.num_ops

    # compute  K_i|ψ⟩  for every (trajectory, Kraus op)
    # result shape:  (n_traj, num_ops, D, 1)
    K_psi_all = []
    for K in channel.kraus_ops:
        def _apply_K(psi, _K=K):
            return apply_local_op(psi, _K, targets, dim_list)
        try:
            K_psi = torch.vmap(_apply_K)(psi_batch)  # (n_traj, D, 1)
        except Exception:
            K_psi = torch.stack([_apply_K(psi_batch[i]) for i in range(n_traj)])
        K_psi_all.append(K_psi)
    K_psi_all = torch.stack(K_psi_all, dim=1)  # (n_traj, num_ops, D, 1)

    # probabilities  p_i = ‖K_i|ψ⟩‖²
    probs = (K_psi_all.abs() ** 2).sum(dim=(-2, -1))  # (n_traj, num_ops)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-15)

    # sample
    indices = torch.multinomial(probs.float(), 1, replacement=True).squeeze(1)

    # select  K_i|ψ⟩  for the sampled index
    batch_idx = torch.arange(n_traj, device=device)
    selected = K_psi_all[batch_idx, indices]  # (n_traj, D, 1)

    # normalise
    norms = selected.reshape(n_traj, -1).norm(dim=1).reshape(n_traj, 1, 1).clamp(min=1e-15)
    selected = selected / norms

    return selected


# ======================================================================
# Noisy measurement
# ======================================================================

def noisy_measure(
    psi_batch: torch.Tensor,
    dim_list: List[int],
    noise_model: Optional[NoiseModel] = None,
    index: Optional[List[int]] = None,
    shots_per_trajectory: int = 1,
) -> dict:
    """Measure a batch of trajectory states and aggregate into a histogram.

    Each trajectory is an independent noise realisation.  For each
    trajectory the Born-rule probability is used to sample an outcome,
    optionally corrupted by the readout-error confusion matrix from
    ``noise_model``.

    Parameters
    ----------
    psi_batch : Tensor
        Shape ``(n_traj, D, 1)`` — the batch of trajectory states.
    dim_list : list[int]
        Per-qudit dimensions.
    noise_model : NoiseModel or None
        If provided and it contains readout errors, they are applied.
    index : list[int] or None
        Qudit indices to measure.  ``None`` measures all.
    shots_per_trajectory : int
        Samples per trajectory (usually 1).

    Returns
    -------
    dict
        ``{ outcome_string: count }`` histogram aggregated over all
        trajectories × shots.
    """
    # handle bare (D, 1) state — treat as a single trajectory
    if psi_batch.dim() == 2:
        psi_batch = psi_batch.unsqueeze(0)

    n_traj = psi_batch.shape[0]
    D = psi_batch.shape[1]
    n_qudits = len(dim_list)
    if index is None:
        index = list(range(n_qudits))

    histogram: dict = {}

    for t in range(n_traj):
        psi = psi_batch[t].squeeze(-1)  # (D,)

        # Born-rule probabilities over full computational basis
        probs_full = (psi.abs() ** 2).float()
        probs_full = probs_full / probs_full.sum().clamp(min=1e-15)

        # sample full-basis outcomes
        samples = torch.multinomial(probs_full, shots_per_trajectory, replacement=True)

        for s in samples:
            outcome_idx = s.item()

            # decode global index → per-qudit digits
            digits = []
            rem = outcome_idx
            for q in range(n_qudits):
                stride = prod(dim_list[q + 1:]) if q < n_qudits - 1 else 1
                digits.append(rem // stride)
                rem = rem % stride

            # keep only measured qudits
            measured = [digits[q] for q in index]

            # apply readout error (classical bit-flip)
            if noise_model is not None:
                for i, q in enumerate(index):
                    cm = noise_model.get_readout_error(q)
                    if cm is not None:
                        true_val = measured[i]
                        row = cm[true_val].float()
                        row = row / row.sum().clamp(min=1e-15)
                        measured[i] = torch.multinomial(row, 1).item()

            key = "-".join(str(d) for d in measured)
            histogram[key] = histogram.get(key, 0) + 1

    return histogram


# ======================================================================
# Expectation value over trajectories
# ======================================================================

def trajectory_expectation(
    psi_batch: torch.Tensor,
    observable: torch.Tensor,
) -> torch.Tensor:
    """Estimate ⟨O⟩ by averaging over trajectories.

    Parameters
    ----------
    psi_batch : Tensor  ``(n_traj, D, 1)``
    observable : Tensor ``(D, D)``

    Returns
    -------
    Scalar tensor — the trajectory-averaged expectation value.
    """
    if not isinstance(observable, torch.Tensor):
        raise TypeError(
            f"observable must be a (D, D) torch.Tensor, got {type(observable).__name__}."
        )
    if not isinstance(psi_batch, torch.Tensor):
        raise TypeError(
            f"psi_batch must be a (n_traj, D, 1) torch.Tensor, got {type(psi_batch).__name__}."
        )
    # handle bare (D, 1) state — treat as a single trajectory
    if psi_batch.dim() == 2:
        psi_batch = psi_batch.unsqueeze(0)

    # ⟨ψ_i | O | ψ_i⟩ for each trajectory
    # O @ psi:  (D, D) @ (n_traj, D, 1)  →  use batched matmul
    O_psi = torch.matmul(observable, psi_batch)  # (n_traj, D, 1)
    exp_vals = torch.matmul(
        psi_batch.conj().transpose(-2, -1), O_psi
    ).squeeze(-1).squeeze(-1)  # (n_traj,)

    return exp_vals.real.mean()