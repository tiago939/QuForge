"""
Noise model registry for QuForge.

Maps gate types to quantum noise channels so the trajectory engine knows
which ``QuantumChannel`` to apply after each gate in a circuit.

Example:

    noise = NoiseModel()
    noise.add_all_qudit_error('RX',  depolarizing(0.01, d=3))
    noise.add_all_qudit_error('CNOT', depolarizing(0.02, d=3, num_qudits=2))

    # qudit-specific override (qudit 0 is noisier)
    noise.add_qudit_error('RX', depolarizing(0.05, d=3), qudits=[0])

Two resolution levels
---------------------
1. **all-qudit**  – applies whenever a gate of this type fires, regardless
   of which qudits it targets.  Registered with ``add_all_qudit_error``.
2. **qudit-specific** – applies only when the gate targets a specific set of
   qudits.  Registered with ``add_qudit_error``.  Takes priority over the
   all-qudit entry for matching targets.

Gate names are matched against the class name of QuForge gate modules
(``type(gate).__name__``), e.g. ``'RX'``, ``'CNOT'``, ``'H'``, ``'U'``.

Readout errors
--------------
``add_readout_error`` attaches a classical confusion matrix to one or all
qudits.  The trajectory engine applies it after measurement sampling.
"""

from __future__ import annotations

import torch
import copy
from typing import Dict, List, Optional, Tuple, Union

from quforge.channels import QuantumChannel


class NoiseModel:
    """Registry that maps gate names to noise channels.

    Parameters
    ----------
    dim : int
        Default local qudit dimension (used only for informational
        purposes / validation messages).

    Example
    -------
    >>> from channels import depolarizing, dephasing
    >>> noise = NoiseModel()
    >>> noise.add_all_qudit_error('RX', depolarizing(0.01, d=3))
    >>> noise.add_all_qudit_error('CNOT', depolarizing(0.02, d=3, num_qudits=2))
    >>> noise.add_qudit_error('RX', depolarizing(0.05, d=3), qudits=[0])
    >>> # look up errors for an RX gate on qudit 0
    >>> errors = noise.get_errors('RX', target_qudits=[0])
    """

    def __init__(self, dim: int = 2):
        self.dim = dim

        # gate_name -> QuantumChannel
        self._all_qudit_errors: Dict[str, List[QuantumChannel]] = {}

        # (gate_name, frozenset(qudits)) -> QuantumChannel
        self._qudit_errors: Dict[Tuple[str, frozenset], List[QuantumChannel]] = {}

        # qudit_index (or 'all') -> confusion matrix  (d x d row-stochastic)
        self._readout_errors: Dict[Union[int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add_all_qudit_error(self, gate_name: str, channel: QuantumChannel) -> None:
        """Attach a noise channel to every instance of *gate_name*.

        If the gate targets *n* qudits the channel must act on *n* qudits
        (``channel.num_qudits == n``).  This is the user's responsibility;
        mismatches are caught at simulation time.

        Multiple calls with the same *gate_name* accumulate — all
        registered channels are applied in order.

        Parameters
        ----------
        gate_name : str
            Gate class name, e.g. ``'RX'``, ``'CNOT'``.
        channel : QuantumChannel
            Noise channel to apply.
        """
        self._all_qudit_errors.setdefault(gate_name, []).append(channel)

    def add_qudit_error(
        self,
        gate_name: str,
        channel: QuantumChannel,
        qudits: List[int],
    ) -> None:
        """Attach a noise channel that fires only when *gate_name* targets
        exactly *qudits*.

        When a match exists, the qudit-specific entry is used instead of
        (not in addition to) the all-qudit entry for that gate name.

        Parameters
        ----------
        gate_name : str
            Gate class name.
        channel : QuantumChannel
            Noise channel to apply.
        qudits : list[int]
            The specific qudit indices this error applies to.
        """
        key = (gate_name, frozenset(qudits))
        self._qudit_errors.setdefault(key, []).append(channel)

    def add_readout_error(
        self,
        confusion_matrix: torch.Tensor,
        qudits: Optional[List[int]] = None,
    ) -> None:
        """Attach a classical readout (measurement) error.

        Parameters
        ----------
        confusion_matrix : Tensor
            A  ``(d, d)``  row-stochastic matrix where entry ``(i, j)`` is
            the probability of reporting outcome *j* when the true state is
            *i*.  Rows must sum to 1.
        qudits : list[int] or None
            Qudit indices this error applies to.  ``None`` means all qudits.
        """
        # validate row-stochastic
        row_sums = confusion_matrix.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise ValueError("Confusion matrix rows must sum to 1 (row-stochastic).")

        if qudits is None:
            self._readout_errors["all"] = confusion_matrix
        else:
            for q in qudits:
                self._readout_errors[q] = confusion_matrix

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get_errors(self, gate_name: str, target_qudits: List[int]) -> List[QuantumChannel]:
        """Look up noise channels for a gate application.

        Resolution order:

        1. Check for a qudit-specific entry matching *target_qudits*.
        2. Fall back to the all-qudit entry for *gate_name*.
        3. Return an empty list if nothing is registered.

        Parameters
        ----------
        gate_name : str
            Gate class name (e.g. ``type(gate).__name__``).
        target_qudits : list[int]
            Which qudits the gate is acting on in this application.

        Returns
        -------
        list[QuantumChannel]
        """
        # qudit-specific takes priority
        key = (gate_name, frozenset(target_qudits))
        if key in self._qudit_errors:
            return self._qudit_errors[key]

        # fall back to all-qudit
        if gate_name in self._all_qudit_errors:
            return self._all_qudit_errors[gate_name]

        return []

    def get_readout_error(self, qudit_index: int) -> Optional[torch.Tensor]:
        """Return the confusion matrix for a qudit, or ``None``.

        Checks for a qudit-specific entry first, then falls back to the
        ``'all'`` entry.
        """
        if qudit_index in self._readout_errors:
            return self._readout_errors[qudit_index]
        if "all" in self._readout_errors:
            return self._readout_errors["all"]
        return None

    def has_readout_error(self) -> bool:
        """Whether any readout errors are registered."""
        return len(self._readout_errors) > 0

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def gate_names(self) -> List[str]:
        """All gate names that have at least one error registered."""
        names = set(self._all_qudit_errors.keys())
        names |= {name for name, _ in self._qudit_errors.keys()}
        return sorted(names)

    @property
    def is_empty(self) -> bool:
        """True if no errors are registered at all."""
        return (
            not self._all_qudit_errors
            and not self._qudit_errors
            and not self._readout_errors
        )

    def summary(self) -> str:
        """Human-readable summary of the noise model."""
        lines = ["NoiseModel"]
        lines.append(f"  default dim: {self.dim}")

        if self._all_qudit_errors:
            lines.append("  All-qudit errors:")
            for name, channels in sorted(self._all_qudit_errors.items()):
                for ch in channels:
                    lines.append(f"    {name}: {ch}")

        if self._qudit_errors:
            lines.append("  Qudit-specific errors:")
            for (name, qset), channels in sorted(self._qudit_errors.items()):
                for ch in channels:
                    lines.append(f"    {name} on {sorted(qset)}: {ch}")

        if self._readout_errors:
            lines.append("  Readout errors:")
            for key, mat in sorted(
                self._readout_errors.items(), key=lambda x: str(x[0])
            ):
                label = f"qudit {key}" if isinstance(key, int) else "all qudits"
                lines.append(f"    {label}: {mat.shape[0]}x{mat.shape[1]} confusion matrix")

        if self.is_empty:
            lines.append("  (no errors registered)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_gate = sum(len(v) for v in self._all_qudit_errors.values())
        n_qudit = sum(len(v) for v in self._qudit_errors.values())
        n_read = len(self._readout_errors)
        return (
            f"NoiseModel(gate_errors={n_gate}, qudit_errors={n_qudit}, "
            f"readout_errors={n_read})"
        )

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------
    def copy(self) -> "NoiseModel":
        """Return a deep copy of this noise model."""
        return copy.deepcopy(self)
