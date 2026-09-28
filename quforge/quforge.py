import torch
import torch.nn as nn
import quforge.aux as aux
import quforge.gates as gates
import quforge.statevector as sv
import quforge.circuit as circuit
import quforge.optimizer as optimizer
import quforge.channels as channels
import quforge.noise as noise_module
import quforge.trajectory as trajectory


def State(*args, **kwargs):
    return sv.State(*args, **kwargs)


def H(*args, **kwargs):
    return gates.H(*args, **kwargs)


def X(*args, **kwargs):
    return gates.X(*args, **kwargs)


def Y(*args, **kwargs):
    return gates.Y(*args, **kwargs)


def Z(*args, **kwargs):
    return gates.Z(*args, **kwargs)


def RX(*args, **kwargs):
    return gates.RX(*args, **kwargs)


def RY(*args, **kwargs):
    return gates.RY(*args, **kwargs)


def RZ(*args, **kwargs):
    return gates.RZ(*args, **kwargs)


def CNOT(*args, **kwargs):
    return gates.CNOT(*args, **kwargs)


def SWAP(*args, **kwargs):
    return gates.SWAP(*args, **kwargs)


def CZ(*args, **kwargs):
    return gates.CZ(*args, **kwargs)


def CRX(*args, **kwargs):
    return gates.CRX(*args, **kwargs)


def CRY(*args, **kwargs):
    return gates.CRX(*args, **kwargs)


def CRZ(*args, **kwargs):
    return gates.CRZ(*args, **kwargs)


def CCNOT(*args, **kwargs):
    return gates.CCNOT(*args, **kwargs)


def MCX(*args, **kwargs):
    return gates.MCX(*args, **kwargs)


def U(*args, **kwargs):
    return gates.U(*args, **kwargs)


def CU(*args, **kwargs):
    return gates.CU(*args, **kwargs)


def Circuit(*args, **kwargs):
    return circuit.Circuit(*args, **kwargs)


def measure(*args, **kwargs):
    return sv.measure(*args, **kwargs)


def exp_value(*args, **kwargs):
    return sv.exp_value(*args, **kwargs)


# ======================================================================
# Noise simulation
# ======================================================================

# -- Noise model --

def NoiseModel(*args, **kwargs):
    """Create a noise model registry.

    Example::

        noise = qf.NoiseModel()
        noise.add_all_qudit_error('RX', qf.depolarizing(0.01, d=3))
        circ = qf.Circuit(dim=3, wires=4, noise_model=noise, n_trajectories=512)
    """
    return noise_module.NoiseModel(*args, **kwargs)


# -- Channel constructors --

def depolarizing(*args, **kwargs):
    r"""*d*-dimensional depolarizing channel.

    ``E(rho) = (1-p) rho + p I/d^n``

    Uses Heisenberg-Weyl operators; probabilities are state-independent.

    Args:
        p (float): depolarizing probability.
        d (int): local qudit dimension.
        num_qudits (int): qudits the channel acts on (default 1).
        device (str): 'cpu' or 'cuda'.
    """
    return channels.depolarizing(*args, **kwargs)


def dephasing(*args, **kwargs):
    r"""*d*-dimensional pure dephasing channel.

    Off-diagonal elements damped: ``rho_ij -> (1-p) rho_ij`` for ``i != j``.

    Args:
        p (float): dephasing probability.
        d (int): qudit dimension.
        device (str): 'cpu' or 'cuda'.
    """
    return channels.dephasing(*args, **kwargs)


def amplitude_damping(*args, **kwargs):
    r"""*d*-dimensional amplitude damping (cascaded decay ``|k> -> |k-1>``).

    Args:
        gamma (float): decay probability per level.
        d (int): qudit dimension.
        device (str): 'cpu' or 'cuda'.
    """
    return channels.amplitude_damping(*args, **kwargs)


def custom_channel(*args, **kwargs):
    """Create a channel from user-supplied Kraus operators.

    Args:
        kraus_ops (list[Tensor]): Kraus matrices.
        dim (int): local qudit dimension.
        num_qudits (int): qudits the channel acts on.
        probabilities (Tensor or None): state-independent probs (if applicable).
        device (str): 'cpu' or 'cuda'.
        validate (bool): check completeness relation.
    """
    return channels.custom_channel(*args, **kwargs)


def compose_channels(*args, **kwargs):
    """Compose two channels: ``ch2 . ch1`` (ch1 applied first)."""
    return channels.compose(*args, **kwargs)


def thermal_relaxation(*args, **kwargs):
    r"""Thermal relaxation channel combining T1 and T2.

    Composes amplitude damping (T1) with pure dephasing (T2).

    Args:
        t1 (float): T1 relaxation time.
        t2 (float): T2 dephasing time (must be <= 2*T1).
        time (float): gate duration (same units as t1, t2).
        d (int): qudit dimension.
        device (str): 'cpu' or 'cuda'.
    """
    return channels.thermal_relaxation(*args, **kwargs)


# -- Trajectory utilities --

def noisy_measure(*args, **kwargs):
    """Measure a batch of trajectory states and aggregate into a histogram.

    Args:
        psi_batch (Tensor): shape ``(n_trajectories, D, 1)``.
        dim_list (list[int]): per-qudit dimensions.
        noise_model (NoiseModel or None): for readout errors.
        index (list[int] or None): qudits to measure (None = all).
        shots_per_trajectory (int): samples per trajectory.

    Returns:
        dict: ``{ outcome_string: count }``
    """
    return trajectory.noisy_measure(*args, **kwargs)


def trajectory_expectation(*args, **kwargs):
    r"""Estimate ``<O>`` by averaging over trajectories.

    Args:
        psi_batch (Tensor): shape ``(n_trajectories, D, 1)``.
        observable (Tensor): shape ``(D, D)``.

    Returns:
        Scalar tensor.
    """
    return trajectory.trajectory_expectation(*args, **kwargs)


# -- Primitive matrices --

def heisenberg_weyl(*args, **kwargs):
    """Heisenberg-Weyl displacement operator ``W_{a,b} = X^a Z^b``."""
    return channels.heisenberg_weyl(*args, **kwargs)


def shift_matrix(*args, **kwargs):
    """Generalised Pauli-X (shift) operator."""
    return channels.shift_matrix(*args, **kwargs)


def clock_matrix(*args, **kwargs):
    """Generalised Pauli-Z (clock) operator."""
    return channels.clock_matrix(*args, **kwargs)


def optim(*args, **kwargs):
    return optimizer.optim(*args, **kwargs)


optim.Adam = optimizer.optim.Adam
optim.SGD = optimizer.optim.SGD


def sum(*args, **kwargs):
    return torch.sum(*args, **kwargs)


def mean(*args, **kwargs):
    return torch.mean(*args, **kwargs)


def kron(*args, **kwargs):
    return aux.kron(*args, **kwargs)


def eye(*args, **kwargs):
    return aux.eye(*args, **kwargs)


def zeros(shape, device='cpu', dtype=torch.float32):
    return torch.zeros(shape, device=device, dtype=dtype)


def ones(shape, device='cpu', dtype=torch.float32):
    return torch.ones(shape, device=device, dtype=dtype)


def Tensor(x, device='cpu', dtype=torch.float32):
    return torch.Tensor(x, device=device, dtype=dtype)


def argmax(x):
    return torch.argmax(x)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()


class ModuleList(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(ModuleList, self).__init__(*args, **kwargs)


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)