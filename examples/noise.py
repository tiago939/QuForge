"""
Demonstrates depolarizing and thermal-relaxation noise using pedagogical
parameters adapted from Qiskit Aer documentation. QuForge attaches noise
channels directly to gate types, so no ``id``/``delay`` gates are needed.

Example noise parameters adapted from Qiskit Aer documentation:
    Single-qudit gate infidelity  p_sq = 0.001  (0.1%)
    Two-qudit gate infidelity     p_tq = 0.01   (1%)
    T1 = 50 µs,   T2 = 70 µs
    Single-qudit gate time  t_sq =  50 ns
    Two-qudit gate time     t_tq = 300 ns
"""

import torch
import numpy as np
import random
import argparse

import quforge.quforge as qf


# ======================================================================
# Random seed
# ======================================================================

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ======================================================================
# Noise model
# ======================================================================

def qiskit_example_noise_model(d: int = 2, device: str = "cpu") -> qf.NoiseModel():
    """Build an example noise model for qudit dimension d.

    Two noise sources per gate:
      1. Depolarising error  (coherent gate infidelity)
      2. Thermal relaxation  (T1/T2 decoherence during gate execution)

    In Qiskit these are attached to separate gates (real gate → depolarising,
    dummy id/delay → decoherence).  In QuForge both are attached to the same
    gate and applied in sequence.
    """
    # -- Gate infidelities --
    p_sq = 0.001          # single-qudit average gate infidelity
    p_tq = 0.01           # two-qudit average gate infidelity

    # Convert infidelity → depolarising parameter:  p_depol = r · d² / (d² − 1)
    d_sq = d              # Hilbert-space dim for single-qudit gate
    d_tq = d ** 2         # Hilbert-space dim for two-qudit gate

    p_depol_sq = p_sq * d_sq ** 2 / (d_sq ** 2 - 1)
    p_depol_tq = p_tq * d_tq ** 2 / (d_tq ** 2 - 1)

    depol_sq = qf.depolarizing(p_depol_sq, d=d, num_qudits=1, device=device)
    depol_tq = qf.depolarizing(p_depol_tq, d=d, num_qudits=2, device=device)

    # -- Decoherence times and gate durations (all in ns) --
    # Pedagogical values adapted from Qiskit Aer documentation.
    t1 = 50_000.0          # T1 = 50 µs
    t2 = 70_000.0          # T2 = 70 µs
    t_sq = 50.0            # representative single-qudit gate time = 50 ns
    t_tq = 300.0           # representative two-qudit gate time = 300 ns

    decoherence_sq = qf.thermal_relaxation(t1, t2, t_sq, d=d, device=device)
    decoherence_tq = qf.thermal_relaxation(t1, t2, t_tq, d=d, device=device)

    # -- Assemble --
    noise = qf.NoiseModel(dim=d)

    # Single-qudit gates: depolarising + decoherence
    for gate_name in ["RX", "RY", "RZ"]:
        noise.add_all_qudit_error(gate_name, depol_sq)
        noise.add_all_qudit_error(gate_name, decoherence_sq)

    # Two-qudit gate: depolarising + decoherence
    noise.add_all_qudit_error("RXX", depol_tq)
    noise.add_all_qudit_error("RXX", decoherence_tq)

    return noise


# ======================================================================
# Random circuit
# ======================================================================

def build_random_circuit(n_qudits: int = 2, n_layers: int = 4, d: int = 2,
                         noise_model=None, n_trajectories: int = 1,
                         device: str = "cpu"):
    """Build a random circuit with single-qudit rotations and RXX entanglers.

    Each layer randomly chooses between:
      - independent RX + RY rotations on each qudit  (prob 0.5)
      - an RXX(π/2) entangling gate on qudits 0,1     (prob 0.5)
    """
    circ = qf.Circuit(
        dim=d, wires=n_qudits, device=device,
        noise_model=noise_model,
        n_trajectories=n_trajectories,
    )

    for layer in range(n_layers):
        if random.random() < 0.5:
            # Single-qudit rotations on every qudit
            for q in range(n_qudits):
                circ.RX(index=[q])
                circ.RY(index=[q])
        else:
            # Entangling gate
            circ.RXX(index=[0, 1])

    return circ


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Qiskit-style noise model — QuForge")
    parser.add_argument("--dim", type=int, default=2,
                        help="Qudit dimension (2=qubit, 3=qutrit, …)")
    parser.add_argument("--wires", type=int, default=2,
                        help="Number of qudits")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of random circuit layers")
    parser.add_argument("--trajectories", type=int, default=2048,
                        help="Monte-Carlo trajectories")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    d = args.dim
    n_qudits = args.wires
    D = d ** n_qudits

    print(f"Qiskit-style noise model — QuForge")
    print(f"  d={d},  wires={n_qudits},  D={D},  trajectories={args.trajectories}")
    print()

    # -- Build noise model --
    noise = qiskit_example_noise_model(d=d, device=args.device)
    print(noise.summary())
    print()

    # -- Build circuit --
    circ = build_random_circuit(
        n_qudits=n_qudits, n_layers=args.layers, d=d,
        noise_model=noise, n_trajectories=args.trajectories,
        device=args.device,
    )

    # -- Initial state: |00…0⟩ --
    label = "-".join(["0"] * n_qudits)
    state = qf.State(label, dim=d)
    print(f"Initial state: |{label}⟩")

    # -- Noisy run --
    print("Running noisy simulation…")
    psi_batch = circ(state)  # (n_trajectories, D, 1)
    print(f"  output shape: {psi_batch.shape}")

    # -- Expectation value of Z on qudit 0 --
    # Generalised Z = diag(1, ω, ω², …, ω^{d-1})
    omega = torch.exp(torch.tensor(2j * np.pi / d))
    Z_local = torch.diag(
        torch.tensor([omega ** k for k in range(d)], dtype=torch.complex64,
                      device=args.device)
    )
    Z_full = Z_local
    for _ in range(n_qudits - 1):
        Z_full = torch.kron(Z_full, torch.eye(d, dtype=torch.complex64,
                                               device=args.device))

    exp_noisy = qf.trajectory_expectation(psi_batch, Z_full)
    print(f"  <Z_0> (noisy):    {exp_noisy.item():.6f}")

    # -- Noiseless reference --
    circ.noise_model = None
    psi_clean = circ(state)  # (D, 1)
    exp_clean = (psi_clean.conj().T @ Z_full @ psi_clean).squeeze().real.item()
    print(f"  <Z_0> (noiseless): {exp_clean:.6f}")
    print(f"  difference:        {abs(exp_noisy.item() - exp_clean):.6f}")

    # -- Measurement histogram --
    circ.noise_model = noise  # toggle back on
    psi_batch = circ(state)
    hist = qf.noisy_measure(
        psi_batch,
        dim_list=[d] * n_qudits,
        index=list(range(n_qudits)),
    )
    print(f"\nMeasurement histogram ({args.trajectories} shots):")
    for outcome, count in sorted(hist.items(), key=lambda x: -x[1]):
        bar = "█" * int(40 * count / args.trajectories)
        print(f"  |{outcome}⟩  {count:5d}  ({count/args.trajectories:.3f})  {bar}")


if __name__ == "__main__":
    main()