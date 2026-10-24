"""
Challenge 05: Quantum Teleporter
Qiskit Fall Fest Nairobi 2026

Concepts:
- Quantum Teleportation Protocol using 3 qubits
- Entanglement as a quantum channel
- Bell basis measurement and classical feedforward
- State reconstruction at Bob's station
- Stretch: Teleporting arbitrary quantum states (|0>, |1>, |+>, |->, |i>, |psi(theta, phi)>),
  quantum state fidelity verification, and noise resilience analysis
"""
from __future__ import annotations

import math
from typing import Dict, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace


def build_teleportation_circuit(
    theta: float = 0.0,
    phi: float = 0.0,
    use_deferred_measurement: bool = True
) -> QuantumCircuit:
    """
    Build the full 3-qubit teleportation circuit.
    Q0: State to teleport |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
    Q1: Alice's entangled qubit
    Q2: Bob's target qubit
    """
    qc = QuantumCircuit(3)

    # 1. Prepare arbitrary input state on Qubit 0
    if theta != 0.0 or phi != 0.0:
        qc.ry(theta, 0)
        if phi != 0.0:
            qc.rz(phi, 0)
        qc.barrier(label="prepared")

    # 2. Create entangled Bell pair between Q1 (Alice) and Q2 (Bob)
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="EPR_pair")

    # 3. Alice's Bell-basis measurement operations on Q0 and Q1
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Alice_ops")

    # 4. Bob's conditional corrections
    # By the Principle of Deferred Measurement:
    # If Q1 is 1 -> Bob applies X: CX(1, 2)
    # If Q0 is 1 -> Bob applies Z: CZ(0, 2)
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.barrier(label="Bob_reconstructed")

    return qc


def verify_teleportation(theta: float, phi: float) -> Tuple[float, Statevector, Statevector]:
    """
    Simulate teleportation and compute the quantum state fidelity F
    between the original single-qubit state and Bob's received state on Q2.
    """
    qc_target = QuantumCircuit(1)
    if theta != 0.0 or phi != 0.0:
        qc_target.ry(theta, 0)
        if phi != 0.0:
            qc_target.rz(phi, 0)
    target_sv = Statevector.from_instruction(qc_target)

    qc_full = build_teleportation_circuit(theta, phi)
    full_sv = Statevector.from_instruction(qc_full)

    bob_density_matrix = partial_trace(full_sv, [0, 1])

    fidelity = state_fidelity(target_sv, bob_density_matrix)
    return round(float(fidelity), 6), target_sv, bob_density_matrix


def run_benchmark_states() -> Dict[str, Dict]:
    """
    Stretch Challenge:
    Teleport standard and arbitrary quantum states and verify fidelity == 1.0.
    """
    benchmark_states = [
        ("|0>", 0.0, 0.0),
        ("|1>", math.pi, 0.0),
        ("|+>", math.pi / 2, 0.0),
        ("|->", -math.pi / 2, 0.0),
        ("|i>", math.pi / 2, math.pi / 2),
        ("|arbitrary (theta=1.2, phi=0.8)>", 1.2, 0.8),
    ]

    results = {}
    for label, theta, phi in benchmark_states:
        fid, target_sv, bob_rho = verify_teleportation(theta, phi)
        results[label] = {
            "theta": round(theta, 3),
            "phi": round(phi, 3),
            "fidelity": fid,
            "success": (fid >= 0.9999),
        }
    return results


def simulate_noisy_teleportation(error_prob: float = 0.1) -> float:
    """
    Stretch Challenge:
    Simulate noise on Bob's qubit during channel transit (e.g. depolarizing / bit-flip error on Q2).
    Computes fidelity F between target state |+> and Bob's received density matrix.
    """
    theta, phi = math.pi / 2, 0.0
    qc_target = QuantumCircuit(1)
    qc_target.ry(theta, 0)
    target_sv = Statevector.from_instruction(qc_target)

    qc_clean = build_teleportation_circuit(theta, phi)
    sv_clean = Statevector.from_instruction(qc_clean)
    rho_clean = partial_trace(sv_clean, [0, 1])

    qc_noisy = QuantumCircuit(3)
    qc_noisy.ry(theta, 0)
    qc_noisy.h(1)
    qc_noisy.cx(1, 2)
    qc_noisy.z(2)
    qc_noisy.cx(0, 1)
    qc_noisy.h(0)
    qc_noisy.cx(1, 2)
    qc_noisy.cz(0, 2)
    sv_noisy = Statevector.from_instruction(qc_noisy)
    rho_noisy = partial_trace(sv_noisy, [0, 1])

    mixed_bob = (1.0 - error_prob) * rho_clean.data + error_prob * rho_noisy.data
    noisy_fidelity = float(state_fidelity(target_sv, mixed_bob))
    return round(noisy_fidelity, 4)


if __name__ == "__main__":
    print("=" * 60)
    print(" Challenge 05: Quantum Teleporter")
    print("=" * 60)

    print("\n1. Teleporting Quantum States (Fidelity Verification):")
    bench = run_benchmark_states()
    for state_name, data in bench.items():
        status = "PERFECT (100% Fidelity)" if data["success"] else "FAILED"
        print(f"State {state_name:<34} | Fidelity: {data['fidelity']:.6f} -> {status}")

    print("\n2. Deeper Questions Conceptual Summary:")
    print(" - Why entanglement? It establishes non-local quantum correlations allowing Bob to")
    print("   reconstruct the unknown quantum state once Alice transmits 2 classical bits.")
    print(" - Faster-than-light communication? NO. Bob cannot read the state until Alice's")
    print("   2 classical correction bits arrive (limited by speed of light).")
    print(" - No-cloning theorem preserved? YES. Alice's measurement destroys the original state.")

    print("\n3. Stretch Challenge: Channel Noise Impact on Teleportation:")
    for p in [0.0, 0.05, 0.10, 0.25]:
        fid = simulate_noisy_teleportation(p)
        print(f"Channel Error Rate: {p:>4.2f} | Resulting State Fidelity: {fid:.4f}")
