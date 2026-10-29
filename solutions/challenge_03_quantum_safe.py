"""
Quantum Module 03: The Quantum Safe
SoilQ Quantum Engine

Concepts:
- Target measurement distributions as cryptographic "combinations"
- Reverse-engineering quantum circuits to unlock safes
- Entangling gates (CNOT) and single-qubit rotations (H, X, Z)
- Multiple non-identical circuits yielding identical measurement distributions
- Stretch: Custom safe creator, distance metrics (Total Variation Distance), and safe unlocking engine
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def build_safe_key_circuit_a() -> QuantumCircuit:
    """Primary unlocking circuit: H on qubit 0, CNOT(0, 1)."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_safe_key_circuit_b() -> QuantumCircuit:
    """Alternative unlocking circuit: H on qubit 1, CNOT(1, 0)."""
    qc = QuantumCircuit(2, 2)
    qc.h(1)
    qc.cx(1, 0)
    qc.measure([0, 1], [0, 1])
    return qc


def build_safe_key_circuit_c() -> QuantumCircuit:
    """
    Third distinct circuit producing the same distribution:
    X(0) -> H(0) -> CX(0, 1) [prepares Phi- state, whose measurement is 00: 50%, 11: 50%].
    """
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


class QuantumSafe:
    """
    The Quantum Safe:
    Configured with a target probability distribution and tolerance epsilon.
    Can be tested against any proposed key circuit.
    """
    def __init__(self, target_distribution: Dict[str, float], n_qubits: int = 2, tolerance: float = 0.08):
        self.target = target_distribution
        self.n_qubits = n_qubits
        self.tolerance = tolerance

    def test_key(self, qc: QuantumCircuit, shots: int = 2000) -> Tuple[bool, float, Dict[str, float]]:
        """
        Evaluate candidate key circuit.
        Returns: (unlocked: bool, total_variation_distance: float, observed_distribution: dict)
        """
        sampler = StatevectorSampler()
        job = sampler.run([(qc,)], shots=shots)
        counts = job.result()[0].data.c.get_counts()

        all_keys = set(self.target.keys()).union(counts.keys())
        observed = {k: counts.get(k, 0) / shots for k in all_keys}

        tvd = 0.5 * sum(abs(observed.get(k, 0.0) - self.target.get(k, 0.0)) for k in all_keys)
        unlocked = (tvd <= self.tolerance)
        return unlocked, round(tvd, 4), observed


def solve_bell_anti_safe() -> QuantumCircuit:
    """
    Stretch: Custom safe lock expecting {'01': 0.5, '10': 0.5}.
    Circuit solution: H(0) -> CX(0, 1) -> X(1).
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    return qc


def solve_ghz_3qubit_safe() -> QuantumCircuit:
    """
    Stretch: Custom 3-qubit safe lock expecting {'000': 0.5, '111': 0.5}.
    Circuit solution: H(0) -> CX(0, 1) -> CX(1, 2).
    """
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


if __name__ == "__main__":
    print("=" * 60)
    print(" Quantum Module 03: The Quantum Safe")
    print("=" * 60)

    # 1. Standard Bell Safe: Target {'00': 0.5, '11': 0.5}
    standard_target = {"00": 0.5, "11": 0.5}
    safe = QuantumSafe(standard_target, n_qubits=2)

    print("\n1. Testing Key Circuit A [H(0) -> CX(0, 1)]:")
    unlocked_a, tvd_a, obs_a = safe.test_key(build_safe_key_circuit_a())
    print(f"Status: {'UNLOCKED' if unlocked_a else 'LOCKED'} | TVD: {tvd_a:.4f} | Observed: {obs_a}")

    print("\n2. Testing Key Circuit B [H(1) -> CX(1, 0)]:")
    unlocked_b, tvd_b, obs_b = safe.test_key(build_safe_key_circuit_b())
    print(f"Status: {'UNLOCKED' if unlocked_b else 'LOCKED'} | TVD: {tvd_b:.4f} | Observed: {obs_b}")

    print("\n3. Testing Key Circuit C [X(0) -> H(0) -> CX(0,1)]:")
    unlocked_c, tvd_c, obs_c = safe.test_key(build_safe_key_circuit_c())
    print(f"Status: {'UNLOCKED' if unlocked_c else 'LOCKED'} | TVD: {tvd_c:.4f} | Observed: {obs_c}")

    # 4. Incorrect key
    bad_circuit = QuantumCircuit(2, 2)
    bad_circuit.h(0)
    bad_circuit.measure([0, 1], [0, 1])
    unlocked_bad, tvd_bad, obs_bad = safe.test_key(bad_circuit)
    print(f"\n4. Testing Wrong Key (H(0) without CNOT):")
    print(f"Status: {'UNLOCKED' if unlocked_bad else 'LOCKED'} | TVD: {tvd_bad:.4f} | Observed: {obs_bad}")

    # 5. Stretch Challenge: 3-Qubit GHZ Safe
    print("\n5. Stretch Challenge: 3-Qubit GHZ Safe {'000': 0.5, '111': 0.5}:")
    ghz_safe = QuantumSafe({"000": 0.5, "111": 0.5}, n_qubits=3)
    unlocked_ghz, tvd_ghz, obs_ghz = ghz_safe.test_key(solve_ghz_3qubit_safe())
    print(f"Status: {'UNLOCKED' if unlocked_ghz else 'LOCKED'} | TVD: {tvd_ghz:.4f} | Observed: {obs_ghz}")
