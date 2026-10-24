"""
Challenge 04: Entanglement Detector
Qiskit Fall Fest Nairobi 2026

Concepts:
- Quantum Entanglement: non-separable bipartite states
- Correlation coefficient: C = P(00) + P(11) - (P(01) + P(10))
  C ~ +1: Strongly correlated (|Phi+>, |Phi->)
  C ~ -1: Strongly anti-correlated (|Psi+>, |Psi->)
  C ~ 0: Unentangled / separable (|00>, |+0>)
- Deeper experiments: removing H, removing CNOT, adding X
- Stretch: The 4 Bell States, Concurrence / Entanglement quantification, and noise simulation
"""
from __future__ import annotations

from typing import Dict, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler


def compute_correlation(counts: Dict[str, int], total_shots: int) -> float:
    """Compute normalized correlation metric C in [-1.0, 1.0]."""
    p00 = counts.get("00", 0) / total_shots
    p11 = counts.get("11", 0) / total_shots
    p01 = counts.get("01", 0) / total_shots
    p10 = counts.get("10", 0) / total_shots
    return (p00 + p11) - (p01 + p10)


def build_phi_plus() -> QuantumCircuit:
    """|Phi+> = (|00> + |11>) / sqrt(2)."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_phi_minus() -> QuantumCircuit:
    """|Phi-> = (|00> - |11>) / sqrt(2)."""
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_psi_plus() -> QuantumCircuit:
    """|Psi+> = (|01> + |10>) / sqrt(2)."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_psi_minus() -> QuantumCircuit:
    """|Psi-> = (|01> - |10>) / sqrt(2)."""
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    return qc


def run_circuit_sampling(qc: QuantumCircuit, shots: int = 2000) -> Tuple[Dict[str, int], float]:
    """Sample circuit and return (counts, correlation)."""
    sampler = StatevectorSampler()
    job = sampler.run([(qc,)], shots=shots)
    counts = job.result()[0].data.c.get_counts()
    corr = compute_correlation(counts, shots)
    return counts, round(corr, 4)


def calculate_concurrence(qc_no_meas: QuantumCircuit) -> float:
    """
    Calculate Wootters Concurrence for pure 2-qubit statevector.
    C = 2 * |c00*c11 - c01*c10|.
    Returns 1.0 for maximally entangled Bell states, 0.0 for product states.
    """
    sv = Statevector.from_instruction(qc_no_meas)
    data = sv.data
    c00, c01, c10, c11 = data[0], data[1], data[2], data[3]
    concurrence = 2.0 * abs(c00 * c11 - c01 * c10)
    return round(float(concurrence), 4)


def run_deeper_experiments() -> Dict[str, Dict]:
    """
    Test variations:
    1. Standard Bell (|Phi+>)
    2. Remove H-gate (Deterministic product state |00>)
    3. Remove CNOT (Separable superposition |+0>)
    4. Swapped order: CNOT then H
    """
    experiments = {}

    # 1. Standard
    qc1 = build_phi_plus()
    counts1, corr1 = run_circuit_sampling(qc1)
    experiments["standard_bell"] = {"desc": "H(0) -> CX(0,1)", "counts": counts1, "correlation": corr1}

    # 2. No H gate
    qc2 = QuantumCircuit(2, 2)
    qc2.cx(0, 1)
    qc2.measure([0, 1], [0, 1])
    counts2, corr2 = run_circuit_sampling(qc2)
    experiments["no_h_gate"] = {"desc": "CX(0,1) only (Deterministic |00>)", "counts": counts2, "correlation": corr2}

    # 3. No CNOT
    qc3 = QuantumCircuit(2, 2)
    qc3.h(0)
    qc3.measure([0, 1], [0, 1])
    counts3, corr3 = run_circuit_sampling(qc3)
    experiments["no_cnot"] = {"desc": "H(0) only (Separable |+0>)", "counts": counts3, "correlation": corr3}

    # 4. Swapped order
    qc4 = QuantumCircuit(2, 2)
    qc4.cx(0, 1)
    qc4.h(0)
    qc4.measure([0, 1], [0, 1])
    counts4, corr4 = run_circuit_sampling(qc4)
    experiments["swapped_order"] = {"desc": "CX(0,1) then H(0)", "counts": counts4, "correlation": corr4}

    return experiments


def run_bell_states_comparison() -> Dict[str, Dict]:
    """
    Stretch Challenge:
    Compare all 4 Bell states: circuits, measurement counts, correlation, and concurrence.
    """
    bell_configs = [
        ("Phi+", build_phi_plus, "H(0) -> CX(0,1)"),
        ("Phi-", build_phi_minus, "X(0) -> H(0) -> CX(0,1)"),
        ("Psi+", build_psi_plus, "H(0) -> CX(0,1) -> X(1)"),
        ("Psi-", build_psi_minus, "X(0) -> H(0) -> CX(0,1) -> X(1)"),
    ]
    results = {}
    for name, builder, circuit_desc in bell_configs:
        qc_meas = builder()
        qc_pure = QuantumCircuit(2)
        if "X(0)" in circuit_desc:
            qc_pure.x(0)
        qc_pure.h(0)
        qc_pure.cx(0, 1)
        if "X(1)" in circuit_desc:
            qc_pure.x(1)

        counts, corr = run_circuit_sampling(qc_meas)
        conc = calculate_concurrence(qc_pure)
        results[name] = {
            "circuit": circuit_desc,
            "counts": counts,
            "correlation": corr,
            "concurrence": conc,
            "interpretation": (
                "Correlated (00, 11)" if corr > 0.8 else "Anti-Correlated (01, 10)"
            ),
        }
    return results


if __name__ == "__main__":
    print("=" * 60)
    print(" Challenge 04: Entanglement Detector")
    print("=" * 60)

    print("\n1. Deeper Questions Experiment (Dissecting Entanglement):")
    deeper = run_deeper_experiments()
    for name, data in deeper.items():
        print(f"[{name:<16}] {data['desc']:<34} | Corr: {data['correlation']:>6.2f} | Counts: {data['counts']}")

    print("\n2. Stretch Challenge: All 4 Bell States Comparison:")
    bell = run_bell_states_comparison()
    for b_name, b_data in bell.items():
        print(f"Bell State |{b_name}> | Concurrence: {b_data['concurrence']:.2f} | Corr: {b_data['correlation']:>6.2f} | {b_data['interpretation']}")
        print(f" Counts: {b_data['counts']}")
