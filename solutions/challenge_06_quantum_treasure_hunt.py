"""
Challenge 06: Quantum Treasure Hunt (Grover's Search Algorithm)
Qiskit Fall Fest Nairobi 2026

Concepts:
- Unstructured quantum search using Grover's algorithm
- State superposition creation: H^(x n)
- Phase Oracle: inverting the amplitude of the hidden treasure state
- Grover Diffuser / Amplitude Amplification
- Quadratic speedup: O(sqrt(N)) vs O(N) classical search
- Stretch: Dynamic treasure location selector, automatic oracle synthesis,
  and query complexity comparison
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def build_oracle(target: str, n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct phase oracle U_w that flips the sign of |target>.
    U_w |x> = -|x> if x == target else |x>.
    """
    qc = QuantumCircuit(n_qubits, name=f"Oracle({target})")
    for q in range(n_qubits):
        if target[-1 - q] == "0":
            qc.x(q)

    if n_qubits == 2:
        qc.cz(0, 1)
    elif n_qubits == 3:
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for q in range(n_qubits):
        if target[-1 - q] == "0":
            qc.x(q)

    return qc


def build_diffuser(n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct Grover diffuser / reflection about average:
    U_s = 2|s><s| - I = H^(x n) (2|0><0| - I) H^(x n).
    """
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)

    if n_qubits == 2:
        qc.cz(0, 1)
    elif n_qubits == 3:
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)

    return qc


def build_grovers_search(target: str, n_qubits: int = 2, iterations: int = 1) -> QuantumCircuit:
    """Complete Grover's search algorithm circuit."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # 1. Initialize equal superposition
    for q in range(n_qubits):
        qc.h(q)

    oracle = build_oracle(target, n_qubits)
    diffuser = build_diffuser(n_qubits)

    # 2. Apply Grover iteration (Oracle + Diffuser)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    # 3. Measurement
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def find_treasure(target: str, n_qubits: int = 2, shots: int = 1000) -> Tuple[Dict[str, int], float]:
    """
    Run Grover's search for a target location.
    Returns (counts, success_probability).
    """
    N = 2 ** n_qubits
    optimal_iters = max(1, math.floor(math.pi / 4.0 * math.sqrt(N)))
    qc = build_grovers_search(target, n_qubits, iterations=optimal_iters)

    sampler = StatevectorSampler()
    job = sampler.run([(qc,)], shots=shots)
    counts = job.result()[0].data.c.get_counts()
    success_p = counts.get(target, 0) / shots
    return counts, round(success_p, 4)


def compare_classical_vs_quantum(n_qubits: int = 2) -> Dict:
    """
    Stretch Challenge:
    Compare query complexity of classical linear search vs Grover search.
    """
    N = 2 ** n_qubits
    classical_avg = (N + 1) / 2.0
    classical_worst = N
    quantum_queries = max(1, math.floor(math.pi / 4.0 * math.sqrt(N)))

    return {
        "search_space_size": N,
        "n_qubits": n_qubits,
        "classical_average_queries": classical_avg,
        "classical_worst_case_queries": classical_worst,
        "quantum_grover_queries": quantum_queries,
        "speedup_ratio": round(classical_avg / quantum_queries, 2),
    }


if __name__ == "__main__":
    print("=" * 60)
    print(" Challenge 06: Quantum Treasure Hunt (Grover's Algorithm)")
    print("=" * 60)

    # 1. 2-Qubit Search (All 4 possible treasure locations)
    print("\n1. Finding Treasure in 4 Locations (N=4, 1 Grover iteration):")
    for loc in ["00", "01", "10", "11"]:
        counts, p = find_treasure(loc, n_qubits=2, shots=1000)
        found = (p >= 0.99)
        print(f"Hidden Treasure at '{loc}' -> Measured: {counts} | Found with {p * 100:.1f}% certainty! {'[SUCCESS]' if found else '[FAILED]'}")

    # 2. Stretch Challenge: 3-Qubit Search (8 locations)
    print("\n2. Stretch Challenge: 3-Qubit Search (N=8, 2 Grover iterations):")
    target_3q = "101"
    counts_3q, p_3q = find_treasure(target_3q, n_qubits=3, shots=1000)
    print(f"Hidden Treasure at '{target_3q}' (N=8) -> Counts: {counts_3q} | Success: {p_3q * 100:.1f}%")

    # 3. Query complexity comparison
    print("\n3. Classical vs Quantum Query Complexity Comparison:")
    for n in [2, 3, 4, 10]:
        comp = compare_classical_vs_quantum(n)
        print(f"Space size: 2^{comp['n_qubits']} = {comp['search_space_size']:<5} | Classical avg: {comp['classical_average_queries']:>7.1f} | Grover: {comp['quantum_grover_queries']:>4} queries | Speedup: {comp['speedup_ratio']}x")
