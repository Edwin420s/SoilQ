"""
Challenge 01: Quantum Random Number Generator (QRNG)
Qiskit Fall Fest Nairobi 2026

Concepts:
- Superposition via Hadamard (H) gates
- Measurement collapse to classical bits
- True quantum randomness vs classical pseudo-randomness
- Stretch: Unbiased arbitrary-range random integer generator (rejection sampling)
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def build_qrng_circuit(n_qubits: int) -> QuantumCircuit:
    """Create an n-qubit circuit with H gates on each qubit and measurement."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def sample_qrng(n_qubits: int, shots: int = 1024) -> Dict[str, int]:
    """Run the QRNG circuit and return measurement counts."""
    qc = build_qrng_circuit(n_qubits)
    sampler = StatevectorSampler()
    job = sampler.run([(qc,)], shots=shots)
    result = job.result()
    return result[0].data.c.get_counts()


def run_shot_experiments(n_qubits: int = 3, shot_levels: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Run the QRNG across varying shot counts (10, 100, 1000, 10000)
    and compute observed probabilities vs ideal uniform distribution.
    """
    if shot_levels is None:
        shot_levels = [10, 100, 1000, 10000]
    num_states = 2 ** n_qubits
    ideal_p = 1.0 / num_states
    summary = {}

    for shots in shot_levels:
        counts = sample_qrng(n_qubits, shots=shots)
        probs = {k: counts.get(k, 0) / shots for k in [format(i, f"0{n_qubits}b") for i in range(num_states)]}
        max_deviation = max(abs(p - ideal_p) for p in probs.values())
        summary[shots] = {
            "counts": counts,
            "probabilities": probs,
            "max_deviation": round(max_deviation, 4),
            "ideal_p": ideal_p,
        }
    return summary


def quantum_random(min_val: int, max_val: int, max_retries: int = 100) -> int:
    """
    Stretch Challenge:
    Generate an unbiased random integer in [min_val, max_val] using quantum measurement
    with rejection sampling to prevent modulo bias for non-power-of-two ranges.
    """
    if min_val > max_val:
        raise ValueError("min_val must be <= max_val.")
    if min_val == max_val:
        return min_val

    range_size = max_val - min_val + 1
    k = math.ceil(math.log2(range_size))
    sampler = StatevectorSampler()
    qc = build_qrng_circuit(k)

    for _ in range(max_retries):
        job = sampler.run([(qc,)], shots=1)
        res = job.result()[0].data.c.get_counts()
        bitstr = list(res.keys())[0]
        val = int(bitstr, 2)
        if val < range_size:
            return min_val + val

    return min_val + (val % range_size)


def analyze_modulo_bias(min_val: int = 0, max_val: int = 2) -> Dict[str, float]:
    """
    Demonstrate mathematical reason for rejection sampling:
    If sampling 2 qubits (0..3) to cover range [0..2] (3 states):
    - With naive modulo: 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 0 (state 0 gets 50% vs 25% each for 1 and 2).
    - With rejection sampling: each of {0, 1, 2} gets exactly 1/3 (33.3%).
    """
    range_size = max_val - min_val + 1
    k = math.ceil(math.log2(range_size))
    total_states = 2 ** k
    modulo_counts = {i: 0 for i in range(range_size)}
    for val in range(total_states):
        mapped = val % range_size
        modulo_counts[mapped] += 1

    return {
        "range_size": range_size,
        "qubits_used": k,
        "total_quantum_states": total_states,
        "naive_modulo_probs": {min_val + k: v / total_states for k, v in modulo_counts.items()},
        "unbiased_ideal_probs": {min_val + k: 1.0 / range_size for k in range(range_size)},
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 01: Quantum Random Number Generator (QRNG)")
    print("=" * 60)

    print("\n1. Sampling 3-qubit QRNG (8 outcomes, 1000 shots):")
    counts = sample_qrng(3, shots=1000)
    print("Counts:", counts)

    print("\n2. Shot Experiments (Convergence to uniform distribution):")
    exp = run_shot_experiments(2, [10, 100, 1000, 10000])
    for s, data in exp.items():
        print(f"Shots: {s:>5} | Max deviation from ideal (0.25): {data['max_deviation']:.4f}")

    print("\n3. Stretch Challenge: quantum_random(1, 6) [Die roll]:")
    rolls = [quantum_random(1, 6) for _ in range(12)]
    print("12 Quantum Die Rolls:", rolls)

    print("\n4. Modulo Bias Analysis for Range [0, 2]:")
    bias = analyze_modulo_bias(0, 2)
    print("Naive Modulo Probabilities:", bias["naive_modulo_probs"])
    print("Unbiased Rejection Probabilities:", bias["unbiased_ideal_probs"])
