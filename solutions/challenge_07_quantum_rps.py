"""
Challenge 07: Quantum Rock, Paper, Scissors
Qiskit Fall Fest Nairobi 2026

Concepts:
- Replacing pseudo-random choices with true quantum measurement
- Encoding 3 discrete choices onto a 2-qubit Hilbert space:
  00 -> Rock
  01 -> Paper
  10 -> Scissors
  11 -> Unused state
- The Bias Problem:
  Mapping 11 -> Rock yields 50% Rock vs 25% Paper/Scissors (severe bias!).
- Unbiased generation solutions:
  1. Rejection sampling (re-sample if 11 is observed)
  2. Unitary state preparation: exact uniform qutrit (|00> + |01> + |10>)/sqrt(3)
- Stretch: Full interactive game engine with round history, score tracking,
  and win/loss statistics
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector


CHOICE_MAP = {
    "00": "Rock",
    "01": "Paper",
    "10": "Scissors",
}

BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def build_naive_rps_circuit() -> QuantumCircuit:
    """Standard 2-qubit superposition (H on both qubits). Produces 00, 01, 10, 11 equally."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_exact_unbiased_qutrit_circuit() -> QuantumCircuit:
    """
    Stretch (Quantum Unitary approach):
    Directly prepare (|00> + |01> + |10>) / sqrt(3) with ZERO amplitude on |11>.
    1. RY(theta) on Q0 where theta = 2 * arccos(1 / sqrt(3)) ~ 1.91063 rad
       State: 1/sqrt(3)|0> + sqrt(2/3)|1>
    2. Controlled-H with Q0 as control, Q1 as target:
       State: 1/sqrt(3)|00> + 1/sqrt(3)|10> + 1/sqrt(3)|11>
    3. X on Q1, then CNOT(1, 0) swaps |11> into |01>:
       Exact state: (|00> + |01> + |10>) / sqrt(3).
    """
    qc = QuantumCircuit(2, 2)
    theta = 2.0 * math.acos(1.0 / math.sqrt(3.0))
    qc.ry(theta, 0)
    qc.ch(0, 1)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def get_quantum_choice(method: str = "rejection", max_retries: int = 100) -> str:
    """
    Generate an unbiased quantum choice for the computer.
    method='rejection': sample 2-qubit H gates, reject '11'.
    method='unitary': sample exact 3-state superposition circuit.
    """
    sampler = StatevectorSampler()
    if method == "unitary":
        qc = build_exact_unbiased_qutrit_circuit()
        job = sampler.run([(qc,)], shots=1)
        res = job.result()[0].data.c.get_counts()
        bitstr = list(res.keys())[0]
        return CHOICE_MAP.get(bitstr, "Rock")

    qc = build_naive_rps_circuit()
    for _ in range(max_retries):
        job = sampler.run([(qc,)], shots=1)
        res = job.result()[0].data.c.get_counts()
        bitstr = list(res.keys())[0]
        if bitstr in CHOICE_MAP:
            return CHOICE_MAP[bitstr]

    return "Rock"


def determine_winner(player: str, computer: str) -> str:
    """Return 'player', 'computer', or 'tie'."""
    p = player.strip().capitalize()
    c = computer.strip().capitalize()
    if p == c:
        return "tie"
    if BEATS.get(p) == c:
        return "player"
    return "computer"


class QuantumRPSGame:
    """
    Stretch Challenge:
    Full Quantum Rock-Paper-Scissors game with score tracking and stats.
    """
    def __init__(self, method: str = "rejection"):
        self.method = method
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0
        self.history: List[Dict] = []

    def play_round(self, player_move: str) -> Dict:
        """Play one round with a player move."""
        move = player_move.strip().capitalize()
        if move not in ("Rock", "Paper", "Scissors"):
            raise ValueError(f"Invalid move '{player_move}'. Choose Rock, Paper, or Scissors.")

        computer_move = get_quantum_choice(method=self.method)
        result = determine_winner(move, computer_move)

        if result == "player":
            self.player_score += 1
        elif result == "computer":
            self.computer_score += 1
        else:
            self.ties += 1

        entry = {
            "round": len(self.history) + 1,
            "player_move": move,
            "computer_move": computer_move,
            "winner": result,
            "player_score": self.player_score,
            "computer_score": self.computer_score,
            "ties": self.ties,
        }
        self.history.append(entry)
        return entry

    def get_stats(self) -> Dict:
        total = max(1, len(self.history))
        return {
            "total_rounds": len(self.history),
            "player_score": self.player_score,
            "computer_score": self.computer_score,
            "ties": self.ties,
            "player_win_rate": round(self.player_score / total, 3),
            "computer_win_rate": round(self.computer_score / total, 3),
            "tie_rate": round(self.ties / total, 3),
        }


def analyze_rps_bias() -> Dict[str, Dict[str, float]]:
    """
    Deeper Question Analysis:
    Compare probability distribution of Naive Mapping (11 -> Rock) vs Unbiased Quantum methods.
    """
    naive_probs = {"Rock": 0.50, "Paper": 0.25, "Scissors": 0.25}
    rejection_probs = {"Rock": 1 / 3, "Paper": 1 / 3, "Scissors": 1 / 3}

    counts = {"Rock": 0, "Paper": 0, "Scissors": 0}
    for _ in range(300):
        c = get_quantum_choice("rejection")
        counts[c] += 1
    empirical_probs = {k: round(v / 300, 3) for k, v in counts.items()}

    return {
        "naive_mapping_biased": naive_probs,
        "unbiased_theoretical": {k: round(v, 4) for k, v in rejection_probs.items()},
        "unbiased_empirical_300_draws": empirical_probs,
    }


if __name__ == "__main__":
    print("=" * 60)
    print(" Challenge 07: Quantum Rock, Paper, Scissors")
    print("=" * 60)

    # 1. Deeper question: Bias Analysis
    print("\n1. Deeper Question: The Bias Problem of State |11>:")
    bias_data = analyze_rps_bias()
    print("Naive Mapping (11 -> Rock) probabilities :", bias_data["naive_mapping_biased"])
    print("Unbiased Quantum theoretical probabilities:", bias_data["unbiased_theoretical"])
    print("Unbiased Quantum empirical (300 draws) :", bias_data["unbiased_empirical_300_draws"])

    # 2. Stretch Challenge: 5-Round Match Simulation
    print("\n2. Stretch Challenge: Quantum RPS Game Match (5 Rounds):")
    game = QuantumRPSGame(method="rejection")
    moves = ["Rock", "Paper", "Scissors", "Rock", "Paper"]
    for m in moves:
        res = game.play_round(m)
        w_str = (
            "YOU WON"
            if res["winner"] == "player"
            else ("COMPUTER WON" if res["winner"] == "computer" else "TIE")
        )
        print(f"Round {res['round']}: You played {res['player_move']:<8} vs Quantum {res['computer_move']:<8} -> {w_str}")

    print("\nFinal Match Statistics:", game.get_stats())
