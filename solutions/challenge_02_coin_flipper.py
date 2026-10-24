"""
Challenge 02: Quantum Coin Flipper
Qiskit Fall Fest Nairobi 2026

Concepts:
- 1-qubit state superposition with H-gate: H|0> = (|0> + |1>)/sqrt(2)
- Probability mapping: |0> -> Heads (50%), |1> -> Tails (50%)
- Biased quantum coin via parameterized rotation RY(theta)
- Stretch: 1-Player and 2-Player Quantum Coin Flip Game with score tracking
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def build_fair_coin() -> QuantumCircuit:
    """Build a 1-qubit circuit for a fair coin flip."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def build_biased_coin(prob_tails: float) -> QuantumCircuit:
    """
    Build a biased quantum coin where P(Tails) = prob_tails.
    Uses RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>.
    sin^2(theta/2) = prob_tails => theta = 2 * arcsin(sqrt(prob_tails)).
    """
    if not (0.0 <= prob_tails <= 1.0):
        raise ValueError("prob_tails must be in [0.0, 1.0].")
    theta = 2.0 * math.asin(math.sqrt(prob_tails))
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc


def flip_coin(biased_prob_tails: float | None = None, shots: int = 1) -> List[str]:
    """
    Flip the quantum coin for shots times.
    Returns list of outcomes: 'Heads' (0) or 'Tails' (1).
    """
    qc = build_biased_coin(biased_prob_tails) if biased_prob_tails is not None else build_fair_coin()
    sampler = StatevectorSampler()
    job = sampler.run([(qc,)], shots=shots)
    res = job.result()[0].data.c.get_counts()

    outcomes = []
    for bit, count in res.items():
        label = "Heads" if bit == "0" else "Tails"
        outcomes.extend([label] * count)
    return outcomes


def run_coin_experiments(shot_levels: List[int] = None) -> Dict[int, Dict[str, float]]:
    """Record percentage of Heads and Tails across shot counts."""
    if shot_levels is None:
        shot_levels = [10, 100, 1000, 10000]
    results = {}
    sampler = StatevectorSampler()
    qc = build_fair_coin()

    for shots in shot_levels:
        job = sampler.run([(qc,)], shots=shots)
        counts = job.result()[0].data.c.get_counts()
        heads = counts.get("0", 0)
        tails = counts.get("1", 0)
        results[shots] = {
            "heads_pct": round(heads / shots * 100, 2),
            "tails_pct": round(tails / shots * 100, 2),
            "total_shots": shots,
        }
    return results


class QuantumCoinGame:
    """
    Stretch Challenge:
    Interactive Quantum Coin Flip Game with 1-Player and 2-Player modes.
    """
    def __init__(self):
        self.p1_score = 0
        self.p2_score = 0
        self.rounds_played = 0
        self.history: List[Dict] = []

    def play_round_single_player(self, player_choice: str) -> Dict:
        """Player guesses Heads or Tails against fair quantum coin."""
        choice = player_choice.strip().capitalize()
        if choice not in ("Heads", "Tails"):
            raise ValueError("Choice must be 'Heads' or 'Tails'.")

        result = flip_coin(shots=1)[0]
        won = (choice == result)
        if won:
            self.p1_score += 1
        self.rounds_played += 1

        entry = {
            "round": self.rounds_played,
            "mode": "1-player",
            "player_choice": choice,
            "quantum_result": result,
            "won": won,
            "player_score": self.p1_score,
        }
        self.history.append(entry)
        return entry

    def play_round_two_player(self, p1_choice: str, p2_choice: str) -> Dict:
        """Two players guess outcomes against fair quantum coin."""
        c1 = p1_choice.strip().capitalize()
        c2 = p2_choice.strip().capitalize()
        for c in (c1, c2):
            if c not in ("Heads", "Tails"):
                raise ValueError("Choice must be 'Heads' or 'Tails'.")

        result = flip_coin(shots=1)[0]
        p1_won = (c1 == result)
        p2_won = (c2 == result)
        if p1_won:
            self.p1_score += 1
        if p2_won:
            self.p2_score += 1
        self.rounds_played += 1

        entry = {
            "round": self.rounds_played,
            "mode": "2-player",
            "p1_choice": c1,
            "p2_choice": c2,
            "quantum_result": result,
            "p1_won": p1_won,
            "p2_won": p2_won,
            "p1_score": self.p1_score,
            "p2_score": self.p2_score,
        }
        self.history.append(entry)
        return entry

    def get_stats(self) -> Dict:
        return {
            "rounds_played": self.rounds_played,
            "p1_score": self.p1_score,
            "p2_score": self.p2_score,
            "p1_win_rate": round(self.p1_score / max(1, self.rounds_played), 3),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 02: Quantum Coin Flipper")
    print("=" * 60)

    print("\n1. Fair Coin Shot Experiments:")
    exp = run_coin_experiments()
    for s, d in exp.items():
        print(f"Shots: {s:>5} | Heads: {d['heads_pct']:>5.1f}% | Tails: {d['tails_pct']:>5.1f}%")

    print("\n2. Biased Coin Experiment (Target: 75% Tails):")
    biased_out = flip_coin(biased_prob_tails=0.75, shots=1000)
    tails_pct = biased_out.count("Tails") / 10.0
    print(f"1000 shots on 75% biased coin -> Observed Tails: {tails_pct:.1f}%")

    print("\n3. Stretch Challenge: Quantum Coin Flip Game (Simulated 5 rounds):")
    game = QuantumCoinGame()
    for r in range(1, 6):
        res = game.play_round_single_player("Heads")
        print(f"Round {r}: Guessed Heads, Result was {res['quantum_result']} -> {'WON' if res['won'] else 'LOST'}")
    print("Final Stats:", game.get_stats())
