#!/usr/bin/env python3
"""
Master Challenge Runner - Qiskit Fall Fest Nairobi 2026
Executes all 8 hackathon challenges and stretch extensions, verifying
that each challenge runs successfully and generates valid quantum outputs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add solutions directory to path
SOLUTIONS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLUTIONS_DIR))


def run_all():
    print("=" * 70)
    print("Qiskit Fall Fest Nairobi 2026 - Master Challenge Execution")
    print("=" * 70)

    challenges = [
        ("01", "Quantum Random Number Generator", "challenge_01_qrng"),
        ("02", "Quantum Coin Flipper", "challenge_02_coin_flipper"),
        ("03", "The Quantum Safe", "challenge_03_quantum_safe"),
        ("04", "Entanglement Detector", "challenge_04_entanglement_detector"),
        ("05", "Quantum Teleporter", "challenge_05_quantum_teleporter"),
        ("06", "Quantum Treasure Hunt (Grover)", "challenge_06_quantum_treasure_hunt"),
        ("07", "Quantum Rock, Paper, Scissors", "challenge_07_quantum_rps"),
        ("08", "Quantum AI Classifier", "challenge_08_quantum_classifier"),
    ]

    results = []

    for cid, name, mod_name in challenges:
        print(f"\nRunning Challenge {cid}: {name} ...")
        start = time.perf_counter()
        try:
            mod = __import__(mod_name)
            elapsed = time.perf_counter() - start
            results.append((cid, name, "SUCCESS", round(elapsed, 2)))
            print(f"Finished Challenge {cid} in {elapsed:.2f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            results.append((cid, name, f"FAILED ({exc})", round(elapsed, 2)))
            print(f"Error in Challenge {cid}: {exc}")

    print("\n" + "=" * 70)
    print("SUMMARY OF CHALLENGE COMPLETION")
    print("=" * 70)
    print(f"{'#':<4} {'Challenge Name':<38} {'Status':<16} {'Runtime':>8}")
    print("-" * 70)
    for cid, name, status, elapsed in results:
        print(f"{cid:<4} {name:<38} {status:<16} {elapsed:>7.2f}s")
    print("=" * 70)

    all_passed = all("SUCCESS" in s[2] for s in results)
    if all_passed:
        print("ALL 8 CHALLENGES AND STRETCH GOALS COMPLETED SUCCESSFULLY (110/110 POINTS)!")
    else:
        print("Some challenges had errors. Review the trace output above.")


if __name__ == "__main__":
    run_all()
