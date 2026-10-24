"""
Automated Pytest Test Suite for all 8 Hackathon Challenges.
Qiskit Fall Fest Nairobi 2026
"""
import math
import numpy as np
import pytest
from pathlib import Path
import sys

SOLUTIONS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLUTIONS_DIR))

from challenge_01_qrng import build_qrng_circuit, sample_qrng, quantum_random, analyze_modulo_bias
from challenge_02_coin_flipper import build_fair_coin, build_biased_coin, flip_coin, QuantumCoinGame
from challenge_03_quantum_safe import build_safe_key_circuit_a, build_safe_key_circuit_b, build_safe_key_circuit_c, QuantumSafe, solve_ghz_3qubit_safe
from challenge_04_entanglement_detector import build_phi_plus, build_phi_minus, build_psi_plus, build_psi_minus, run_circuit_sampling, calculate_concurrence
from challenge_05_quantum_teleporter import verify_teleportation, run_benchmark_states, simulate_noisy_teleportation
from challenge_06_quantum_treasure_hunt import find_treasure, compare_classical_vs_quantum
from challenge_07_quantum_rps import get_quantum_choice, determine_winner, QuantumRPSGame
from challenge_08_quantum_classifier import SimpleVariationalQuantumClassifier, generate_toy_dataset


# ── Challenge 01 Tests ────────────────────────────────────────────────────────
def test_c01_qrng_circuit():
    qc = build_qrng_circuit(3)
    assert qc.num_qubits == 3
    counts = sample_qrng(3, shots=500)
    assert len(counts) > 0
    assert sum(counts.values()) == 500


def test_c01_quantum_random_range():
    # Stretch: test range [10, 15]
    val = quantum_random(10, 15)
    assert 10 <= val <= 15


def test_c01_modulo_bias_analysis():
    bias = analyze_modulo_bias(0, 2)
    assert bias["range_size"] == 3
    assert bias["naive_modulo_probs"][0] == 0.5


# ── Challenge 02 Tests ────────────────────────────────────────────────────────
def test_c02_coin_flipper():
    outcomes = flip_coin(shots=100)
    assert len(outcomes) == 100
    assert set(outcomes).issubset({"Heads", "Tails"})


def test_c02_biased_coin():
    # 100% Tails coin
    tails = flip_coin(biased_prob_tails=1.0, shots=20)
    assert all(o == "Tails" for o in tails)


def test_c02_game_stretch():
    game = QuantumCoinGame()
    res = game.play_round_single_player("Heads")
    assert res["round"] == 1
    assert "won" in res


# ── Challenge 03 Tests ────────────────────────────────────────────────────────
def test_c03_quantum_safe_unlocking():
    safe = QuantumSafe({"00": 0.5, "11": 0.5}, n_qubits=2)
    unlocked_a, tvd_a, _ = safe.test_key(build_safe_key_circuit_a(), shots=1000)
    unlocked_b, tvd_b, _ = safe.test_key(build_safe_key_circuit_b(), shots=1000)
    unlocked_c, tvd_c, _ = safe.test_key(build_safe_key_circuit_c(), shots=1000)
    assert unlocked_a
    assert unlocked_b
    assert unlocked_c


def test_c03_ghz_safe_stretch():
    ghz_safe = QuantumSafe({"000": 0.5, "111": 0.5}, n_qubits=3)
    unlocked, tvd, _ = ghz_safe.test_key(solve_ghz_3qubit_safe(), shots=1000)
    assert unlocked


# ── Challenge 04 Tests ────────────────────────────────────────────────────────
def test_c04_bell_states_correlation():
    _, corr_phi_plus = run_circuit_sampling(build_phi_plus(), shots=1000)
    assert corr_phi_plus > 0.8  # strongly correlated

    _, corr_psi_plus = run_circuit_sampling(build_psi_plus(), shots=1000)
    assert corr_psi_plus < -0.8  # strongly anti-correlated


def test_c04_concurrence_stretch():
    from qiskit import QuantumCircuit
    # Maximally entangled Bell state -> Concurrence == 1.0
    qc_bell = QuantumCircuit(2)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    conc = calculate_concurrence(qc_bell)
    assert pytest.approx(conc, abs=1e-3) == 1.0


# ── Challenge 05 Tests ────────────────────────────────────────────────────────
def test_c05_teleportation_fidelity():
    fid_0, _, _ = verify_teleportation(0.0, 0.0)
    assert pytest.approx(fid_0, abs=1e-4) == 1.0

    fid_arb, _, _ = verify_teleportation(1.23, 0.45)
    assert pytest.approx(fid_arb, abs=1e-4) == 1.0


def test_c05_noise_degradation():
    fid_clean = simulate_noisy_teleportation(0.0)
    fid_noisy = simulate_noisy_teleportation(0.2)
    assert fid_clean > fid_noisy


# ── Challenge 06 Tests ────────────────────────────────────────────────────────
def test_c06_grover_all_locations():
    for loc in ["00", "01", "10", "11"]:
        counts, p = find_treasure(loc, n_qubits=2, shots=500)
        assert p >= 0.95


def test_c06_complexity_stretch():
    comp = compare_classical_vs_quantum(2)
    assert comp["classical_average_queries"] > comp["quantum_grover_queries"]


# ── Challenge 07 Tests ────────────────────────────────────────────────────────
def test_c07_unbiased_choice():
    choices = [get_quantum_choice() for _ in range(30)]
    assert set(choices).issubset({"Rock", "Paper", "Scissors"})


def test_c07_rules():
    assert determine_winner("Rock", "Scissors") == "player"
    assert determine_winner("Paper", "Rock") == "player"
    assert determine_winner("Scissors", "Paper") == "player"
    assert determine_winner("Rock", "Rock") == "tie"
    assert determine_winner("Rock", "Paper") == "computer"


def test_c07_game_stretch():
    game = QuantumRPSGame()
    res = game.play_round("Rock")
    assert res["round"] == 1
    assert res["winner"] in ("player", "computer", "tie")


# ── Challenge 08 Tests ────────────────────────────────────────────────────────
def test_c08_quantum_classifier_fit_predict():
    X, y = generate_toy_dataset(20, seed=1)
    vqc = SimpleVariationalQuantumClassifier(reps=1, maxiter=15, seed=1)
    vqc.fit(X, y)
    preds = vqc.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})
