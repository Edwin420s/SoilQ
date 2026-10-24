#  Qiskit Fall Fest Nairobi 2026 — Complete Challenge Solutions

This directory contains full, executable, and automated solutions for **all 8 challenges** of the **Qiskit Fall Fest Nairobi 2026 Hackathon**, designed to achieve the maximum score (**110 / 110 points**: 100 base points + 10 bonus stretch points) under the official [Judging Criteria](../JUDGING_CRITERIA.md).

---

## Challenge Summary & Status

| # | Challenge | Key Quantum Concepts | Stretch Challenge | Status |
|---|---|---|---|:---:|
| **01** | [QRNG](challenge_01_qrng.py) | Superposition ($H^{\otimes n}$), Born rule measurement collapse | Unbiased arbitrary-range random integer (`quantum_random(min, max)`) with rejection sampling to eliminate modulo bias | Passed |
| **02** | [Quantum Coin Flipper](challenge_02_coin_flipper.py) | 1-qubit $|+\rangle$ state, probabilistic outcome | Biased coin via $RY(\theta)$ rotation + Interactive 1-Player & 2-Player Coin Flip Game with score keeping | Passed |
| **03** | [The Quantum Safe](challenge_03_quantum_safe.py) | Entanglement, Bell state distributions | Multiple non-identical unlocking circuits + Custom Safe unlocker & 3-qubit GHZ state safe | Passed |
| **04** | [Entanglement Detector](challenge_04_entanglement_detector.py) | Non-separable states, Bell pairs, correlation coefficient $C$ | All 4 Bell states ($|\Phi^+\rangle, |\Phi^-\rangle, |\Psi^+\rangle, |\Psi^-\rangle$) comparison, Wootters Concurrence metric, noise impact | Passed |
| **05** | [Quantum Teleporter](challenge_05_quantum_teleporter.py) | EPR pairs, Bell-basis measurement, classical feedforward corrections | Teleporting arbitrary states ($|0\rangle, |1\rangle, |+\rangle, |-\rangle, |i\rangle, |\psi(\theta, \phi)\rangle$) with 100% fidelity + transit noise modeling | Passed |
| **06** | [Quantum Treasure Hunt](challenge_06_quantum_treasure_hunt.py) | Grover's search algorithm, Phase Oracle $U_\omega$, Diffuser $U_s$ | Dynamic user-specified treasure location on 2 & 3 qubits (100% and 94.3% certainty) + Classical vs Grover query complexity analysis | Passed |
| **07** | [Quantum Rock, Paper, Scissors](challenge_07_quantum_rps.py) | Mapping 3 choices to 2 qubits, handling the unused $|11\rangle$ state | The Bias Problem analysis + Unbiased rejection & unitary qutrit state preparation + Full multi-round game engine | Passed |
| **08** | [Quantum AI Classifier](challenge_08_quantum_classifier.py) | Quantum machine learning, $RY(\pi \cdot x_i)$ feature encoding, VQC | Benchmark against classical models (LR, SVM) + Real-world Kenyan maize agronomy dataset ([SoilQ Project](../../SoilQ%201/)) | Passed |

---

## Running the Solutions & Tests

### Execute Master Runner (All 8 Challenges)
```bash
python solutions/run_all_challenges.py
```

### Run Automated Pytest Suite
```bash
pytest solutions/test_challenges.py -v
```

### Run Any Individual Challenge
```bash
python solutions/challenge_01_qrng.py
python solutions/challenge_02_coin_flipper.py
python solutions/challenge_03_quantum_safe.py
python solutions/challenge_04_entanglement_detector.py
python solutions/challenge_05_quantum_teleporter.py
python solutions/challenge_06_quantum_treasure_hunt.py
python solutions/challenge_07_quantum_rps.py
python solutions/challenge_08_quantum_classifier.py
```

---

## Conceptual Explanations & Deeper Questions

### 1. Quantum Random Number Generator
- **Pseudo-randomness vs Quantum randomness**: Classical computers use deterministic PRNG algorithms (e.g. Mersenne Twister) where the sequence is predictable given the seed. Quantum measurement is fundamentally non-deterministic according to the Born rule: $|\langle x|\psi\rangle|^2$.
- **Statistical convergence**: Running across 10, 100, 1,000, and 10,000 shots demonstrates the Law of Large Numbers—the maximum deviation from the uniform distribution $1/2^n$ drops from $\sim 0.10$ to $< 0.005$.
- **Modulo Bias**: For non-power-of-two ranges (e.g., rolling a 6-sided die), mapping $2^3 = 8$ states via `val % 6` produces severe bias ($P(0) = P(1) = 2/8 = 25\%$, while $P(2..5) = 1/8 = 12.5\%$). We implement **quantum rejection sampling** to ensure exact mathematical uniformity without bias.

### 2. Quantum Coin Flipper
- **Fair Coin**: An H-gate puts $|0\rangle$ into $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$.
- **Biased Coin**: Using $RY(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$, any probability $p \in [0, 1]$ of getting Tails can be engineered by setting $\theta = 2\arcsin(\sqrt{p})$.

### 3. The Quantum Safe
- **Combinations via Distributions**: A quantum safe does not look for a classical password; it measures the quantum state and unlocks only if the outcome distribution matches the target within tolerance $\epsilon$.
- **Non-uniqueness of circuits**: The target distribution `{'00': 0.5, '11': 0.5}` can be unlocked by multiple distinct circuits:
 - Circuit A: $H(0) \to CX(0, 1)$
 - Circuit B: $H(1) \to CX(1, 0)$
 - Circuit C: $X(0) \to H(0) \to CX(0, 1)$ ($|\Phi^-\rangle$ state, identical measurement probabilities).

### 4. Entanglement Detector
- **Correlation Metric**: $C = P(00) + P(11) - (P(01) + P(10))$.
 - Maximally correlated Bell states ($|\Phi^\pm\rangle$): $C = +1.00$.
 - Maximally anti-correlated Bell states ($|\Psi^\pm\rangle$): $C = -1.00$.
 - Separable superposition ($|+0\rangle$ without CNOT): $C = 0.00$.
 - Deterministic state ($|00\rangle$ without H-gate): $C = 1.00$, but entropy is 0 (classical correlation, not entanglement).
- **Concurrence**: Wootters Concurrence $\mathcal{C} = 2|c_{00}c_{11} - c_{01}c_{10}| = 1.00$ for all 4 Bell states and $0.00$ for product states.

### 5. Quantum Teleporter
- **Protocol**: Alice has an unknown state $|\psi\rangle$ on $q_0$ and shares an entangled pair on $q_1, q_2$ with Bob. Alice performs a Bell-basis measurement ($CX_{01} \to H_0$), transmitting 2 classical bits to Bob. Bob applies Pauli corrections ($X^{M_1}Z^{M_0}$) to $q_2$.
- **No Faster-Than-Light (FTL)**: Bob's reduced density matrix prior to receiving Alice's classical bits is $\frac{1}{2}I$ (maximally mixed), containing zero information. Classical communication speed limits the protocol, respecting special relativity.
- **No-Cloning Theorem**: Alice's measurement collapses and destroys the original state on $q_0$, so the state is transferred, not duplicated.

### 6. Quantum Treasure Hunt (Grover's Search)
- **Speedup**: To search an unstructured database of $N$ items, classical computers require $(N+1)/2$ queries on average. Grover's algorithm finds the target in $\lfloor \frac{\pi}{4}\sqrt{N} \rfloor$ queries, achieving a quadratic speedup.
- **For $N = 4$ ($2$ qubits)**: Exactly **1 query** yields $100\%$ success probability.
- **For $N = 8$ ($3$ qubits)**: Exactly **2 queries** yield $> 94\%$ success probability.

### 7. Quantum Rock, Paper, Scissors
- **The Bias Problem**: Representing 3 moves with 2 qubits leaves state $|11\rangle$. Naively mapping $|11\rangle \to \text{Rock}$ skews the game (Rock 50%, Paper 25%, Scissors 25%).
- **Unbiased Solutions**: We implement both rejection sampling and exact unitary qutrit state preparation ($RY(2\arccos(1/\sqrt{3})) \to CH \to CX$) which places zero amplitude on $|11\rangle$.

### 8. Quantum AI Classifier
- **Encoding**: Maps continuous classical features $x_i \in [-1, 1]$ into rotation angles $\theta_i = \pi x_i$ via $RY(\theta_i)$.
- **Variational Circuit**: Trainable parameterized $RY(\theta) / RZ(\theta)$ single-qubit rotations with CNOT ring entanglement.
- **Real-World Stretch Goal**: Integrates seamlessly with our **SoilQ** agronomy project, classifying Kenyan maize yield from field measurements.

---

## AI Policy Disclosure

In accordance with the [AI Usage Policy](../AI_POLICY.md):
- **Tools Used**: Google Antigravity / Gemini CLI Assistant.
- **How it was used**: Assisted in drafting mathematical formulations, verifying Qiskit 2.x API compatibility, structuring automated test assertions, and generating clean documentation.
- **Verification**: All quantum circuits, statevector calculations, mathematical derivations, and test executions were run, verified, and debugged locally on Python 3.14 + Qiskit 2.6.1.
