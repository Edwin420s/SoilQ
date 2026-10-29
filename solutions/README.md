# SoilQ Quantum Algorithm Primitives & Reference Implementations

This directory provides standalone, executable, and fully automated reference implementations of foundational and applied quantum algorithms in Qiskit 2.6.1. These modules serve as the mathematical and algorithmic building blocks for the SoilQ quantum computing engine.

---

## Quantum Algorithm Suite Overview

| # | Module | Key Quantum Concepts | Advanced Feature / Mathematical Proof | Test Status |
|---|---|---|---|:---:|
| **01** | [QRNG](challenge_01_qrng.py) | Superposition ($H^{\otimes n}$), Born rule measurement collapse | Unbiased arbitrary-range random integer (`quantum_random(min, max)`) with rejection sampling to eliminate modulo bias | Passed |
| **02** | [Quantum Coin Flipper](challenge_02_coin_flipper.py) | 1-qubit $|+\rangle$ state, probabilistic outcome | Biased coin via $RY(\theta)$ rotation with analytical $\theta = 2\arcsin\sqrt{p}$ derivation + Interactive simulation engine | Passed |
| **03** | [The Quantum Safe](challenge_03_quantum_safe.py) | Entanglement, Bell state distributions | Multiple non-identical unlocking circuits + Total Variation Distance ($TVD$) validation + 3-qubit GHZ state safe | Passed |
| **04** | [Entanglement Detector](challenge_04_entanglement_detector.py) | Non-separable states, Bell pairs, correlation coefficient $C$ | All 4 Bell states ($|\Phi^\pm\rangle, |\Psi^\pm\rangle$) comparison, exact Wootters Concurrence metric, noise impact | Passed |
| **05** | [Quantum Teleporter](challenge_05_quantum_teleporter.py) | EPR pairs, Bell-basis measurement, classical feedforward corrections | Teleporting arbitrary states ($|0\rangle, |1\rangle, |+\rangle, |-\rangle, |i\rangle, |\psi(\theta, \phi)\rangle$) with 100% fidelity + transit noise modeling | Passed |
| **06** | [Quantum Treasure Hunt](challenge_06_quantum_treasure_hunt.py) | Grover's search algorithm, Phase Oracle $U_\omega$, Diffuser $U_s$ | Dynamic user-specified treasure location on 2 & 3 qubits (100% and 94.3% certainty) + Classical vs Grover query complexity analysis | Passed |
| **07** | [Quantum Rock, Paper, Scissors](challenge_07_quantum_rps.py) | Mapping 3 choices to 2 qubits, handling the unused $|11\rangle$ state | The Bias Problem analysis + Unbiased rejection & unitary qutrit state preparation + Multi-round match engine | Passed |
| **08** | [Quantum AI Classifier](challenge_08_quantum_classifier.py) | Quantum machine learning, $RY(\pi \cdot x_i)$ feature encoding, VQC | Benchmark against classical models (LR, SVM) + Real-world Kenyan maize agronomy dataset ([SoilQ Platform](../)) | Passed |

---

## Running the Algorithm Suite & Test Suites

### Execute Master Runner (All 8 Modules)
```bash
python solutions/run_all_challenges.py
```

### Run Automated Pytest Suite
```bash
pytest solutions/test_challenges.py -v
```

### Run Any Individual Module
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

## Conceptual Formulations & Mathematical Analysis

### 1. Quantum Random Number Generator
- **Pseudo-randomness vs Quantum randomness**: Classical computers use deterministic PRNG algorithms (e.g. Mersenne Twister) where the sequence is predictable given the seed state. Quantum measurement is non-deterministic according to the Born rule: $|\langle x|\psi\rangle|^2$.
- **Statistical convergence**: Running across 10, 100, 1,000, and 10,000 shots demonstrates the Law of Large Numbers -- the maximum deviation from the uniform distribution $1/2^n$ drops from $\sim 0.10$ to $< 0.005$.
- **Modulo Bias Elimination**: For non-power-of-two ranges (e.g., rolling a 6-sided die), mapping $2^3 = 8$ states via `val % 6` produces severe bias ($P(0) = P(1) = 2/8 = 25\%$, while $P(2..5) = 1/8 = 12.5\%$). We implement **quantum rejection sampling** to ensure exact mathematical uniformity without bias.

### 2. Quantum Coin Flipper
- **Fair Coin**: An H-gate puts $|0\rangle$ into $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$.
- **Biased Coin**: Using $RY(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$, any probability $p \in [0, 1]$ of getting Tails can be engineered by setting $\theta = 2\arcsin(\sqrt{p})$.

### 3. The Quantum Safe
- **Combinations via Distributions**: A quantum safe measures the quantum state and unlocks only if the outcome distribution matches the target within tolerance $\epsilon$.
- **Circuit Non-Uniqueness**: Distinct circuits can generate identical measurement distributions. For example, $H(0) \to CX(0, 1)$ and $H(1) \to CX(1, 0)$ both generate the distribution $\{00: 50\%, 11: 50\%\}$.

### 4. Entanglement Detector
- **Non-Separability**: Bipartite state $|\psi\rangle_{AB}$ cannot be decomposed as $|\psi\rangle_A \otimes |\psi\rangle_B$.
- **Correlation Metric**: $C = (P_{00} + P_{11}) - (P_{01} + P_{10})$. Evaluates to $+1$ for $|\Phi^+\rangle$ and $-1$ for $|\Psi^+\rangle$.
- **Concurrence**: Exact Wootters concurrence for pure states: $\mathcal{C}(\psi) = 2|c_{00}c_{11} - c_{01}c_{10}| = 1.0000$ for maximally entangled states.

### 5. Quantum Teleporter
- **Channel Entanglement**: Entanglement creates non-local quantum correlations allowing Bob to reconstruct an unknown single-qubit state $|\psi\rangle$ after receiving 2 classical bits from Alice.
- **No-Cloning**: The protocol does not violate the no-cloning theorem because Alice's measurement destroys the original state $|\psi\rangle$.
- **Fidelity**: Exact state fidelity $F(\rho_{\text{target}}, \rho_{\text{Bob}}) = 1.000000$ across standard and arbitrary benchmark states.

### 6. Quantum Treasure Hunt (Grover Search)
- **Quadratic Speedup**: Classical unstructured search requires on average $(N+1)/2$ queries; Grover search locates the marked item in $R = \lfloor \frac{\pi}{4}\sqrt{N} \rfloor$ queries.
- **Oracle and Diffuser**: Phase oracle inverts amplitude of target item; diffuser inverts amplitudes around the mean.

### 7. Quantum Rock, Paper, Scissors
- **The Bias Problem**: Mapping $2^2 = 4$ states to 3 game choices ($00 \to \text{Rock}, 01 \to \text{Paper}, 10 \to \text{Scissors}$) leaves $|11\rangle$ unused. Naively mapping $|11\rangle \to \text{Rock}$ introduces severe bias ($50\%$ Rock vs $25\%$ Paper/Scissors).
- **Solutions**:
  1. Rejection sampling discarding $|11\rangle$.
  2. Unitary state preparation directly constructing $\frac{|00\rangle + |01\rangle + |10\rangle}{\sqrt{3}}$.

### 8. Quantum AI Classifier
- **Feature Encoding**: continuous features are normalized to $[-1, 1]$ and mapped to qubit rotation angles via $RY(\pi \cdot x_i)$.
- **Variational Optimization**: Parameterized ansatz layers with CNOT entanglement trained using COBYLA gradient-free optimization.
- **Agricultural Application**: Directly connected to the SoilQ platform for smallholder maize yield classification over Kenyan field trial data.
