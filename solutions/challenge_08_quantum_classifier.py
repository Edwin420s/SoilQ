"""
Challenge 08: Quantum AI Classifier
Qiskit Fall Fest Nairobi 2026

Concepts:
- Quantum Machine Learning (QML) classification pipeline
- Quantum feature encoding: RY(pi * x_i) angle mapping
- Parameterized variational circuit with CNOT entanglement layers
- Pauli-Z expectation value measurement <Z_0>
- Deeper Questions:
  - Expressibility vs depth (reps=1, 2, 3)
  - Optimization scaling with dataset size
  - Barren plateaus and simulation overhead
- Stretch: Real-world agronomy dataset integration (SoilQ Kenyan Maize)
  and benchmark against classical models (Logistic Regression, RBF SVM)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def generate_toy_dataset(n_samples: int = 40, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2-feature binary classification dataset.
    Features scaled in [-1, 1].
    Class 0: lower-left blob
    Class 1: upper-right blob
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    X0 = rng.normal(loc=-0.5, scale=0.25, size=(n_per_class, 2))
    X1 = rng.normal(loc=0.5, scale=0.25, size=(n_per_class, 2))

    X = np.vstack([X0, X1])
    X = np.clip(X, -1.0, 1.0)
    y = np.array([0] * n_per_class + [1] * n_per_class)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class SimpleVariationalQuantumClassifier:
    """
    Variational Quantum Classifier for 2D data:
    1. Angle encoding: RY(pi * x_0) on Q0, RY(pi * x_1) on Q1
    2. Variational layers (reps): RY(theta), RZ(theta), CX(0, 1)
    3. Expectation value <Z_0> mapped to prediction
    4. Gradient-free COBYLA optimization
    """
    def __init__(self, n_features: int = 2, reps: int = 2, maxiter: int = 40, seed: int = 42):
        self.n_features = n_features
        self.reps = reps
        self.maxiter = maxiter
        self.seed = seed
        self.theta_ = None

    def _build_circuit(self, x: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_features)
        for q in range(self.n_features):
            qc.ry(float(np.pi * x[q]), q)

        idx = 0
        for _ in range(self.reps):
            for q in range(self.n_features):
                qc.ry(float(theta[idx]), q)
                idx += 1
                qc.rz(float(theta[idx]), q)
                idx += 1
            for q in range(self.n_features - 1):
                qc.cx(q, q + 1)
            if self.n_features > 2:
                qc.cx(self.n_features - 1, 0)
        return qc

    def _expectation(self, x: np.ndarray, theta: np.ndarray) -> float:
        qc = self._build_circuit(x, theta)
        sv = Statevector.from_instruction(qc)
        probs = np.asarray(sv.probabilities(), dtype=float)
        return float(probs[0::2].sum() - probs[1::2].sum())

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        y_signed = np.where(y == 1, 1.0, -1.0)
        n_params = self.reps * self.n_features * 2

        rng = np.random.default_rng(self.seed)
        initial = rng.uniform(-0.25, 0.25, n_params)

        def loss(theta):
            preds = np.array([self._expectation(row, theta) for row in X])
            return float(np.mean(np.logaddexp(0.0, -y_signed * 2.0 * preds)) + 1e-4 * np.sum(theta ** 2))

        cobyla_maxiter = max(self.maxiter, len(initial) + 2)
        res = minimize(loss, initial, method="COBYLA", options={"maxiter": cobyla_maxiter, "rhobeg": 0.4})
        self.theta_ = res.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted.")
        scores = np.array([self._expectation(row, self.theta_) for row in X])
        return np.where(scores >= 0, 1, 0)


def evaluate_depth_scaling(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    """
    Deeper Question:
    Does adding more circuit layers (reps = 1, 2, 3) improve classification accuracy?
    """
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    depth_results = {}
    for reps in [1, 2, 3]:
        vqc = SimpleVariationalQuantumClassifier(reps=reps, maxiter=30, seed=42)
        vqc.fit(X_train, y_train)
        preds = vqc.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        depth_results[reps] = round(acc, 3)
    return depth_results


def compare_classical_vs_quantum_toy() -> Dict[str, Dict]:
    """Compare QML classifier with classical Logistic Regression and SVM on toy data."""
    X, y = generate_toy_dataset(n_samples=50, seed=42)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr = LogisticRegression(random_state=42).fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    svm = SVC(kernel="rbf", random_state=42).fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    vqc = SimpleVariationalQuantumClassifier(reps=2, maxiter=40, seed=42).fit(X_train, y_train)
    vqc_pred = vqc.predict(X_test)

    return {
        "LogisticRegression": {"accuracy": round(float(accuracy_score(y_test, lr_pred)), 3), "f1": round(float(f1_score(y_test, lr_pred)), 3)},
        "RBF_SVM": {"accuracy": round(float(accuracy_score(y_test, svm_pred)), 3), "f1": round(float(f1_score(y_test, svm_pred)), 3)},
        "QuantumVQC": {"accuracy": round(float(accuracy_score(y_test, vqc_pred)), 3), "f1": round(float(f1_score(y_test, vqc_pred)), 3)},
    }


if __name__ == "__main__":
    print("=" * 60)
    print(" Challenge 08: Quantum AI Classifier")
    print("=" * 60)

    # 1. Toy Dataset Benchmark
    print("\n1. Benchmarking Classical Models vs Quantum VQC on Toy Dataset:")
    bench = compare_classical_vs_quantum_toy()
    for model, metrics in bench.items():
        print(f"Model: {model:<25} | Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")

    # 2. Deeper Question: Depth scaling
    print("\n2. Deeper Question: Does circuit depth (reps) improve performance?")
    X, y = generate_toy_dataset(40)
    depth_res = evaluate_depth_scaling(X, y)
    for reps, acc in depth_res.items():
        print(f"Layers (reps = {reps}) -> Test Accuracy: {acc * 100:.1f}%")

    # 3. Stretch Challenge: Real-World Dataset Connection
    print("\n3. Stretch Challenge: Real-World Agronomy Dataset (SoilQ Project):")
    print("SoilQ connects this Variational Classifier to the Kenyan Maize dataset (448 rows).")
    print("Classical benchmarks (Random Forest: ~79%) vs Qiskit VQC (~66%).")
    print("See full real-world experiment report at: data/reports/experiment_001.json")
