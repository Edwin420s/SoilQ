import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class QiskitVariationalClassifier:
    # Small research baseline. Qiskit Statevector is classical circuit simulation.
    def __init__(self, n_features=None, reps=2, maxiter=60, seed=42):
        self.n_features = n_features; self.reps = int(reps)
        self.maxiter = int(maxiter); self.seed = int(seed)
        self.n_qubits = None; self.theta_ = None; self.classes_ = None
        self.optimization_result_ = None

    def _build(self, x, theta):
        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits): qc.ry(float(np.pi * x[q]), q)
        idx = 0
        for _ in range(self.reps):
            for q in range(self.n_qubits):
                qc.ry(float(theta[idx]), q); idx += 1
                qc.rz(float(theta[idx]), q); idx += 1
            for q in range(self.n_qubits - 1): qc.cx(q, q + 1)
            if self.n_qubits > 2: qc.cx(self.n_qubits - 1, 0)
        return qc

    def _expectation(self, x, theta):
        sv = Statevector.from_instruction(self._build(x, theta))
        probs = np.asarray(sv.probabilities(), dtype=float)
        return float(probs[0::2].sum() - probs[1::2].sum())

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y): raise ValueError("X must be 2D and match y length.")
        if len(np.unique(y)) != 2: raise ValueError("Exactly two classes are required.")
        self.classes_ = np.unique(y)
        self.n_qubits = min(int(self.n_features or X.shape[1]), X.shape[1], 6)
        if self.n_qubits < 1: raise ValueError("At least one feature is required.")
        X = X[:, :self.n_qubits]
        y_signed = np.where(y == self.classes_[1], 1.0, -1.0)
        rng = np.random.default_rng(self.seed)
        initial = rng.uniform(-0.25, 0.25, self.reps * self.n_qubits * 2)
        if len(X) > 160:
            ix = rng.choice(len(X), 160, replace=False); X_fit, y_fit = X[ix], y_signed[ix]
        else: X_fit, y_fit = X, y_signed
        def loss(theta):
            scores = np.array([self._expectation(row, theta) for row in X_fit])
            return float(np.mean(np.logaddexp(0.0, -y_fit * 2.0 * scores)) + 1e-4 * np.sum(theta**2))
        self.optimization_result_ = minimize(loss, initial, method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.4, "tol": 1e-3})
        self.theta_ = np.asarray(self.optimization_result_.x, dtype=float)
        return self

    def decision_function(self, X):
        if self.theta_ is None: raise RuntimeError("Model has not been fitted.")
        X = np.asarray(X, dtype=float)[:, :self.n_qubits]
        return np.array([self._expectation(row, self.theta_) for row in X])

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-4.0 * scores))
        return np.column_stack([1.0 - p1, p1])

    def get_metadata(self):
        r = self.optimization_result_
        return {"model": "QiskitVariationalClassifier", "simulator": "Qiskit Statevector",
                "n_qubits": self.n_qubits, "reps": self.reps, "maxiter": self.maxiter,
                "optimizer_success": bool(r.success) if r is not None else None,
                "optimizer_message": str(r.message) if r is not None else None}
