"""
SoilQ Quantum Machine Learning Module

Implements:
1. QiskitVariationalClassifier:
   - Parameterized quantum circuit with configurable feature maps (Angle RY or ZZ entangling)
   - Configurable entanglement topologies (ring, linear, full)
   - Regularized cross-entropy loss with loss history tracking
   - Statevector classical simulation of quantum states
2. QuantumKernelClassifier:
   - Quantum Support Vector Machine (QSVM) using quantum state fidelity kernel
   - K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
   - Dual-space quadratic optimization via scikit-learn SVC
3. Circuit inspection and export tools (ASCII diagrams, depth, gate counts)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def build_feature_map(
    x: np.ndarray,
    n_qubits: int,
    feature_map_type: str = "angle",
) -> QuantumCircuit:
    """
    Encode classical features into a quantum circuit state.
    'angle': RY(pi * x_i) on each qubit
    'zz': RY(pi * x_i) followed by ZZ entangling gates: CX -> RZ(pi * (1-x_i)(1-x_j)) -> CX
    """
    qc = QuantumCircuit(n_qubits, name="FeatureMap")
    for q in range(n_qubits):
        qc.ry(float(np.pi * x[q]), q)

    if feature_map_type.lower() in ("zz", "zz_feature_map") and n_qubits > 1:
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            phi = float(np.pi * (1.0 - x[q]) * (1.0 - x[q + 1]))
            qc.rz(phi, q + 1)
            qc.cx(q, q + 1)
    return qc


def build_variational_ansatz(
    n_qubits: int,
    theta: np.ndarray,
    reps: int = 2,
    entanglement: str = "ring",
) -> QuantumCircuit:
    """
    Construct parameterized variational layers.
    Per layer: RY(theta_i) and RZ(theta_j) on each qubit, followed by entangling CX gates.
    """
    qc = QuantumCircuit(n_qubits, name="Ansatz")
    idx = 0
    for _ in range(reps):
        for q in range(n_qubits):
            qc.ry(float(theta[idx]), q); idx += 1
            qc.rz(float(theta[idx]), q); idx += 1

        if entanglement == "linear":
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        elif entanglement == "full":
            for q1 in range(n_qubits):
                for q2 in range(q1 + 1, n_qubits):
                    qc.cx(q1, q2)
        else:  # "ring"
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)
    return qc


class QiskitVariationalClassifier(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier (VQC) using Qiskit Statevector simulation.

    Parameters
    ----------
    n_features : int, optional
        Number of features/qubits to use (capped at 6).
    reps : int, default=2
        Number of variational ansatz layers.
    maxiter : int, default=60
        Maximum optimizer iterations.
    seed : int, default=42
        Random seed for parameter initialization and subsampling.
    feature_map : str, default='angle'
        'angle' (RY rotations) or 'zz' (entangled feature map).
    entanglement : str, default='ring'
        'ring', 'linear', or 'full' CX entanglement topology.
    optimizer : str, default='COBYLA'
        'COBYLA', 'SLSQP', or 'L-BFGS-B'.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        reps: int = 2,
        maxiter: int = 60,
        seed: int = 42,
        feature_map: str = "angle",
        entanglement: str = "ring",
        optimizer: str = "COBYLA",
    ):
        self.n_features = n_features
        self.reps = int(reps)
        self.maxiter = int(maxiter)
        self.seed = int(seed)
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.optimizer = optimizer

        self.n_qubits: Optional[int] = None
        self.theta_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.optimization_result_: Optional[Any] = None
        self.loss_history_: List[float] = []

    def _build_full_circuit(self, x: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        fmap_type = getattr(self, "feature_map", "angle")
        ent_type = getattr(self, "entanglement", "ring")
        fmap = build_feature_map(x, self.n_qubits, fmap_type)
        ansatz = build_variational_ansatz(self.n_qubits, theta, self.reps, ent_type)
        qc.compose(fmap, inplace=True)
        qc.compose(ansatz, inplace=True)
        return qc

    def _expectation(self, x: np.ndarray, theta: np.ndarray) -> float:
        qc = self._build_full_circuit(x, theta)
        sv = Statevector.from_instruction(qc)
        probs = np.asarray(sv.probabilities(), dtype=float)
        # Pauli-Z expectation on qubit 0
        return float(probs[0::2].sum() - probs[1::2].sum())

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and match y length.")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Exactly two classes are required.")

        self.n_qubits = min(int(self.n_features or X.shape[1]), X.shape[1], 6)
        if self.n_qubits < 1:
            raise ValueError("At least one feature is required.")

        X = X[:, :self.n_qubits]
        y_signed = np.where(y == self.classes_[1], 1.0, -1.0)

        rng = np.random.default_rng(self.seed)
        n_params = self.reps * self.n_qubits * 2
        initial = rng.uniform(-0.25, 0.25, n_params)

        # Subsample if dataset is large to maintain interactive responsiveness
        if len(X) > 160:
            ix = rng.choice(len(X), 160, replace=False)
            X_fit, y_fit = X[ix], y_signed[ix]
        else:
            X_fit, y_fit = X, y_signed

        self.loss_history_ = []

        def loss(theta):
            scores = np.array([self._expectation(row, theta) for row in X_fit])
            val = float(
                np.mean(np.logaddexp(0.0, -y_fit * 2.0 * scores))
                + 1e-4 * np.sum(theta ** 2)
            )
            self.loss_history_.append(round(val, 5))
            return val

        cobyla_maxiter = max(self.maxiter, len(initial) + 2)
        opt_method = self.optimizer.upper() if self.optimizer.upper() in ("COBYLA", "SLSQP", "L-BFGS-B") else "COBYLA"

        options = {"maxiter": cobyla_maxiter}
        if opt_method == "COBYLA":
            options.update({"rhobeg": 0.4, "tol": 1e-3})

        self.optimization_result_ = minimize(
            loss,
            initial,
            method=opt_method,
            options=options,
        )
        self.theta_ = np.asarray(self.optimization_result_.x, dtype=float)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model has not been fitted.")
        X = np.asarray(X, dtype=float)[:, :self.n_qubits]
        return np.array([self._expectation(row, self.theta_) for row in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-4.0 * scores))
        return np.column_stack([1.0 - p1, p1])

    def get_circuit_diagram(self) -> str:
        """Return an ASCII drawing of the trained circuit on a dummy zero input."""
        if self.n_qubits is None:
            return "Circuit not initialized."
        dummy_x = np.zeros(self.n_qubits)
        theta = self.theta_ if self.theta_ is not None else np.zeros(self.reps * self.n_qubits * 2)
        qc = self._build_full_circuit(dummy_x, theta)
        return qc.draw(output="text", fold=80).__str__()

    def get_metadata(self) -> Dict[str, Any]:
        r = getattr(self, "optimization_result_", None)
        loss_hist = getattr(self, "loss_history_", [])
        return {
            "model": "QiskitVariationalClassifier",
            "simulator": "Qiskit Statevector",
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "maxiter": self.maxiter,
            "feature_map": getattr(self, "feature_map", "angle"),
            "entanglement": getattr(self, "entanglement", "ring"),
            "optimizer": getattr(self, "optimizer", "COBYLA"),
            "optimizer_success": bool(r.success) if r is not None else None,
            "optimizer_message": str(r.message) if r is not None else None,
            "loss_iterations": len(loss_hist),
            "final_loss": loss_hist[-1] if loss_hist else None,
        }


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Classifier (QSVM) using Quantum State Fidelity Kernel.
    Computes K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2 in Hilbert space.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        feature_map: str = "zz",
        C: float = 1.0,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.feature_map = feature_map
        self.C = C
        self.seed = seed

        self.n_qubits: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self.svc_: Optional[SVC] = None

    def _statevector(self, x: np.ndarray) -> Statevector:
        qc = build_feature_map(x, self.n_qubits, self.feature_map)
        return Statevector.from_instruction(qc)

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        svs1 = [self._statevector(x) for x in X1]
        svs2 = [self._statevector(x) for x in X2] if X1 is not X2 else svs1
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2), dtype=float)

        for i in range(n1):
            start_j = i if X1 is X2 else 0
            for j in range(start_j, n2):
                fid = float(state_fidelity(svs1[i], svs2[j]))
                K[i, j] = fid
                if X1 is X2:
                    K[j, i] = fid
        return K

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Exactly two classes required.")

        self.n_qubits = min(int(self.n_features or X.shape[1]), X.shape[1], 6)
        self.X_train_ = X[:, :self.n_qubits]

        # Kernel matrix computation
        K_train = self._compute_kernel_matrix(self.X_train_, self.X_train_)
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.seed)
        self.svc_.fit(K_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.svc_ is None or self.X_train_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)[:, :self.n_qubits]
        K_test = self._compute_kernel_matrix(X, self.X_train_)
        return self.svc_.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.svc_ is None or self.X_train_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)[:, :self.n_qubits]
        K_test = self._compute_kernel_matrix(X, self.X_train_)
        return self.svc_.predict_proba(K_test)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model": "QuantumKernelClassifier",
            "simulator": "Qiskit Statevector",
            "kernel": "Quantum State Fidelity |<phi(x)|phi(x')>|^2",
            "n_qubits": self.n_qubits,
            "feature_map": self.feature_map,
            "C": self.C,
            "n_support_vectors": int(len(self.svc_.support_)) if self.svc_ is not None else 0,
        }
