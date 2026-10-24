"""
tests/test_quantum.py — QiskitVariationalClassifier unit tests.
"""
import numpy as np
import pytest
from soilq.quantum import QiskitVariationalClassifier


def test_quantum_smoke_1d():
    """Single-feature classifier should fit and predict correctly shaped output."""
    X = np.array([[-.8], [-.5], [-.2], [.2], [.5], [.8]])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = QiskitVariationalClassifier(n_features=1, reps=1, maxiter=10, seed=2).fit(X, y)
    preds = m.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})


def test_quantum_predict_proba():
    """predict_proba should return shape (n, 2) with rows summing to ~1."""
    X = np.array([[-.8], [-.5], [-.2], [.2], [.5], [.8]])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = QiskitVariationalClassifier(n_features=1, reps=1, maxiter=5, seed=3).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6), atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_quantum_2_features():
    """Two-qubit circuit should work and use exactly 2 qubits."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (20, 2))
    y = (X[:, 0] > 0).astype(int)
    m = QiskitVariationalClassifier(n_features=2, reps=2, maxiter=10, seed=7).fit(X, y)
    assert m.n_qubits == 2
    assert m.predict(X).shape == (20,)


def test_quantum_caps_at_6_features():
    """Classifier should silently cap at 6 qubits regardless of input width."""
    rng = np.random.default_rng(9)
    X = rng.uniform(-1, 1, (20, 10))
    y = (X[:, 0] > 0).astype(int)
    m = QiskitVariationalClassifier(reps=1, maxiter=5, seed=9).fit(X, y)
    assert m.n_qubits == 6


def test_quantum_rejects_single_class():
    """Classifier should raise ValueError when only one class is present."""
    X = np.ones((10, 1))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="two classes"):
        QiskitVariationalClassifier(maxiter=5).fit(X, y)


def test_quantum_metadata():
    """get_metadata should return the expected keys after fitting."""
    X = np.array([[-.5], [.5], [-.3], [.3]])
    y = np.array([0, 1, 0, 1])
    m = QiskitVariationalClassifier(n_features=1, reps=1, maxiter=5).fit(X, y)
    meta = m.get_metadata()
    for key in ("model", "simulator", "n_qubits", "reps", "maxiter"):
        assert key in meta, f"Missing key: {key}"
    assert meta["simulator"] == "Qiskit Statevector"


def test_quantum_not_fitted_raises():
    """predict before fit should raise RuntimeError."""
    m = QiskitVariationalClassifier()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(np.array([[0.5]]))


def test_quantum_kernel_smoke():
    """Test QuantumKernelClassifier fit and predict."""
    from soilq.quantum import QuantumKernelClassifier
    X = np.array([[-.8, -.5], [-.2, -.1], [.2, .3], [.5, .8]])
    y = np.array([0, 0, 1, 1])
    qsvm = QuantumKernelClassifier(n_features=2, feature_map="zz", C=1.0, seed=42)
    qsvm.fit(X, y)
    preds = qsvm.predict(X)
    assert preds.shape == y.shape
    proba = qsvm.predict_proba(X)
    assert proba.shape == (4, 2)
    meta = qsvm.get_metadata()
    assert meta["model"] == "QuantumKernelClassifier"
    assert "kernel" in meta
