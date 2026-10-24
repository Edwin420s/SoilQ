import numpy as np
from soilq.quantum import QiskitVariationalClassifier

def test_quantum_smoke():
    X = np.array([[-.8], [-.5], [-.2], [.2], [.5], [.8]])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = QiskitVariationalClassifier(n_features=1, reps=1, maxiter=10, seed=2).fit(X, y)
    assert m.predict(X).shape == y.shape
