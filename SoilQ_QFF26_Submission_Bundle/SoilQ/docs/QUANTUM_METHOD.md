# Quantum classifier method

The Qiskit model uses `qiskit.quantum_info.Statevector`:

1. Features are scaled to [-1, 1].
2. Encode each feature using `RY(pi * x_i)` on a qubit.
3. Each variational layer applies trainable RY/RZ rotations.
4. CNOTs entangle adjacent qubits and close a ring when there are more than two qubits.
5. The output is the Z expectation on qubit 0.
6. COBYLA minimizes a regularized logistic-style loss using signed labels.

At most six features are used, in the supplied order. Feature order should be chosen and justified by the researcher. The optimizer is a small hackathon baseline, not a definitive quantum machine-learning study.

Statevector is classical simulation and scales exponentially. The probability transform is not calibrated. This is one circuit design, not proof of quantum advantage. Compare with classical models on the same held-out rows.
