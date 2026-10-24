# Scope and release checklist

Included: multi-format data audit, explicit binary classification, training-only imputation/scaling, Logistic Regression/RBF SVM/Random Forest, Qiskit Statevector classifier, metrics and JSON report, model persistence, FastAPI, Streamlit, Docker and tests.

Not included: source dataset, invented irrigation labels, farm-specific calibration, irrigation hardware control, claim of quantum advantage, public deployment security, automatic inference of valid group/time splits.

Before submission:
- [ ] Audit the real dataset.
- [ ] Justify target and feature selection.
- [ ] Review leakage, licensing and units.
- [ ] Run tests and a smoke experiment.
- [ ] Compare classical baselines and quantum model.
- [ ] Use group/time-aware validation where appropriate.
- [ ] Explain the circuit and disclose AI assistance according to hackathon rules.
