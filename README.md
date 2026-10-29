# SoilQ -- Quantum-Assisted Smallholder Agronomy & Decision Support Platform

[![Python 3.14](https://img.shields.io/badge/Python-3.14.8-blue.svg)](https://www.python.org/)
[![Qiskit 2.6](https://img.shields.io/badge/Qiskit-2.6.1-613394.svg)](https://qiskit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-66%20Passing-brightgreen.svg)]()

SoilQ is an enterprise-grade agricultural intelligence and decision-support platform pairing **quantum machine learning (Qiskit 2.x)** with **classical machine learning (Scikit-Learn)** to address crop productivity and soil fertility challenges in Sub-Saharan Africa. The system models smallholder maize yield outcomes using multi-season field trial data from Kenya (*Mutuku et al., Mendeley Data CC BY 4.0*).

---

## 1. Problem Statement & Agronomic Solution

Smallholder maize farming in Kenya suffers from chronic yield gaps driven by acidic Humic Nitisols, unpredictable bimodal rainfall, and delayed detection of crop stress. 

**SoilQ solves this by:**
- Transforming multi-season field trial data (448 plots across Chuka and Kandara) into an analysis-ready predictive foundation.
- Deploying hybrid quantum-classical machine learning architectures:
  - **Variational Quantum Classifier (VQC)**: Parameterized ansatz with continuous angle encoding and ring CNOT entanglement.
  - **Quantum Kernel Support Vector Machine (QSVM)**: Exact quantum state fidelity evaluation in Hilbert space.
- Benchmarking quantum models against classical baselines (Random Forest, Logistic Regression, RBF SVM).
- Providing practical decision-support interfaces: an interactive **Streamlit Advisory Dashboard** and a **FastAPI REST Microservice** for field agronomists.

---

## 2. System Architecture

```
+------------------------------------------------------------------------+
|                      Raw Agronomy Data Layer                           |
|  Maize Yield . Plant Height . SPAD Chlorophyll . Soil Moisture . Rain  |
+-----------------------------------+------------------------------------+
                                    | Plot-level aggregation & merge
                                    v
+------------------------------------------------------------------------+
|                   Analysis-Ready Processed Dataset                     |
|  448 Plots . Balanced 50/50 Median Target (high_yield: >0.735 Mg/ha)   |
+-------------------+--------------------------------+-------------------+
                    |                                |
                    v                                v
+--------------------------------------+ +-------------------------------+
|       Classical ML Pipelines         | |      Quantum ML Pipelines     |
| . Logistic Regression                | | . Variational VQC (Statevec)  |
| . Support Vector Classifier (RBF)    | | . Quantum Kernel (QSVM)       |
| . Random Forest (100 estimators)     | | . Pauli-Z Expectation <Z_0>   |
+-------------------+------------------+ +---------------+---------------+
                    |                                    |
                    +----------------+-------------------+
                                     |
                                     v
+------------------------------------------------------------------------+
|                      Serving & Interface Layer                         |
| . FastAPI REST Endpoints (/health, /predict, /train, /circuit, /audit) |
| . Streamlit Research Dashboard (Field Predictor, EDA, Circuit Studio)  |
| . Interactive Standalone Jupyter Notebooks (.ipynb)                    |
+------------------------------------------------------------------------+
```

---

## 3. Performance Benchmarks (5-Fold Cross-Validation)

Evaluated using 5-Fold Stratified Cross-Validation on the full 448-plot dataset:

| Model Architecture | Accuracy (Mean +- Std) | F1 Score (Mean +- Std) | ROC-AUC | Brier Score | Fit Time (s) |
|---|---|---|---|---|---|
| **Random Forest (Classical)** | **0.8705 +- 0.0338** | **0.8634 +- 0.0396** | **0.9387** | 0.0984 | 0.18 s |
| **Logistic Regression (Classical)** | 0.8460 +- 0.0276 | 0.8373 +- 0.0285 | 0.9082 | 0.1192 | 0.01 s |
| **RBF Support Vector Machine** | 0.8415 +- 0.0313 | 0.8265 +- 0.0362 | 0.9067 | 0.1130 | 0.02 s |
| **Quantum Kernel Classifier (QSVM)** | **0.7322 +- 0.0301** | **0.7285 +- 0.0185** | **0.7950** | 0.1812 | 0.48 s |
| **Qiskit Variational Classifier (VQC)** | 0.6607 +- 0.0394 | 0.6389 +- 0.0412 | 0.7114 | 0.2185 | 5.12 s |

---

## 4. Quantum Algorithm Reference Suite

Located in `solutions/`:

| Module | Algorithm | Core Mathematical Formulation | Test Status |
|---|---|---|:---:|
| **01** | [QRNG](solutions/challenge_01_qrng.py) | $H^{\otimes n}|0\rangle^{\otimes n} \to \frac{1}{\sqrt{2^n}}\sum |k\rangle$ with rejection sampling | Passed (3/3) |
| **02** | [Coin Flipper](solutions/challenge_02_coin_flipper.py) | Fair $H$ & Biased $RY(2\arcsin\sqrt{p})$ with game engine | Passed (3/3) |
| **03** | [Quantum Safe](solutions/challenge_03_quantum_safe.py) | Distribution matching + Total Variation Distance ($TVD$) validation | Passed (2/2) |
| **04** | [Entanglement Detector](solutions/challenge_04_entanglement_detector.py) | All 4 Bell states + exact Wootters Concurrence metric | Passed (2/2) |
| **05** | [Quantum Teleporter](solutions/challenge_05_quantum_teleporter.py) | 3-qubit teleportation + state fidelity verification ($F=1.000000$) | Passed (2/2) |
| **06** | [Treasure Hunt](solutions/challenge_06_quantum_treasure_hunt.py) | Grover search with automated phase oracle and diffuser | Passed (2/2) |
| **07** | [Quantum RPS](solutions/challenge_07_quantum_rps.py) | 2-qubit mapping + bias elimination via uniform qutrit state preparation | Passed (3/3) |
| **08** | [Quantum AI Classifier](solutions/challenge_08_quantum_classifier.py) | VQC angle encoding + expectation measurement on agricultural data | Passed (1/1) |

---

## 5. Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Edwin420s/SoilQ.git
cd SoilQ
pip install -e .
```

### 2. Run All Automated Tests (66 Tests)
```bash
pytest tests/ solutions/test_challenges.py -v
```

### 3. Run Quantum Algorithm Suite
```bash
python solutions/run_all_challenges.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app/dashboard.py
```
Access the dashboard at `http://localhost:8501`.

### 5. Launch the REST API Microservice
```bash
uvicorn soilq.api:app --app-dir src --host 127.0.0.1 --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 6. Project Layout

```
SoilQ/
├── README.md                                       <- Project overview and quickstart
├── SUBMISSION.md                                   <- Master technical architecture report
├── SoilQ_Product_Technical_Whitepaper.pdf          <- 5-page publication whitepaper
├── app/
│   └── dashboard.py                                <- Enterprise 6-tab Streamlit dashboard
├── src/soilq/
│   ├── api.py                                      <- Production FastAPI microservice (7 endpoints)
│   ├── config.py                                   <- Root paths and configuration
│   ├── data.py                                     <- Data ingestion and SHA-256 auditing
│   ├── pipeline.py                                 <- 5-model benchmark & cross-validation engine
│   ├── prepare.py                                  <- 5-table consolidation & target definition
│   └── quantum.py                                  <- VQC & Quantum Kernel QSVM implementations
├── data/
│   ├── raw/                                        <- Raw Mendeley experimental CSV tables
│   ├── processed/                                  <- Merged analysis table (448 rows)
│   └── reports/                                    <- cross_validation.json & experiment_001.json
├── models/                                         <- Serialized .joblib model artifacts
├── notebooks/                                      <- 8 standalone interactive Jupyter notebooks
├── solutions/                                      <- Standalone quantum algorithm modules (01 to 08)
├── tests/                                          <- Automated pytest test suite
├── pyproject.toml                                  <- Package specifications and dependencies
├── Makefile                                        <- Build, test, and run automation
├── Dockerfile & docker-compose.yml                 <- Containerized deployment specifications
└── LICENSE                                         <- Apache 2.0 open-source license
```

---

## 7. License & Citations

- **Code License**: Apache 2.0
- **Agronomic Dataset**: Mutuku et al., Mendeley Data, Version 2 (2020), CC BY 4.0. `doi:10.17632/8bx88zpvdp.2`.
