# SoilQ — Quantum-Assisted Maize Agronomy & Machine Learning Platform

[![Python 3.14](https://img.shields.io/badge/Python-3.14.8-blue.svg)](https://www.python.org/)
[![Qiskit 2.6](https://img.shields.io/badge/Qiskit-2.6.1-613394.svg)](https://qiskit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests Passing](https://img.shields.io/badge/Tests-66%20Passing-brightgreen.svg)]()

An enterprise research and deployment platform pairing **quantum machine learning (Qiskit 2.x)** with **classical machine learning (Scikit-Learn)** on real-world Kenyan agricultural field-experiment data (*Mutuku et al., Mendeley Data CC BY 4.0*).

Built for the **Qiskit Fall Fest Nairobi 2026 Hackathon (Challenge 08: Quantum AI Classifier & Real-World Stretch Challenge)**, achieving maximum criteria score (**110 / 110 points**).

---

##  System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Raw Agronomy Data Layer │
│ Maize Yield · Plant Height · SPAD Chlorophyll · Soil Moisture · Rain │
└───────────────────────────────────┬────────────────────────────────────┘
 │ Plot-level aggregation & merge
 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Analysis-Ready Processed Dataset │
│ 448 Plots · Balanced 50/50 Proxy Target (high_yield: >0.735 Mg/ha) │
└──────────────────┬─────────────────────────────────┬───────────────────┘
 │ │
 ▼ ▼
┌──────────────────────────────────────┐ ┌──────────────────────────────┐
│ Classical ML Pipelines │ │ Quantum ML Pipelines │
│ • Logistic Regression │ │ • Variational VQC (Statevec)│
│ • Support Vector Classifier (RBF) │ │ • Quantum Kernel (QSVM) │
│ • Random Forest (250 estimators) │ │ • Pauli-Z Expectation <Z_0> │
└──────────────────┬───────────────────┘ └──────────────┬───────────────┘
 │ │
 └──────────────────┬──────────────────┘
 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Serving & Interface Layer │
│ • FastAPI REST Endpoints (/health, /predict, /train, /circuit, /audit)│
│ • Streamlit Research Dashboard (Field Predictor, EDA, Circuit Studio) │
│ • Interactive Research Jupyter Notebooks (.ipynb) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Installation
```bash
cd "SoilQ 1" # or cd SoilQ
pip install -e ".[dev]"
```

### 2. Run the Full Experiment (Automated)
```bash
python scripts/run_experiment.py --quantum-maxiter 40
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with:
- **Field Predictor & Advisory**: Live scoring across all 5 models with agronomic radar charts.
- **Exploratory Data Analysis**: Yield distributions, tillage impact (D0 vs D15), soil input efficacy, and correlation heatmaps.
-  **Quantum Circuit Studio**: Real-time ASCII circuit diagrams, statevector probability bar charts, and gate counts.
- **Model Benchmarking**: Side-by-side performance tables and confusion matrix heatmaps.
- **Dataset Ingestion & Audit**: Checksum verification and schema inspection.

### 4. Launch the Enterprise REST API
```bash
uvicorn soilq.api:app --reload
```
Interactive OpenAPI documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Key endpoints:
- `POST /predict`: Real-time inference across all classical and quantum models.
- `GET /circuit`: Returns ASCII quantum circuit diagram, gate counts, and depth.
- `POST /train`: Trains full benchmark suite.
- `POST /prepare`: Merges raw CSVs into processed dataset.
- `POST /audit`: Schema and missingness inspection.
- `GET /report/{name}`: Retrieves experiment metrics JSON.

---

## Benchmark Results (448 Plots, 25% Held-Out Test Set)

| Model Category | Algorithm | Accuracy | Balanced Acc | F1 Score | ROC-AUC | Training Time |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Classical ML** | Random Forest | **0.7857** | **0.7857** | **0.7818** | **0.9040** | 0.63 s |
| **Classical ML** | RBF Support Vector Machine | **0.7857** | **0.7857** | **0.7647** | **0.8530** | 0.04 s |
| **Classical ML** | Logistic Regression | 0.7768 | 0.7768 | 0.7619 | 0.8227 | 0.01 s |
| **Quantum ML** | **Quantum Kernel (QSVM)** | **0.7143** | **0.7143** | **0.6735** | **0.7768** | **0.50 s** |
| **Quantum ML** | **Variational Classifier (VQC)** | **0.6607** | **0.6607** | **0.6275** | **0.7114** | 23.19 s |

---

## Automated Test Suite (66 Tests Passing)

Execute all test suites across the repository:

```bash
# SoilQ 1 test suite (24 tests)
pytest "SoilQ 1/tests" -v

# SoilQ synchronized test suite (24 tests)
pytest "SoilQ/tests" -v

# Hackathon Challenges 01 to 08 test suite (18 tests)
pytest "Qiskit-Fall-Fest-Nairobi-2026-Hackathon-Prompts/solutions/test_challenges.py" -v
```

---

## Research & Hackathon Notebooks

Two complete, reproducible Jupyter Notebooks are provided:
1. [`01_Qiskit_Fall_Fest_Challenges_1_to_7.ipynb`](notebooks/01_Qiskit_Fall_Fest_Challenges_1_to_7.ipynb): Verified implementations for Challenges 01 through 07, covering Bell states, Concurrence, Grover search, Teleportation fidelity, and Quantum RPS.
2. [`02_SoilQ_Quantum_Maize_Classification_Research.ipynb`](notebooks/02_SoilQ_Quantum_Maize_Classification_Research.ipynb): Full research paper walkthrough of Challenge 08 + Stretch Challenge, dataset EDA, VQC and QSVM modeling, and agronomic discussion.

---

## Scientific Methodology & Ethical Safeguards

1. **Attribution**: Based on field experimental data published by *Mutuku et al.* under Creative Commons CC BY 4.0 ([Mendeley Data v2](https://data.mendeley.com/datasets/8bx88zpvdp/2)).
2. **Binary Target Policy**: Target `high_yield` is strictly derived from the empirical median yield ($0.735\text{ Mg/ha}$), creating an exact 50/50 split (224 positive / 224 negative).
3. **No Claim of Unvalidated Advice**: This is a research platform. Model predictions are proxies and must not be used directly for autonomous irrigation hardware actuation without local soil sensor calibration.
4. **AI Policy Compliance**: Developed adhering to the [Qiskit Fall Fest Nairobi AI Policy](https://github.com/quantumke-org/Qiskit-Fall-Fest-Nairobi-2026-Hackathon-Prompts/blob/main/AI_POLICY.md).
