# SoilQ — Quantum-Assisted Soil-Water Classification Research

A reproducible research starter codebase to audit soil/agronomy datasets, compare classical classifiers, and experiment with a small Qiskit variational classifier. Includes CLI, FastAPI API, Streamlit dashboard, tests, Docker and research documentation.

**Important:** the Kenyan maize dataset is not bundled; download it from https://data.mendeley.com/datasets/8bx88zpvdp/2 and put the archive/files in `data/raw/`. This code will not invent labels. You must explicitly select a binary target and features after auditing the actual data. No claim of quantum advantage or validated irrigation advice is made.

## Structure

- `src/soilq/data.py`: file inventory, table loading and schema audit
- `src/soilq/pipeline.py`: classical benchmarks, Qiskit model, metrics and reports
- `src/soilq/quantum.py`: variational circuit using Qiskit Statevector
- `src/soilq/api.py`: FastAPI research endpoints
- `app/dashboard.py`: Streamlit dashboard
- `scripts/`: CLI commands
- `docs/`: dataset, target and methodology guidance
- `tests/`: smoke tests

## Install (Linux)

Python 3.11 or 3.12 recommended.

```bash
cd SoilQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
```

## Audit the real dataset first

Download the Kenyan maize field experiment archive from the official Mendeley page, then place it in `data/raw/`.

```bash
python scripts/audit_dataset.py --input data/raw --output data/reports/kenyan_maize_audit.json
```

The audit reports file names, dimensions, columns, data types, missingness, unique counts, sample values, likely context columns and moisture-related columns. It does not decide the scientific target.

Supported tables: CSV, TSV, Excel, JSON/JSONL, Parquet. ZIP files are inventoried; CSV/TSV members are sampled without permanent extraction.

## Train after choosing a defensible binary target

The names below are placeholders, not presumed columns in the Kenyan dataset. Replace them with names actually found in the audit.

```bash
python scripts/train.py \
  --input data/raw/your_table.csv \
  --target YOUR_BINARY_TARGET \
  --features FEATURE_A FEATURE_B FEATURE_C \
  --positive-label YOUR_POSITIVE_CLASS \
  --output-dir models/experiment_001 \
  --report data/reports/experiment_001.json
```

Models: Logistic Regression, RBF SVM, Random Forest, and a small Qiskit variational classifier. The report contains accuracy, balanced accuracy, precision, recall, F1, ROC-AUC where available, confusion matrices, class counts, split settings, fit times, and dataset SHA-256.

The quantum classifier uses up to six supplied features and Qiskit's local Statevector simulator. It is a classical simulation of a quantum circuit, not execution on quantum hardware. It may perform worse than classical models.

## Run dashboard and API

```bash
streamlit run app/dashboard.py
uvicorn soilq.api:app --reload
```

API docs: http://127.0.0.1:8000/docs

The API is for trusted local use. Do not expose it publicly without authentication, path restrictions, upload limits, and deployment hardening.

## Tests

```bash
pytest
```

Tests validate software plumbing on toy data only; they do not establish scientific validity.

## Research cautions

- Establish what one row represents and whether repeated observations are grouped by plot/site/season.
- The default stratified random split is a baseline only. For repeated plots or time-series, use group-aware or chronological evaluation to avoid leakage.
- Do not call yield, treatment, or post-outcome variables “irrigation urgency” without evidence.
- A proxy label requires an agronomic rationale appropriate to crop, soil, depth, and context.
- If the Kenyan dataset is sparse or lacks a valid target, document that and consider EOCIS Africa soil-moisture data for time-series exploration; it is model-derived, coarse-grid data and does not directly label farmer decisions.
- Preserve dataset attribution/license and disclose AI assistance as required by the hackathon.
