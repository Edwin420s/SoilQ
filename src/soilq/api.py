"""
SoilQ FastAPI Research Endpoints

API Endpoints:
- GET  /health          — Service health check
- POST /prepare         — Consolidate raw agronomy datasets into analysis-ready table
- POST /audit           — Audit dataset schemas, statistics, missingness
- POST /train           — Train classical + Qiskit VQC + Quantum Kernel benchmarks
- POST /predict         — Run real-time inference with classical and quantum models
- GET  /circuit         — Retrieve quantum circuit diagram and specifications
- GET  /models          — Enumerate saved .joblib model bundles
- GET  /report/{name}   — Retrieve stored experiment benchmark report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MODEL_DIR, PROJECT_ROOT
from .data import audit_dataset
from .pipeline import predict_from_model_dir, train_benchmark
from .prepare import prepare as _prepare

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

app = FastAPI(
    title="SoilQ Quantum Agronomy API",
    version="1.0.0",
    description=(
        "Enterprise research API for quantum-assisted maize yield classification "
        "and agronomic benchmarking over Kenyan field experiment data."
    ),
)


# ── Pydantic Request / Response Models ────────────────────────────────────────

class PrepareResponse(BaseModel):
    rows: int
    columns: List[str]
    class_counts: Dict[str, int]
    merged_csv: Optional[str] = None
    note: str


class AuditRequest(BaseModel):
    input_path: str = "data/raw"


class TrainRequest(BaseModel):
    input_path: str = "data/processed/kenyan_maize_merged.csv"
    target: str = "high_yield"
    features: List[str] = Field(
        default=["Height_6th", "Height_10th", "SPAD_6th", "SPAD_10th", "Stover_Mg/ha"],
        min_length=1,
    )
    output_dir: str = "models/api_run"
    report_path: Optional[str] = None
    positive_label: Optional[str] = "1"
    test_size: float = Field(default=0.25, gt=0.1, lt=0.5)
    seed: int = 42
    quantum_maxiter: int = Field(default=40, ge=10, le=200)
    include_quantum_kernel: bool = True


class PredictRequest(BaseModel):
    model_dir: str = "models/experiment_001"
    features: Dict[str, float] = Field(
        default={
            "Height_6th": 32.5,
            "Height_10th": 145.0,
            "SPAD_6th": 42.0,
            "SPAD_10th": 44.5,
            "Stover_Mg/ha": 5.2,
        }
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "service": "SoilQ",
        "version": "1.0.0",
        "note": "Production-grade research API for quantum agronomy machine learning.",
    }


@app.post("/prepare", response_model=PrepareResponse, tags=["Data Pipeline"])
def prepare_endpoint():
    """Merge five raw agronomy datasets into a single analysis-ready table."""
    try:
        merged, prov = _prepare(save=True)
        return PrepareResponse(
            rows=len(merged),
            columns=list(merged.columns),
            class_counts={str(k): int(v) for k, v in prov["class_counts"].items()},
            merged_csv=prov.get("merged_csv"),
            note=prov["note"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/audit", tags=["Data Pipeline"])
def audit(req: AuditRequest):
    """Inspect dataset schemas, missingness, and column statistics."""
    try:
        path = Path(req.input_path)
        if not path.is_absolute() and not path.exists():
            path = PROJECT_ROOT / path
        return audit_dataset(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/train", tags=["Machine Learning"])
def train(req: TrainRequest):
    """
    Train benchmark suite: Logistic Regression, RBF SVM, Random Forest,
    Qiskit VQC, and Quantum Kernel Classifier.
    """
    try:
        path = Path(req.input_path)
        if not path.is_absolute() and not path.exists():
            path = PROJECT_ROOT / path
        out_dir = Path(req.output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        rep_path = Path(req.report_path) if req.report_path else None
        if rep_path and not rep_path.is_absolute():
            rep_path = PROJECT_ROOT / rep_path

        return train_benchmark(
            input_path=path,
            target=req.target,
            features=req.features,
            output_dir=out_dir,
            report_path=rep_path,
            positive_label=req.positive_label,
            test_size=req.test_size,
            seed=req.seed,
            quantum_maxiter=req.quantum_maxiter,
            include_quantum_kernel=req.include_quantum_kernel,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict", tags=["Inference"])
def predict(req: PredictRequest):
    """
    Run real-time inference on input features across all classical and quantum models.
    """
    mdir = Path(req.model_dir)
    if not mdir.is_absolute():
        mdir = PROJECT_ROOT / mdir
    if not mdir.exists():
        raise HTTPException(status_code=404, detail=f"Model directory '{mdir}' not found. Run /train first.")

    try:
        results = predict_from_model_dir(mdir, req.features)
        if not results:
            raise HTTPException(status_code=404, detail="No trained model bundles found in model directory.")
        return {
            "model_dir": str(mdir),
            "input_features": req.features,
            "predictions": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/circuit", tags=["Quantum Circuit"])
def get_quantum_circuit(model_dir: str = "models/experiment_001"):
    """
    Retrieve ASCII circuit diagram and quantum specifications of the trained VQC.
    """
    mdir = Path(model_dir)
    if not mdir.is_absolute():
        mdir = PROJECT_ROOT / mdir
    vqc_path = mdir / "qiskit_variational_classifier.joblib"
    if not vqc_path.exists():
        raise HTTPException(status_code=404, detail=f"VQC model not found at {vqc_path}.")

    bundle = joblib.load(vqc_path)
    qmodel = bundle["model"]
    return {
        "model": "QiskitVariationalClassifier",
        "n_qubits": qmodel.n_qubits,
        "reps": qmodel.reps,
        "features": bundle["features"],
        "metadata": qmodel.get_metadata(),
        "circuit_diagram": qmodel.get_circuit_diagram(),
    }


@app.get("/report/{name}", tags=["Reporting"])
def get_report(name: str):
    """Retrieve saved experiment benchmark JSON report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{name}' not found.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/models", tags=["Model Registry"])
def list_models():
    """List all saved model artifacts in the models/ registry."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = sorted(str(p.relative_to(PROJECT_ROOT)) for p in MODEL_DIR.rglob("*.joblib"))
    return {"models": artifacts, "count": len(artifacts)}
