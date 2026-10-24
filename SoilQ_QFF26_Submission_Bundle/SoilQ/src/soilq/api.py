from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .data import audit_dataset
from .pipeline import train_benchmark

app = FastAPI(title="SoilQ Research API", version="0.1.0",
              description="Local research API; not a farm-control system.")

class AuditRequest(BaseModel):
    input_path: str

class TrainRequest(BaseModel):
    input_path: str
    target: str
    features: list[str] = Field(min_length=1)
    output_dir: str = "models/api_run"
    report_path: str | None = None
    positive_label: str | None = None
    test_size: float = Field(default=0.25, gt=0.1, lt=0.5)
    seed: int = 42
    quantum_maxiter: int = Field(default=40, ge=10, le=200)

@app.get("/health")
def health():
    return {"status": "ok", "service": "SoilQ", "note": "Research prototype only."}

@app.post("/audit")
def audit(req: AuditRequest):
    try: return audit_dataset(req.input_path)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc))

@app.post("/train")
def train(req: TrainRequest):
    try:
        return train_benchmark(req.input_path, req.target, req.features, req.output_dir,
            req.report_path, req.positive_label, req.test_size, req.seed, req.quantum_maxiter)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc))
