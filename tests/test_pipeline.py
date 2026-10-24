"""
tests/test_pipeline.py — Pipeline validation and benchmark smoke tests.
"""
import numpy as np
import pandas as pd
import pytest
from soilq.pipeline import train_benchmark


def _make_csv(tmp_path, n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "height": rng.uniform(20, 60, n),
            "chloro": rng.uniform(25, 55, n),
            "stover": rng.uniform(1, 10, n),
            "label": (rng.random(n) > 0.5).astype(int),
        }
    )
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)
    return p


def test_rejects_multiclass(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"x": range(15), "target": ["a", "b", "c"] * 5}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="exactly two"):
        train_benchmark(str(p), "target", ["x"], str(tmp_path / "models"))


def test_rejects_missing_target(tmp_path):
    p = _make_csv(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        train_benchmark(str(p), "nonexistent", ["height"], str(tmp_path / "models"))


def test_rejects_missing_feature(tmp_path):
    p = _make_csv(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        train_benchmark(str(p), "label", ["not_a_col"], str(tmp_path / "models"))


def test_rejects_target_in_features(tmp_path):
    p = _make_csv(tmp_path)
    with pytest.raises(ValueError, match="must not be in the feature list"):
        train_benchmark(str(p), "label", ["height", "label"], str(tmp_path / "models"))


def test_full_pipeline_smoke(tmp_path):
    """End-to-end: five models trained, report written, returns dict with expected keys."""
    p = _make_csv(tmp_path, n=48)
    report = train_benchmark(
        str(p),
        "label",
        ["height", "chloro", "stover"],
        str(tmp_path / "models"),
        str(tmp_path / "report.json"),
        quantum_maxiter=10,  # keep test fast
    )
    assert "models" in report
    assert len(report["models"]) == 5  # LR, SVM, RF, Qiskit VQC, Quantum Kernel
    for mname, mvals in report["models"].items():
        assert "accuracy" in mvals, f"Missing 'accuracy' in {mname}"
        assert "f1" in mvals, f"Missing 'f1' in {mname}"
    assert report["rows_used"] == 48
    import json
    from pathlib import Path
    saved = json.loads(Path(report["report_path"]).read_text())
    assert saved["rows_used"] == 48

    # Test inference function
    from soilq.pipeline import predict_from_model_dir
    preds = predict_from_model_dir(
        str(tmp_path / "models"),
        {"height": 35.0, "chloro": 40.0, "stover": 5.0},
    )
    assert "logisticregression" in preds
    assert "qiskit_variational_classifier" in preds
    assert "quantum_kernel_classifier" in preds
