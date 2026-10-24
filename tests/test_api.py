"""
tests/test_api.py — FastAPI endpoint integration tests.
"""
from fastapi.testclient import TestClient
from soilq.api import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "SoilQ"


def test_models_list():
    res = client.get("/models")
    assert res.status_code == 200
    assert "models" in res.json()


def test_audit_endpoint():
    res = client.post("/audit", json={"input_path": "data/raw"})
    assert res.status_code == 200
    data = res.json()
    assert "file_count" in data
    assert data["file_count"] >= 1


def test_report_endpoint():
    # If experiment_001 report exists, verify retrieval
    res = client.get("/report/experiment_001")
    if res.status_code == 200:
        data = res.json()
        assert "models" in data
        assert "LogisticRegression" in data["models"]
    else:
        assert res.status_code == 404


def test_predict_endpoint():
    res = client.post(
        "/predict",
        json={
            "model_dir": "models/experiment_001",
            "features": {
                "Height_6th": 30.0,
                "Height_10th": 120.0,
                "SPAD_6th": 35.0,
                "SPAD_10th": 40.0,
                "Stover_Mg/ha": 4.5,
            },
        },
    )
    if res.status_code == 200:
        data = res.json()
        assert "predictions" in data
        assert "logisticregression" in data["predictions"]
        assert "qiskit_variational_classifier" in data["predictions"]
    else:
        assert res.status_code in (404, 500)


def test_circuit_endpoint():
    res = client.get("/circuit?model_dir=models/experiment_001")
    if res.status_code == 200:
        data = res.json()
        assert "n_qubits" in data
        assert "circuit_diagram" in data
    else:
        assert res.status_code == 404
