#!/usr/bin/env python3
"""
scripts/train.py
----------------
Benchmark classical ML models against the Qiskit variational classifier
on a specified dataset file.

Example (after prepare_data.py)
-------------------------------
    python scripts/train.py \\
        --input data/processed/kenyan_maize_merged.csv \\
        --target high_yield \\
        --features Height_6th Height_10th SPAD_6th SPAD_10th "Stover_Mg/ha" \\
        --positive-label 1 \\
        --output-dir models/experiment_001 \\
        --report data/reports/experiment_001.json \\
        --quantum-maxiter 60
"""
import argparse
import json
from soilq.pipeline import train_benchmark


def main():
    p = argparse.ArgumentParser(
        description="Benchmark classical models against a Qiskit variational classifier."
    )
    p.add_argument("--input", required=True, help="CSV/TSV/Excel/Parquet data file")
    p.add_argument("--target", required=True, help="Binary target column name")
    p.add_argument("--features", nargs="+", required=True, help="Feature column names")
    p.add_argument("--positive-label", help="String label of the positive class")
    p.add_argument("--output-dir", default="models/experiment", help="Directory for model files")
    p.add_argument("--report", help="Path for the JSON experiment report")
    p.add_argument("--test-size", type=float, default=0.25, help="Holdout fraction (0.1–0.4)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--quantum-maxiter", type=int, default=60, help="Max COBYLA iterations")
    a = p.parse_args()

    r = train_benchmark(
        a.input,
        a.target,
        a.features,
        a.output_dir,
        a.report,
        a.positive_label,
        a.test_size,
        a.seed,
        a.quantum_maxiter,
    )

    summary = {
        "report_path": r["report_path"],
        "rows_used": r["rows_used"],
        "holdout_rows": r["holdout_rows"],
        "f1": {k: round(v.get("f1", 0), 4) for k, v in r["models"].items()},
        "balanced_accuracy": {k: round(v.get("balanced_accuracy", 0), 4) for k, v in r["models"].items()},
        "roc_auc": {k: round(v.get("roc_auc") or 0, 4) for k, v in r["models"].items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
