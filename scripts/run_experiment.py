#!/usr/bin/env python3
"""
scripts/run_experiment.py
-------------------------
End-to-end experiment: prepare data -> run full benchmark -> print summary.

Usage
-----
    python scripts/run_experiment.py
    python scripts/run_experiment.py --quantum-maxiter 40 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Run the full SoilQ experiment pipeline.")
    p.add_argument("--quantum-maxiter", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--output-dir", default="models/experiment_001")
    p.add_argument("--report", default="data/reports/experiment_001.json")
    a = p.parse_args()

    # Step 1: Prepare data
    print("Step 1/2 - Preparing and merging data ...")
    from soilq.prepare import DEFAULT_FEATURES, DEFAULT_TARGET, prepare

    merged, prov = prepare(save=True)
    print(f"  {len(merged)} rows merged. Class counts: {prov['class_counts']}")
    print(f"  Saved: {prov.get('merged_csv', 'data/processed/kenyan_maize_merged.csv')}")

    # Step 2: Train
    print(f"\nStep 2/2 - Training models (quantum_maxiter={a.quantum_maxiter}) ...")
    from soilq.pipeline import train_benchmark
    from soilq.prepare import PROCESSED_DIR

    input_csv = PROCESSED_DIR / "kenyan_maize_merged.csv"
    report = train_benchmark(
        input_path=str(input_csv),
        target=DEFAULT_TARGET,
        features=DEFAULT_FEATURES,
        output_dir=a.output_dir,
        report_path=a.report,
        positive_label="1",
        test_size=a.test_size,
        seed=a.seed,
        quantum_maxiter=a.quantum_maxiter,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    header = f"{'Model':<35} {'Accuracy':>9} {'Bal.Acc':>9} {'F1':>8} {'ROC-AUC':>9} {'Time(s)':>8}"
    print(header)
    print("-" * 60)
    for mname, mvals in report["models"].items():
        print(
            f"{mname:<35} "
            f"{mvals.get('accuracy', 0):>9.4f} "
            f"{mvals.get('balanced_accuracy', 0):>9.4f} "
            f"{mvals.get('f1', 0):>8.4f} "
            f"{(mvals.get('roc_auc') or 0):>9.4f} "
            f"{mvals.get('fit_seconds', 0):>8.2f}"
        )
    print("=" * 60)
    print(f"\nReport saved to: {report['report_path']}")

    # Note on quantum result
    q = report["models"].get("QiskitVariationalClassifier", {})
    meta = q.get("quantum_metadata", {})
    print(f"\nQuantum model: {meta.get('n_qubits', '?')} qubits, "
          f"{meta.get('reps', '?')} layers, "
          f"optimizer_success={meta.get('optimizer_success')}")
    print("Note: Classical Statevector simulation - not quantum hardware.")


if __name__ == "__main__":
    main()
