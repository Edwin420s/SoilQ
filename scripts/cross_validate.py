#!/usr/bin/env python3
"""
scripts/cross_validate.py
-------------------------
Run rigorous 5-fold cross-validation across classical and quantum models.
Evaluates both standard Stratified K-Fold and Group K-Fold (grouped by Site_Season)
to detect and document any spatial or temporal leakage.

Usage:
    python scripts/cross_validate.py [--cv-type stratified|group] [--n-splits 5]
"""
import argparse
import json
import sys
from pathlib import Path

from soilq.config import PROJECT_ROOT
from soilq.pipeline import run_cross_validation
from soilq.prepare import DEFAULT_FEATURES, DEFAULT_TARGET, PROCESSED_DIR, prepare


def main():
    p = argparse.ArgumentParser(description="Cross-validation evaluation for SoilQ.")
    p.add_argument("--input", default="data/processed/kenyan_maize_merged.csv", help="Input dataset")
    p.add_argument("--cv-type", choices=["stratified", "group"], default="stratified", help="CV strategy")
    p.add_argument("--n-splits", type=int, default=5, help="Number of folds")
    p.add_argument("--output", default="data/reports/cross_validation.json", help="Output report")
    a = p.parse_args()

    input_path = Path(a.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        print("Dataset not found, preparing...")
        prepare(save=True)

    print(f"Running {a.cv_type.capitalize()} Cross-Validation ({a.n_splits} splits)...")
    res = run_cross_validation(
        input_path=str(input_path),
        target=DEFAULT_TARGET,
        features=DEFAULT_FEATURES,
        cv_type=a.cv_type,
        group_column="Site" if a.cv_type == "group" else None,
        n_splits=a.n_splits,
        include_quantum=True,
    )

    out_file = Path(a.output)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"CROSS-VALIDATION RESULTS: {res['strategy']}")
    print("=" * 75)
    print(f"{'Model':<30} {'Accuracy (Mean +- Std)':<25} {'F1 (Mean +- Std)':<25}")
    print("-" * 75)
    for mname, m in res["models"].items():
        acc_str = f"{m['accuracy_mean']:.4f} +- {m['accuracy_std']:.4f}"
        f1_str = f"{m['f1_mean']:.4f} +- {m['f1_std']:.4f}"
        print(f"{mname:<30} {acc_str:<25} {f1_str:<25}")
    print("=" * 75)
    print(f"Report saved to: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
