#!/usr/bin/env python3
"""
scripts/prepare_data.py
-----------------------
Copy the five raw CSVs from the upstream Quantum/ directory into
data/raw/, merge them into a single analysis-ready CSV and write a
provenance JSON to data/reports/prepare_report.json.

Usage
-----
    python scripts/prepare_data.py [--output-dir data/processed] [--report data/reports/prepare_report.json]
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Merge the five Kenyan maize dataset CSVs into a single analysis-ready table."
    )
    p.add_argument("--output-dir", default="data/processed", help="Directory for merged CSV")
    p.add_argument("--report", default="data/reports/prepare_report.json", help="Provenance report path")
    a = p.parse_args()

    from soilq.prepare import PROCESSED_DIR, prepare

    merged, prov = prepare(save=True)
    prov["merged_csv"] = str((Path(a.output_dir) / "kenyan_maize_merged.csv").resolve())

    report_path = Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")

    print(f"Merged {len(merged)} rows x {len(merged.columns)} columns")
    print(f"Target 'high_yield' class counts: {merged['high_yield'].value_counts().to_dict()}")
    print("Missing per column (non-zero only):")
    miss = merged.isna().sum()
    for col, cnt in miss[miss > 0].items():
        print(f"  {col}: {cnt} ({100 * cnt / len(merged):.1f} %)")
    print(f"Provenance report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
