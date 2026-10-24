#!/usr/bin/env python3
import argparse, json
from soilq.pipeline import train_benchmark

def main():
    p = argparse.ArgumentParser(description="Benchmark classical models against a Qiskit variational classifier.")
    p.add_argument("--input", required=True); p.add_argument("--target", required=True)
    p.add_argument("--features", nargs="+", required=True); p.add_argument("--positive-label")
    p.add_argument("--output-dir", default="models/experiment"); p.add_argument("--report")
    p.add_argument("--test-size", type=float, default=0.25); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantum-maxiter", type=int, default=60)
    a = p.parse_args()
    r = train_benchmark(a.input, a.target, a.features, a.output_dir, a.report,
                        a.positive_label, a.test_size, a.seed, a.quantum_maxiter)
    print(json.dumps({"report_path": r["report_path"], "rows_used": r["rows_used"],
                      "f1": {k: v.get("f1") for k,v in r["models"].items()}}, indent=2))
if __name__ == "__main__": main()
