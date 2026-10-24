#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from soilq.data import audit_dataset

def main():
    p = argparse.ArgumentParser(description="Audit dataset files without choosing a target.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/reports/audit.json")
    a = p.parse_args()
    result = audit_dataset(a.input)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Audited {result['file_count']} file(s); report: {out.resolve()}")
    for f in result["files"]:
        print(f"- {f['name']} ({f.get('size_bytes', 0)} bytes)")
        for key in ("table",):
            if key in f: print(f"  {f[key]['rows']} rows x {f[key]['columns_count']} columns: {f[key]['columns']}")
        for sheet, s in f.get("tables", {}).items():
            print(f"  sheet {sheet}: {s['rows']} rows x {s['columns_count']} columns: {s['columns']}")
        for member in f.get("members", []):
            print(f"  archive: {member['name']} ({member['size_bytes']} bytes)")
            if "sampled_table" in member:
                s = member["sampled_table"]; print(f"    sample columns: {s['columns']}")
if __name__ == "__main__": main()
