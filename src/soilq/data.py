from __future__ import annotations
import hashlib, zipfile
from pathlib import Path
import pandas as pd

SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_table(path, sheet_name=0):
    p = Path(path); ext = p.suffix.lower()
    if ext == ".csv": return pd.read_csv(p)
    if ext == ".tsv": return pd.read_csv(p, sep="\t")
    if ext in {".xlsx", ".xls"}: return pd.read_excel(p, sheet_name=sheet_name)
    if ext == ".json":
        try: return pd.read_json(p)
        except ValueError: return pd.read_json(p, lines=True)
    if ext == ".jsonl": return pd.read_json(p, lines=True)
    if ext == ".parquet": return pd.read_parquet(p)
    raise ValueError(f"Unsupported table format: {p.name} ({ext})")

def discover_files(root):
    p = Path(root)
    if p.is_file(): return [p]
    if not p.exists(): raise FileNotFoundError(f"Input path does not exist: {p}")
    return [x for x in sorted(p.rglob("*")) if x.is_file() and
            (x.suffix.lower() in SUPPORTED or x.suffix.lower() == ".zip")]

def frame_summary(df):
    missing = df.isna().sum()
    sample = {}
    for col in df.columns:
        vals = df[col].dropna().head(5).tolist()
        sample[str(col)] = [v.isoformat() if hasattr(v, "isoformat") else
            (v.item() if hasattr(v, "item") else v) for v in vals]
    return {
        "rows": int(len(df)), "columns_count": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_count": {str(k): int(v) for k, v in missing.items()},
        "missing_percent": {str(k): round(float(v / max(len(df), 1) * 100), 3) for k, v in missing.items()},
        "unique_count": {str(c): int(df[c].nunique(dropna=True)) for c in df.columns},
        "sample_values": sample,
        "likely_context_columns": [str(c) for c in df.columns if any(t in str(c).lower()
            for t in ("date", "time", "season", "site", "plot", "treat", "farm", "field", "stage", "replicate", "block"))],
        "likely_moisture_columns": [str(c) for c in df.columns if any(t in str(c).lower()
            for t in ("moist", "water", "soil", "smc", "rain", "precip", "irrig", "humidity"))]
    }

def audit_path(path):
    p = Path(path)
    result = {"path": str(p), "name": p.name, "suffix": p.suffix.lower()}
    if not p.exists(): return {**result, "error": "file not found"}
    result["size_bytes"] = p.stat().st_size
    if p.suffix.lower() == ".zip":
        members = []
        with zipfile.ZipFile(p) as zf:
            for info in zf.infolist():
                if info.is_dir(): continue
                entry = {"name": info.filename, "size_bytes": info.file_size}
                suffix = Path(info.filename).suffix.lower()
                if suffix in {".csv", ".tsv"} and info.file_size <= 100_000_000:
                    try:
                        with zf.open(info) as f:
                            df = pd.read_csv(f, sep="\t" if suffix == ".tsv" else ",", nrows=5000)
                        entry["sampled_table"] = frame_summary(df)
                    except Exception as exc: entry["read_error"] = str(exc)
                elif suffix in {".xlsx", ".xls"}:
                    entry["note"] = "Spreadsheet listed but not inspected inside archive; extract it first."
                members.append(entry)
        result["members"] = members; result["sha256"] = sha256_file(p)
        return result
    if p.suffix.lower() in SUPPORTED:
        if p.suffix.lower() in {".xlsx", ".xls"}:
            xls = pd.ExcelFile(p); result["sheets"] = xls.sheet_names
            result["tables"] = {s: frame_summary(pd.read_excel(p, sheet_name=s, nrows=5000)) for s in xls.sheet_names}
        else: result["table"] = frame_summary(read_table(p))
        result["sha256"] = sha256_file(p)
        return result
    return {**result, "note": "Unsupported file; inventory only."}

def audit_dataset(root):
    found = discover_files(root)
    return {"input": str(root), "file_count": len(found), "files": [audit_path(p) for p in found]}
