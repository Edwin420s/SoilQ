"""
SoilQ data preparation: merge the five Kenyan maize field-experiment CSVs
into a single analysis-ready table and persist it to data/processed/.

Dataset source: https://data.mendeley.com/datasets/8bx88zpvdp/2
Authors: Mutuku et al.  |  License: CC BY 4.0

Each row in the merged table represents one plot observation identified by
(Site, Season, Tillage, Soil_Input, Rep).

Target variable: high_yield (binary proxy)
  • 1  → Yield_Mg/ha > 0.735  (observed dataset median across all plots)
  • 0  → Yield_Mg/ha ≤ 0.735
  
Rationale: The dataset median provides a balanced split (224 / 224) and
is defensible as a relative within-dataset threshold distinguishing
higher-yielding from lower-yielding treatment combinations. This is
explicitly a proxy label; it is not a validated irrigation-decision signal.

Features used (all numeric, from on-station measurements):
  Height_6th    — plant height at 6th-leaf stage (cm)
  Height_10th   — plant height at 10th-leaf stage (cm)
  SPAD_6th      — relative chlorophyll at 6th-leaf stage (SPAD units)
  SPAD_10th     — relative chlorophyll at 10th-leaf stage (SPAD units)
  Stover_Mg/ha  — stover biomass yield (Mg/ha)

Soil water content (SWC) columns are only measured in SR seasons and thus
have 50 % missingness after the join; they are retained as optional extra
features but excluded from the default feature set to avoid halving the
usable sample.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent  # SoilQ 1/
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# Source CSVs (placed in data/raw/ by the prepare step)
SOURCE_FILES = {
    "soil":   "Soil water content_Dataset_Version 2.csv",
    "rain":   "Rainfall_Dataset_Version 2.csv",
    "yield":  "Maize yield_Dataset_Version 2.csv",
    "plant":  "Plant height_Dataset_Version 2.csv",
    "chloro": "Relative Chlorophyll content_Dataset_Version 2.csv",
}

# Canonical upstream location (sibling of the SoilQ 1/ directory)
UPSTREAM_DIR = PROJECT_ROOT.parent  # /home/.../Quantum/


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_raws() -> dict[str, Path]:
    """Copy source CSVs from the upstream directory into data/raw/ if needed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in SOURCE_FILES.items():
        dest = RAW_DIR / filename
        if not dest.exists():
            src = UPSTREAM_DIR / filename
            if src.exists():
                shutil.copy2(src, dest)
            else:
                raise FileNotFoundError(
                    f"Cannot find '{filename}'. "
                    f"Expected at {src} or already in {RAW_DIR}."
                )
        paths[key] = dest
    return paths


def load_raw(paths: dict[str, Path] | None = None) -> dict[str, pd.DataFrame]:
    """Load all five source CSVs and return them as a dict of DataFrames."""
    if paths is None:
        paths = _copy_raws()
    dfs: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        dfs[key] = pd.read_csv(path)
    return dfs


def _fix_soil(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and aggregate soil-water-content data to plot level."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in [
        "Depth 20cm_Amount mm",
        "Depth 30cm_Amount mm",
        "Depth 40cm_Amount mm",
        "Depth 50cm_Amount mm",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = (
        df.groupby(["Site", "Season", "Tillage", "Soil Input", "Rep"], as_index=False)
        .agg(
            SWC_10cm=("Depth 10cm_amount mm", "mean"),
            SWC_20cm=("Depth 20cm_Amount mm", "mean"),
            SWC_30cm=("Depth 30cm_Amount mm", "mean"),
            SWC_40cm=("Depth 40cm_Amount mm", "mean"),
            SWC_50cm=("Depth 50cm_Amount mm", "mean"),
        )
        .rename(columns={"Soil Input": "Soil_Input"})
    )
    return agg


def build_merged(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge five source tables into one analysis-ready DataFrame.
    Returns the merged table with the binary proxy target `high_yield`.
    """
    yield_df = dfs["yield"].rename(columns={"Site ": "Site", "Soil Input": "Soil_Input"})
    plant_df = dfs["plant"].rename(
        columns={
            "Soil Input": "Soil_Input",
            "6th leaf stage_cm": "Height_6th",
            "10th leaf stage_cm": "Height_10th",
        }
    )
    chloro_df = dfs["chloro"].rename(
        columns={
            "Soil Input": "Soil_Input",
            "6th_leaf_SPAD value": "SPAD_6th",
            "10th leaf_SPAD value": "SPAD_10th",
        }
    )
    soil_agg = _fix_soil(dfs["soil"])

    keys = ["Site", "Season", "Tillage", "Soil_Input", "Rep"]
    merged = (
        yield_df
        .merge(plant_df[keys + ["Height_6th", "Height_10th"]], on=keys, how="left")
        .merge(chloro_df[keys + ["SPAD_6th", "SPAD_10th"]], on=keys, how="left")
        .merge(soil_agg[keys + ["SWC_10cm", "SWC_20cm", "SWC_30cm", "SWC_40cm", "SWC_50cm"]], on=keys, how="left")
    )

    # ── Binary proxy target ──────────────────────────────────────────────────
    # Threshold: dataset median yield across all 448 plot-season observations.
    # Produces a perfectly balanced 224 / 224 split.  This is a relative
    # within-dataset proxy, not a validated agronomic irrigation threshold.
    threshold = merged["Yield_Mg/ha"].median()
    merged["high_yield"] = (merged["Yield_Mg/ha"] > threshold).astype(int)
    merged.attrs["yield_threshold"] = float(threshold)

    return merged


def prepare(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Full preparation pipeline: copy raws → load → merge → optionally save.

    Returns (merged_df, provenance_dict).
    """
    paths = _copy_raws()
    dfs = load_raw(paths)
    merged = build_merged(dfs)

    provenance = {
        "source": "Mutuku et al. — Mendeley Data https://data.mendeley.com/datasets/8bx88zpvdp/2",
        "license": "CC BY 4.0",
        "files": {key: {"path": str(p), "sha256": _sha256(p)} for key, p in paths.items()},
        "rows": int(len(merged)),
        "columns": list(merged.columns),
        "target": "high_yield",
        "target_definition": "Yield_Mg/ha > dataset median (0.735 Mg/ha)",
        "class_counts": merged["high_yield"].value_counts().to_dict(),
        "missing_counts": merged.isna().sum().to_dict(),
        "note": (
            "Proxy label derived from dataset-median yield split. "
            "Not a validated irrigation-urgency label. "
            "SWC columns have ~50 % missingness (SR seasons only). "
            "Stratified random split is a baseline; "
            "season-aware evaluation is preferred for time-structured data."
        ),
    }

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / "kenyan_maize_merged.csv"
        merged.to_csv(out, index=False)
        provenance["merged_csv"] = str(out)

    return merged, provenance


# ── Convenience accessors ────────────────────────────────────────────────────

#: Default feature set — all fully observed numeric agronomic measurements.
DEFAULT_FEATURES = ["Height_6th", "Height_10th", "SPAD_6th", "SPAD_10th", "Stover_Mg/ha"]

#: Default binary target column
DEFAULT_TARGET = "high_yield"

#: Optional soil-water features (SR seasons only; ~50 % missing after merge)
SOIL_FEATURES = ["SWC_10cm", "SWC_20cm", "SWC_30cm", "SWC_40cm", "SWC_50cm"]


def load_prepared() -> pd.DataFrame:
    """Load the pre-merged CSV from data/processed/. Runs prepare() if absent."""
    out = PROCESSED_DIR / "kenyan_maize_merged.csv"
    if not out.exists():
        prepare(save=True)
    return pd.read_csv(out)
