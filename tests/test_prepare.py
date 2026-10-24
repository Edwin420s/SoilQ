"""
tests/test_prepare.py — Dataset preparation and merging tests.
"""
from pathlib import Path
import pandas as pd
import pytest


def test_prepare_runs(tmp_path, monkeypatch):
    """
    prepare() should merge toy CSVs into a single DataFrame with the
    expected columns and a balanced binary target.
    """
    # Create minimal fake CSVs mimicking the real dataset structure
    yield_df = pd.DataFrame({
        "Site ": ["Chuka"] * 8, "Season": ["SR16"] * 8,
        "Tillage": ["D0"] * 8, "Soil Input": ["C"] * 8,
        "Rep": list(range(1, 9)),
        "Yield_Mg/ha": [0.5, 1.0, 0.3, 2.0, 0.6, 1.5, 0.2, 3.0],
        "Stover_Mg/ha": [2.0] * 8,
    })
    plant_df = pd.DataFrame({
        "Site": ["Chuka"] * 8, "Season": ["SR16"] * 8,
        "Tillage": ["D0"] * 8, "Soil Input": ["C"] * 8,
        "Rep": list(range(1, 9)),
        "6th leaf stage_cm": [25.0] * 8,
        "10th leaf stage_cm": [110.0] * 8,
    })
    chloro_df = pd.DataFrame({
        "Site": ["Chuka"] * 8, "Season": ["SR16"] * 8,
        "Tillage": ["D0"] * 8, "Soil Input": ["C"] * 8,
        "Rep": list(range(1, 9)),
        "6th_leaf_SPAD value": [35.0] * 8,
        "10th leaf_SPAD value": [38.0] * 8,
    })
    soil_df = pd.DataFrame({
        "Site": ["Chuka"] * 8, "Season": ["SR16"] * 8,
        "Tillage": ["D0"] * 8, "Soil Input": ["C"] * 8,
        "Rep": list(range(1, 9)),
        "Phenology": ["Vegetative phase"] * 8,
        "Date": ["10/11/2016"] * 8,
        "Depth 10cm_amount mm": [30] * 8,
        "Depth 20cm_Amount mm": [60] * 8,
        "Depth 30cm_Amount mm": [90] * 8,
        "Depth 40cm_Amount mm": [120] * 8,
        "Depth 50cm_Amount mm": [150] * 8,
    })
    rain_df = pd.DataFrame({
        "Season": ["SR16"], "Date": ["10/11/2016"],
        "Chuka(amount in mm)": [5.0], "Kandara (amount in mm)": [3.0],
    })

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    yield_df.to_csv(raw_dir / "Maize yield_Dataset_Version 2.csv", index=False)
    plant_df.to_csv(raw_dir / "Plant height_Dataset_Version 2.csv", index=False)
    chloro_df.to_csv(raw_dir / "Relative Chlorophyll content_Dataset_Version 2.csv", index=False)
    soil_df.to_csv(raw_dir / "Soil water content_Dataset_Version 2.csv", index=False)
    rain_df.to_csv(raw_dir / "Rainfall_Dataset_Version 2.csv", index=False)

    # Monkey-patch prepare module paths
    import soilq.prepare as prep_mod
    monkeypatch.setattr(prep_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(prep_mod, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(prep_mod, "UPSTREAM_DIR", raw_dir)

    merged, prov = prep_mod.prepare(save=True)

    assert "high_yield" in merged.columns
    assert set(merged["high_yield"].unique()).issubset({0, 1})
    assert len(merged) == 8
    assert "Height_6th" in merged.columns
    assert "SPAD_6th" in merged.columns
    assert (tmp_path / "processed" / "kenyan_maize_merged.csv").exists()
