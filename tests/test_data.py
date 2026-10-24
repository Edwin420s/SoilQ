"""
tests/test_data.py — Data loading and audit smoke tests.
"""
import pandas as pd
import pytest
from soilq.data import audit_dataset, read_table


def test_audit_csv(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"soil_moisture": [10, 20, None], "class": ["low", "high", "low"]}).to_csv(
        p, index=False
    )
    result = audit_dataset(p)
    assert result["file_count"] == 1
    assert result["files"][0]["table"]["rows"] == 3
    assert result["files"][0]["table"]["missing_count"]["soil_moisture"] == 1


def test_audit_directory(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n")
    (tmp_path / "b.csv").write_text("a,b\n5,6\n")
    result = audit_dataset(tmp_path)
    assert result["file_count"] == 2


def test_read_table_tsv(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("x\ty\n1\t2\n3\t4\n")
    df = read_table(p)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2


def test_audit_missing_path():
    with pytest.raises(FileNotFoundError):
        audit_dataset("/nonexistent/path")
