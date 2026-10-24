import pandas as pd
from soilq.data import audit_dataset

def test_audit_csv(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"soil_moisture": [10, 20, None], "class": ["low", "high", "low"]}).to_csv(p, index=False)
    result = audit_dataset(p)
    assert result["file_count"] == 1
    assert result["files"][0]["table"]["rows"] == 3
    assert result["files"][0]["table"]["missing_count"]["soil_moisture"] == 1
