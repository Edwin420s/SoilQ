import pandas as pd
import pytest
from soilq.pipeline import train_benchmark

def test_rejects_multiclass(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"x": range(15), "target": ["a", "b", "c"] * 5}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="exactly two"):
        train_benchmark(str(p), "target", ["x"], str(tmp_path / "models"))
