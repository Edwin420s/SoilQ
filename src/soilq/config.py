"""
SoilQ configuration — project-root resolution and environment settings.
"""
from __future__ import annotations
import os
from pathlib import Path

# The project root is the directory containing this package's src/ folder.
# Resolves correctly whether the package is installed in development mode or
# run directly from the repository.
_THIS_FILE = Path(__file__).resolve()
# src/soilq/config.py → src/soilq → src → PROJECT_ROOT
PROJECT_ROOT: Path = _THIS_FILE.parent.parent.parent

# ── Environment overrides (from .env or shell) ──────────────────────────────
LOG_LEVEL: str = os.environ.get("SOILQ_LOG_LEVEL", "INFO")
DATA_DIR: Path = Path(os.environ.get("SOILQ_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR: Path = Path(os.environ.get("SOILQ_MODEL_DIR", PROJECT_ROOT / "models"))
RANDOM_SEED: int = int(os.environ.get("SOILQ_RANDOM_SEED", "42"))
