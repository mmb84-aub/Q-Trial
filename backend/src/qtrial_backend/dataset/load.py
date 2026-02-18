from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = p.suffix.lower()
    if suffix in [".csv"]:
        return pd.read_csv(p)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(p)

    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")
