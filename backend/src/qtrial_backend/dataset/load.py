from __future__ import annotations

from pathlib import Path
import pandas as pd

from qtrial_backend.agentic.schemas import MissingnessDisclosure


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


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ValueError with a plain-language message if the dataset is structurally empty."""
    if df.shape[1] == 0:
        raise ValueError(
            "The uploaded file contains no columns. "
            "Please check that it is a valid CSV or Excel file with at least one column."
        )
    if df.shape[0] == 0:
        raise ValueError(
            "The uploaded file contains no data rows. "
            "Please upload a file that contains at least one row of data."
        )


def classify_missingness(df: pd.DataFrame) -> dict[str, MissingnessDisclosure]:
    """
    Return a per-column missingness classification as MissingnessDisclosure objects.

    Tiers:
      >50%        → "excluded"
      20–50%      → "high_missingness_section"
      <20%        → "listwise_deletion"
    """
    result: dict[str, MissingnessDisclosure] = {}
    n_rows = len(df)
    if n_rows == 0:
        return result

    for col in df.columns:
        missing = int(df[col].isna().sum())
        rate = missing / n_rows

        if rate > 0.50:
            action = "excluded"
        elif rate >= 0.20:
            action = "high_missingness_section"
        else:
            action = "listwise_deletion"

        result[col] = MissingnessDisclosure(
            column=col,
            missingness_rate=rate,
            rows_dropped=missing if action == "listwise_deletion" else 0,
            action=action,
        )

    return result
