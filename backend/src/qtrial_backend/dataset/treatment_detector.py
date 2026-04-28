"""
Treatment column heuristic detector.

Identifies columns that are likely to represent treatment group assignments
using two complementary checks:
  1. Column name pattern matching against known treatment-related terms.
  2. Value cardinality check: 2–5 unique values, no single value exceeds 60% of rows.

Both conditions must be satisfied for a column to be flagged.
"""
from __future__ import annotations

import re

import pandas as pd

# Terms that commonly appear in treatment/arm column names (case-insensitive)
_TREATMENT_PATTERNS = re.compile(
    r"\b(treatment|treat|arm|group|grp|intervention|control|allocation|randomized|randomised|trt)\b",
    re.IGNORECASE,
)


def detect_treatment_columns(df: pd.DataFrame) -> list[str]:
    """
    Return a list of column names that are likely treatment group assignments.

    A column is flagged when ALL of the following hold:
      - Its name matches one of the treatment-related name patterns.
      - It has between 2 and 5 unique non-null values.
      - No single value accounts for more than 60% of all rows.
    """
    candidates: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return candidates

    for col in df.columns:
        # 1. Name pattern check
        if not _TREATMENT_PATTERNS.search(col):
            continue

        # 2. Cardinality check
        value_counts = df[col].value_counts(dropna=True)
        n_unique = len(value_counts)
        if not (2 <= n_unique <= 5):
            continue

        # 3. Even distribution check — no value exceeds 60% of total rows
        max_freq = value_counts.iloc[0] / n_rows
        if max_freq > 0.60:
            continue

        candidates.append(col)

    return candidates
