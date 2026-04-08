"""
Input: pd.DataFrame + optional column list
Output: dict with digit preference test results per numeric column
Purpose: Detect non-uniform terminal digit distribution — a signal of manual data entry
  errors or fabrication. Standard practice in clinical trial auditing (ICH E6 GCP).
Reference: ICH E6(R2) Good Clinical Practice; Buyse et al. (1999) Statistical methods
  for assessing bioequivalence and data integrity in clinical trials.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class DigitPreferenceParams(BaseModel):
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Numeric columns to test for terminal digit preference. "
            "Null = test all numeric columns in the dataset."
        ),
    )


def _digit_preference_logic(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Core logic — callable directly with a DataFrame for programmatic use."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if columns is not None:
        cols_to_test = [c for c in columns if c in numeric_cols]
    else:
        cols_to_test = numeric_cols

    flagged: list[dict] = []
    clean: list[str] = []

    for col in cols_to_test:
        vals = df[col].dropna()
        if len(vals) < 10:
            # Too few values to run a meaningful chi-square test
            continue

        # Extract terminal (last) digit: round to integer, abs, then mod 10
        digits = vals.apply(lambda v: int(round(abs(float(v)), 0)) % 10)
        counts = {d: int((digits == d).sum()) for d in range(10)}
        total = sum(counts.values())

        # Chi-square against uniform expected distribution (n/10 per bin)
        observed = [counts[d] for d in range(10)]
        chi2_stat, p_value = stats.chisquare(observed)

        dominant = max(counts, key=lambda d: counts[d])
        dominant_pct = round(counts[dominant] / total * 100, 2) if total > 0 else 0.0

        entry = {
            "column": col,
            "chi_square": round(float(chi2_stat), 4),
            "p_value": round(float(p_value), 6),
            "digit_counts": counts,
            "dominant_digit": dominant,
            "dominant_digit_pct": dominant_pct,
        }

        if p_value < 0.05:
            flagged.append(entry)
        else:
            clean.append(col)

    return {
        "flagged_columns": flagged,
        "clean_columns": clean,
        "total_numeric_columns_tested": len(cols_to_test),
        "integrity_concern": len(flagged) > 0,
    }


@tool(
    name="digit_preference_test",
    description=(
        "Detect data fabrication via non-uniform terminal digit distribution. "
        "For each numeric column, extracts the last digit (0–9) of every value and "
        "runs a chi-square test against an expected uniform distribution. "
        "Flags columns where p < 0.05 as potential manual entry bias or data fabrication. "
        "Standard practice in clinical trial auditing under ICH E6 GCP."
    ),
    params_model=DigitPreferenceParams,
    category="stats",
)
def digit_preference_test(params: DigitPreferenceParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    if params.columns:
        missing = [c for c in params.columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. Available: {ctx.column_names}"
            )
    return _digit_preference_logic(df, params.columns)
