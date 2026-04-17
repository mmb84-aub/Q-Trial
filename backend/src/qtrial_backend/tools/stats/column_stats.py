"""
Descriptive column statistics — Stage 4 statistical tool.

Input:  pd.DataFrame + column name(s).
Output: ColumnStatsResult with count, mean, std, min, max, percentiles,
        n_missing, n_unique for numeric columns; frequency table for categoricals.
Does:   produces per-column summary statistics as the LLM agent's first
        step when profiling any column of interest.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field
from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


def _safe_float(val: object) -> float | None:
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


class ColumnStatsParams(BaseModel):
    column: str = Field(description="Exact column name to analyse")


@tool(
    name="column_detailed_stats",
    description=(
        "Get detailed statistics for a single column. "
        "Numeric: count, mean, std, min, max, percentiles, skewness, kurtosis. "
        "Categorical: count, unique, top values with frequencies, mode."
    ),
    params_model=ColumnStatsParams,
    category="stats",
)
def column_detailed_stats(params: ColumnStatsParams, ctx: AgentContext) -> dict:
    col = params.column
    if col not in ctx.dataframe.columns:
        raise ValueError(
            f"Column '{col}' not found. Available: {ctx.column_names}"
        )

    series = ctx.dataframe[col]
    base: dict = {
        "column": col,
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "null_count": int(series.isna().sum()),
        "null_pct": round(float(series.isna().mean() * 100), 2),
        "unique_count": int(series.nunique()),
    }

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        base.update(
            {
                "mean": _safe_float(series.mean()),
                "std": _safe_float(series.std()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "percentile_5": _safe_float(desc.get("5%")),
                "percentile_25": _safe_float(desc.get("25%")),
                "median": _safe_float(desc.get("50%")),
                "percentile_75": _safe_float(desc.get("75%")),
                "percentile_95": _safe_float(desc.get("95%")),
                "skewness": _safe_float(series.skew()),
                "kurtosis": _safe_float(series.kurtosis()),
                "n_zeros": int((series == 0).sum()),
            }
        )
    else:
        vc = series.value_counts().head(10)
        base["top_values"] = {str(k): int(v) for k, v in vc.items()}
        mode = series.mode()
        base["mode"] = str(mode.iloc[0]) if not mode.empty else None

    return base
