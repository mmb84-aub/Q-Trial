from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class DistributionParams(BaseModel):
    column: str = Field(description="Column name")
    n_bins: int = Field(default=20, description="Number of histogram bins (numeric only)")


@tool(
    name="distribution_info",
    description=(
        "Get distribution information for a column. "
        "Numeric: histogram bins and counts, skewness, kurtosis. "
        "Categorical: entropy and cardinality ratio."
    ),
    params_model=DistributionParams,
    category="stats",
)
def distribution_info(params: DistributionParams, ctx: AgentContext) -> dict:
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
    }

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if clean.empty:
            base["histogram"] = {"bins": [], "counts": []}
            return base

        counts, bin_edges = np.histogram(clean, bins=params.n_bins)
        base["histogram"] = {
            "bin_edges": [round(float(e), 4) for e in bin_edges],
            "counts": [int(c) for c in counts],
        }
        base["skewness"] = round(float(clean.skew()), 4)
        base["kurtosis"] = round(float(clean.kurtosis()), 4)
    else:
        vc = series.value_counts(normalize=True)
        probs = vc.values
        entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
        base["entropy"] = round(entropy, 4)
        max_entropy = math.log2(max(series.nunique(), 1))
        base["max_entropy"] = round(max_entropy, 4)
        base["cardinality_ratio"] = round(
            series.nunique() / max(series.count(), 1), 4
        )

    return base
