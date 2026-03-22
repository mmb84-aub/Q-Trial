from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class HypothesisTestParams(BaseModel):
    numeric_column: str = Field(description="Numeric column to test")
    group_column: str = Field(description="Categorical column defining the two groups")
    group_a: str = Field(description="Label for group A")
    group_b: str = Field(description="Label for group B")
    alpha: float = Field(default=0.05, description="Significance level (default 0.05)")


@tool(
    name="hypothesis_test",
    description=(
        "Compare a numeric column between exactly two groups. "
        "Automatically selects an independent t-test (when both groups pass "
        "normality or n >= 30) or Mann-Whitney U (non-parametric fallback). "
        "Returns test name, statistic, p-value, means, medians, and sample sizes."
    ),
    params_model=HypothesisTestParams,
    category="stats",
)
def hypothesis_test(params: HypothesisTestParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in (params.numeric_column, params.group_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")

    mask_a = df[params.group_column].astype(str) == str(params.group_a)
    mask_b = df[params.group_column].astype(str) == str(params.group_b)

    # Track rows before and after listwise deletion
    a_raw = df.loc[mask_a, params.numeric_column]
    b_raw = df.loc[mask_b, params.numeric_column]
    a = a_raw.dropna().to_numpy(dtype=float)
    b = b_raw.dropna().to_numpy(dtype=float)
    rows_dropped_a = int(a_raw.isna().sum())
    rows_dropped_b = int(b_raw.isna().sum())

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            f"Not enough observations: group_a n={len(a)}, group_b n={len(b)}. "
            "Need at least 2 per group."
        )

    # Choose parametric vs non-parametric
    use_parametric = len(a) >= 30 and len(b) >= 30
    if not use_parametric:
        _, p_norm_a = stats.shapiro(a[:5000])
        _, p_norm_b = stats.shapiro(b[:5000])
        use_parametric = p_norm_a > 0.05 and p_norm_b > 0.05

    if use_parametric:
        stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        test_name = "Welch's t-test"
    else:
        stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return {
        "test": test_name,
        "numeric_column": params.numeric_column,
        "group_column": params.group_column,
        "group_a": {
            "label": str(params.group_a),
            "n": int(len(a)),
            "n_before_dropna": int(len(a_raw)),
            "rows_dropped": rows_dropped_a,
            "mean": round(float(np.mean(a)), 4),
            "std": round(float(np.std(a, ddof=1)), 4),
            "median": round(float(np.median(a)), 4),
        },
        "group_b": {
            "label": str(params.group_b),
            "n": int(len(b)),
            "n_before_dropna": int(len(b_raw)),
            "rows_dropped": rows_dropped_b,
            "mean": round(float(np.mean(b)), 4),
            "std": round(float(np.std(b, ddof=1)), 4),
            "median": round(float(np.median(b)), 4),
        },
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < params.alpha),
        "alpha": params.alpha,
        "listwise_deletion": {
            "total_rows_dropped": rows_dropped_a + rows_dropped_b,
            "note": "Rows with missing values in numeric_column were excluded per group.",
        },
    }
