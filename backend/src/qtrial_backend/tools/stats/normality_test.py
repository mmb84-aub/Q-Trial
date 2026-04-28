"""
Normality testing tool — Stage 4 statistical tool.

Input:  pd.DataFrame + column name + optional test (shapiro|anderson|ks).
Output: NormalityTestResult with test name, statistic, p-value, and
        a boolean normality_rejected flag.
Does:   determines whether a column follows a normal distribution, which
        controls the choice between parametric and non-parametric downstream tests.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class NormalityParams(BaseModel):
    columns: Optional[list[str]] = Field(
        default=None,
        description=(
            "Numeric columns to test. Leave empty to test all numeric columns."
        ),
    )
    alpha: float = Field(
        default=0.05,
        description="Significance level for the normality decision (default 0.05).",
    )


@tool(
    name="normality_test",
    description=(
        "Test whether numeric columns follow a normal distribution. "
        "Uses Shapiro-Wilk for n <= 5000, D'Agostino-Pearson K² otherwise. "
        "Returns test statistic, p-value, and a plain-language verdict. "
        "Crucial for deciding between parametric and non-parametric tests."
    ),
    params_model=NormalityParams,
    category="stats",
)
def normality_test(params: NormalityParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe

    if params.columns:
        for c in params.columns:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found. Available: {ctx.column_names}")
        numeric_cols = [c for c in params.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return {"error": "No numeric columns found to test."}

    results: dict = {}
    for col in numeric_cols:
        series = df[col].dropna()
        n = int(len(series))

        if n < 8:
            results[col] = {"n": n, "skipped": "need at least 8 non-null values"}
            continue

        values = series.to_numpy(dtype=float)

        if n <= 5000:
            stat, p_val = stats.shapiro(values)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_val = stats.normaltest(values)
            test_name = "D'Agostino-Pearson K²"

        is_normal = bool(p_val >= params.alpha)
        results[col] = {
            "n": n,
            "test": test_name,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "is_normal": is_normal,
            "verdict": (
                f"Likely normal (p={p_val:.4f} >= {params.alpha})"
                if is_normal
                else f"Non-normal (p={p_val:.4f} < {params.alpha}) — prefer non-parametric tests"
            ),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
        }

    return {
        "alpha": params.alpha,
        "columns_tested": numeric_cols,
        "results": results,
    }
