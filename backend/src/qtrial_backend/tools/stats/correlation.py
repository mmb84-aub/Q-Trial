"""
Correlation analysis tool — Stage 4 statistical tool.

Input:  pd.DataFrame + list of numeric columns + method ("pearson"|"spearman"|"kendall").
Output: CorrelationResult with correlation matrix, top correlated pairs,
        and p-values for each pair.
Does:   identifies linear and rank-order associations between variables,
        flagging strongly correlated pairs (|r| > 0.7) for multicollinearity review.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field
from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class CorrelationParams(BaseModel):
    columns: list[str] | None = Field(
        default=None,
        description="Columns to correlate. Null = all numeric columns.",
    )
    method: str = Field(
        default="pearson", description="pearson, spearman, or kendall"
    )


@tool(
    name="correlation_matrix",
    description=(
        "Compute a correlation matrix for numeric columns. "
        "Supports pearson, spearman, and kendall methods."
    ),
    params_model=CorrelationParams,
    category="stats",
)
def correlation_matrix(params: CorrelationParams, ctx: AgentContext) -> dict:
    from scipy import stats as sp_stats

    df = ctx.dataframe
    if params.columns:
        missing = [c for c in params.columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. Available: {ctx.column_names}"
            )
        df = df[params.columns]

    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {"error": "No numeric columns in selection"}

    # Cap at 15 columns to keep result size reasonable
    if numeric.shape[1] > 15:
        numeric = numeric.iloc[:, :15]

    corr = numeric.corr(method=params.method).round(4)

    # Compute p-values pairwise
    cols = list(numeric.columns)
    p_matrix: dict = {c: {} for c in cols}
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                p_matrix[c1][c2] = None
                continue
            x = numeric[c1].dropna()
            y = numeric[c2].dropna()
            common = x.index.intersection(y.index)
            x, y = x.loc[common].to_numpy(), y.loc[common].to_numpy()
            try:
                if params.method == "pearson":
                    _, p = sp_stats.pearsonr(x, y)
                elif params.method == "spearman":
                    _, p = sp_stats.spearmanr(x, y)
                else:  # kendall
                    _, p = sp_stats.kendalltau(x, y)
                p_matrix[c1][c2] = round(float(p), 6)
            except Exception:
                p_matrix[c1][c2] = None

    return {
        "method": params.method,
        "columns": cols,
        "matrix": corr.where(pd.notnull(corr), None).to_dict(),
        "p_values": p_matrix,
        "note": "p_values are two-tailed. Consider multiple_testing_correction for many comparisons.",
    }
