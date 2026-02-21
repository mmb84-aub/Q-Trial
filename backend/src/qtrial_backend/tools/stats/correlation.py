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
    return {
        "method": params.method,
        "columns": list(corr.columns),
        "matrix": corr.where(pd.notnull(corr), None).to_dict(),
    }
