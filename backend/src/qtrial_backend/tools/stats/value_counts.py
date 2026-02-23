from __future__ import annotations

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class ValueCountsParams(BaseModel):
    column: str = Field(description="Column name")
    top_n: int = Field(default=20, description="Number of top values to return")
    normalize: bool = Field(
        default=False, description="Return proportions instead of counts"
    )


@tool(
    name="value_counts",
    description=(
        "Get frequency distribution for a column. "
        "Returns value counts (or proportions) for the top N values."
    ),
    params_model=ValueCountsParams,
    category="stats",
)
def value_counts(params: ValueCountsParams, ctx: AgentContext) -> dict:
    col = params.column
    if col not in ctx.dataframe.columns:
        raise ValueError(
            f"Column '{col}' not found. Available: {ctx.column_names}"
        )

    series = ctx.dataframe[col]
    vc = series.value_counts(normalize=params.normalize).head(params.top_n)
    return {
        "column": col,
        "total_values": int(series.count()),
        "unique_count": int(series.nunique()),
        "normalized": params.normalize,
        "values": {
            str(k): round(float(v), 4) if params.normalize else int(v)
            for k, v in vc.items()
        },
    }
