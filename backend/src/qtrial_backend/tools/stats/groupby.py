from __future__ import annotations

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class GroupByParams(BaseModel):
    group_columns: list[str] = Field(description="Columns to group by")
    target_columns: list[str] = Field(description="Columns to aggregate")
    aggregations: list[str] = Field(
        default=["mean", "median", "count", "std"],
        description="Aggregation functions to apply",
    )


@tool(
    name="group_by_summary",
    description=(
        "Group-by aggregation on selected columns. "
        "Returns aggregated statistics per group (limited to top 50 groups)."
    ),
    params_model=GroupByParams,
    category="stats",
)
def group_by_summary(params: GroupByParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for c in params.group_columns + params.target_columns:
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found. Available: {ctx.column_names}"
            )

    grouped = df.groupby(params.group_columns)[params.target_columns].agg(
        params.aggregations
    )
    # Flatten multi-level columns
    grouped.columns = ["_".join(col).strip() for col in grouped.columns]
    grouped = grouped.round(4).head(50).reset_index()

    return {
        "group_columns": params.group_columns,
        "target_columns": params.target_columns,
        "aggregations": params.aggregations,
        "n_groups": int(df.groupby(params.group_columns).ngroups),
        "results": grouped.where(grouped.notna(), None).to_dict(orient="records"),
    }
