from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class CrosstabParams(BaseModel):
    row_column: str = Field(description="Column for cross-tab rows")
    col_column: str = Field(description="Column for cross-tab columns")
    normalize: bool = Field(default=False, description="Return proportions")
    margins: bool = Field(default=False, description="Include row/col totals")


@tool(
    name="cross_tabulation",
    description=(
        "Compute a cross-tabulation of two columns. "
        "Useful for examining relationships between categorical variables."
    ),
    params_model=CrosstabParams,
    category="stats",
)
def cross_tabulation(params: CrosstabParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for c in (params.row_column, params.col_column):
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found. Available: {ctx.column_names}"
            )

    ct = pd.crosstab(
        df[params.row_column],
        df[params.col_column],
        margins=params.margins,
        normalize="all" if params.normalize else False,
    )

    if params.normalize:
        ct = ct.round(4)

    return {
        "row_column": params.row_column,
        "col_column": params.col_column,
        "normalized": params.normalize,
        "table": {
            str(idx): {str(col): float(ct.loc[idx, col]) for col in ct.columns}
            for idx in ct.index
        },
    }
