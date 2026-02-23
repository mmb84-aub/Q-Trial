from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class OutlierParams(BaseModel):
    columns: Optional[list[str]] = Field(
        default=None,
        description=(
            "Numeric columns to check. Leave empty to check all numeric columns."
        ),
    )
    iqr_multiplier: float = Field(
        default=1.5,
        description="IQR fence multiplier (default 1.5; use 3.0 for extreme outliers only).",
    )
    zscore_threshold: float = Field(
        default=3.0,
        description="Absolute Z-score threshold for outlier flagging (default 3.0).",
    )
    include_mad: bool = Field(
        default=True,
        description=(
            "Also run Median Absolute Deviation (MAD) method. "
            "More robust than Z-score for skewed distributions."
        ),
    )


@tool(
    name="outlier_detection",
    description=(
        "Detect outliers in numeric columns using IQR fences, Z-score, and MAD. "
        "IQR: values beyond Q1/Q3 ± multiplier*IQR. "
        "Z-score: values with |z| > threshold. "
        "MAD: values beyond median ± threshold * (MAD / 0.6745), robust for skewed distributions. "
        "Returns outlier counts, percentages, and up to 10 example row indices per column."
    ),
    params_model=OutlierParams,
    category="stats",
)
def outlier_detection(params: OutlierParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe

    if params.columns:
        for c in params.columns:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found. Available: {ctx.column_names}")
        numeric_cols = [c for c in params.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return {"error": "No numeric columns found to analyse."}

    results: dict = {}
    for col in numeric_cols:
        series = df[col].dropna()
        n = int(len(series))
        if n < 4:
            results[col] = {"n": n, "skipped": "too few non-null values"}
            continue

        # IQR method
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - params.iqr_multiplier * iqr
        upper = q3 + params.iqr_multiplier * iqr
        iqr_mask = (df[col] < lower) | (df[col] > upper)
        iqr_indices = df.index[iqr_mask & df[col].notna()].tolist()

        # Z-score method
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        if std == 0:
            z_indices: list = []
        else:
            z_scores = ((df[col] - mean) / std).abs()
            z_mask = z_scores > params.zscore_threshold
            z_indices = df.index[z_mask & df[col].notna()].tolist()

        # MAD method
        mad_entry: dict = {}
        if params.include_mad:
            median_val = float(series.median())
            mad = float((series - median_val).abs().median())
            if mad > 0:
                modified_z = 0.6745 * (series - median_val) / mad
                mad_mask = modified_z.abs() > params.zscore_threshold
                mad_indices = df.index[mad_mask & df[col].notna()].tolist()
                mad_entry = {
                    "threshold": params.zscore_threshold,
                    "mad_value": round(mad, 4),
                    "n_outliers": int(len(mad_indices)),
                    "pct_outliers": round(len(mad_indices) / n * 100, 2),
                    "example_indices": mad_indices[:10],
                }
            else:
                mad_entry = {"skipped": "MAD=0 (constant values in majority)"}

        col_result: dict = {
            "n_non_null": n,
            "iqr_method": {
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "n_outliers": int(len(iqr_indices)),
                "pct_outliers": round(len(iqr_indices) / n * 100, 2),
                "example_indices": iqr_indices[:10],
            },
            "zscore_method": {
                "threshold": params.zscore_threshold,
                "n_outliers": int(len(z_indices)),
                "pct_outliers": round(len(z_indices) / n * 100, 2),
                "example_indices": z_indices[:10],
            },
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }
        if mad_entry:
            col_result["mad_method"] = mad_entry
        results[col] = col_result

    return {
        "columns_analysed": numeric_cols,
        "iqr_multiplier": params.iqr_multiplier,
        "zscore_threshold": params.zscore_threshold,
        "results": results,
    }
