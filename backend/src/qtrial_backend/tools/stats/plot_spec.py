"""
Plot specification generator — Stage 4 statistical tool.

Input:  pd.DataFrame + plot type + column mapping (x, y, color, facet).
Output: PlotSpec dict with Vega-Lite compatible specification and
        a text description of what the chart shows.
Does:   produces declarative chart specs the frontend can render;
        does NOT render images — purely a specification generator.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class PlotSpecParams(BaseModel):
    kind: str = Field(
        description=(
            "Plot type: 'histogram', 'boxplot', 'kde', 'bar', 'scatter', "
            "'km_curve', 'correlation_heatmap'."
        )
    )
    x_column: str = Field(description="Primary column for the x-axis or distribution.")
    y_column: Optional[str] = Field(
        default=None,
        description="Secondary column (required for scatter, km_curve).",
    )
    group_column: Optional[str] = Field(
        default=None,
        description="Column used to split data into groups/series.",
    )
    n_bins: int = Field(default=20, description="Number of bins for histogram (default 20).")
    event_codes: Optional[list[int]] = Field(
        default=None,
        description="For km_curve: event codes to treat as events (others = censored).",
    )


@tool(
    name="plot_spec",
    description=(
        "Compute the data payload for a chart without rendering it. "
        "Returns a structured JSON spec with pre-computed data that a frontend can directly render. "
        "Supported: histogram (bin edges + counts), boxplot (quartiles + whiskers + outliers), "
        "kde (kernel density x/y arrays), bar (category counts), "
        "scatter (x/y arrays, sampled if > 500 rows), "
        "km_curve (Kaplan-Meier survival function data), "
        "correlation_heatmap (correlation matrix values)."
    ),
    params_model=PlotSpecParams,
    category="stats",
)
def plot_spec(params: PlotSpecParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    kind = params.kind.lower()

    if params.x_column not in df.columns:
        raise ValueError(f"x_column '{params.x_column}' not found. Available: {ctx.column_names}")
    if params.y_column and params.y_column not in df.columns:
        raise ValueError(f"y_column '{params.y_column}' not found. Available: {ctx.column_names}")
    if params.group_column and params.group_column not in df.columns:
        raise ValueError(f"group_column '{params.group_column}' not found. Available: {ctx.column_names}")

    base: dict = {
        "kind": kind,
        "x_column": params.x_column,
        "group_column": params.group_column,
        "title": f"{kind.replace('_', ' ').title()} of {params.x_column}",
    }

    if kind == "histogram":
        series = df[params.x_column].dropna().astype(float)
        if params.group_column:
            groups = {}
            for grp, sub in df.groupby(params.group_column):
                vals = sub[params.x_column].dropna().astype(float)
                counts, edges = np.histogram(vals, bins=params.n_bins)
                groups[str(grp)] = {
                    "bin_edges": [round(e, 4) for e in edges.tolist()],
                    "counts": counts.tolist(),
                }
            base["series"] = groups
        else:
            counts, edges = np.histogram(series, bins=params.n_bins)
            base["bin_edges"] = [round(e, 4) for e in edges.tolist()]
            base["counts"] = counts.tolist()
            base["x_label"] = params.x_column
            base["y_label"] = "Count"

    elif kind == "boxplot":
        def _box(s: pd.Series) -> dict:
            s = s.dropna().astype(float)
            q1, q2, q3 = float(s.quantile(0.25)), float(s.quantile(0.5)), float(s.quantile(0.75))
            iqr = q3 - q1
            whisker_lo = float(s[s >= q1 - 1.5 * iqr].min())
            whisker_hi = float(s[s <= q3 + 1.5 * iqr].max())
            outliers = s[(s < whisker_lo) | (s > whisker_hi)].round(4).tolist()
            return {
                "q1": round(q1, 4), "median": round(q2, 4), "q3": round(q3, 4),
                "whisker_low": round(whisker_lo, 4), "whisker_high": round(whisker_hi, 4),
                "outliers": outliers[:50],
                "n": int(len(s)),
            }

        if params.group_column:
            base["series"] = {
                str(grp): _box(sub[params.x_column])
                for grp, sub in df.groupby(params.group_column)
            }
        else:
            base.update(_box(df[params.x_column]))

    elif kind == "kde":
        from scipy.stats import gaussian_kde
        series = df[params.x_column].dropna().astype(float)
        if params.group_column:
            base["series"] = {}
            for grp, sub in df.groupby(params.group_column):
                vals = sub[params.x_column].dropna().astype(float)
                if len(vals) >= 5:
                    kde = gaussian_kde(vals)
                    xs = np.linspace(float(vals.min()), float(vals.max()), 100)
                    base["series"][str(grp)] = {
                        "x": [round(v, 4) for v in xs.tolist()],
                        "y": [round(v, 6) for v in kde(xs).tolist()],
                    }
        else:
            kde = gaussian_kde(series)
            xs = np.linspace(float(series.min()), float(series.max()), 100)
            base["x"] = [round(v, 4) for v in xs.tolist()]
            base["y"] = [round(v, 6) for v in kde(xs).tolist()]

    elif kind == "bar":
        vc = df[params.x_column].value_counts().head(30)
        base["categories"] = [str(k) for k in vc.index.tolist()]
        base["counts"] = vc.tolist()
        if params.group_column:
            base["series"] = {}
            for grp, sub in df.groupby(params.group_column):
                sub_vc = sub[params.x_column].value_counts()
                base["series"][str(grp)] = {str(k): int(v) for k, v in sub_vc.items()}

    elif kind == "scatter":
        if not params.y_column:
            raise ValueError("y_column is required for scatter plot.")
        sub = df[[params.x_column, params.y_column]].dropna()
        if len(sub) > 500:
            sub = sub.sample(n=500, random_state=42)
        base["x"] = sub[params.x_column].round(4).tolist()
        base["y"] = sub[params.y_column].round(4).tolist()
        base["y_column"] = params.y_column
        base["n_shown"] = len(sub)

    elif kind == "km_curve":
        if not params.y_column:
            raise ValueError("y_column (event indicator) is required for km_curve.")
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            raise RuntimeError("lifelines is required for km_curve. Run: poetry add lifelines")

        T_col, E_col = params.x_column, params.y_column
        tmp = df[[T_col, E_col]].dropna().copy()
        T_numeric = pd.to_numeric(tmp[T_col], errors="coerce")
        tmp = tmp[T_numeric > 0].copy()
        if params.event_codes:
            tmp["_ev"] = tmp[E_col].isin(params.event_codes).astype(int)
        else:
            tmp["_ev"] = pd.to_numeric(tmp[E_col], errors="coerce").fillna(0).astype(int)

        def _km_data(T_s: pd.Series, E_s: pd.Series, label: str) -> dict:
            kmf = KaplanMeierFitter()
            kmf.fit(T_s, event_observed=E_s, label=label)
            sf = kmf.survival_function_.reset_index()
            return {
                "label": label,
                "timeline": sf.iloc[:, 0].round(2).tolist(),
                "survival": sf.iloc[:, 1].round(4).tolist(),
                "median_survival": (
                    None if pd.isna(kmf.median_survival_time_)
                    else round(float(kmf.median_survival_time_), 2)
                ),
                "n_events": int(E_s.sum()),
                "n_total": int(len(T_s)),
            }

        if params.group_column:
            if params.group_column not in df.columns:
                raise ValueError(f"group_column '{params.group_column}' not found.")
            # Re-attach group column to tmp for groupby
            tmp_with_grp = tmp.join(df[[params.group_column]], how="left")
            base["series"] = []
            for grp, sub_tmp in tmp_with_grp.groupby(params.group_column):
                if len(sub_tmp) >= 5:
                    base["series"].append(
                        _km_data(
                            pd.to_numeric(sub_tmp[T_col], errors="coerce"),
                            sub_tmp["_ev"],
                            str(grp),
                        )
                    )
        else:
            T_s = pd.to_numeric(tmp[T_col], errors="coerce")
            base.update(_km_data(T_s, tmp["_ev"], "Overall"))

    elif kind == "correlation_heatmap":
        cols = [params.x_column]
        if params.y_column:
            cols += [params.y_column]
        numeric = df.select_dtypes(include="number")
        if len(cols) == 1:
            numeric = numeric.iloc[:, :15]
        corr = numeric.corr(method="pearson").round(4)
        base["columns"] = list(corr.columns)
        base["matrix"] = corr.where(pd.notnull(corr), None).to_dict()

    else:
        raise ValueError(
            f"Unknown plot kind '{kind}'. Choose from: "
            "histogram, boxplot, kde, bar, scatter, km_curve, correlation_heatmap."
        )

    return base
