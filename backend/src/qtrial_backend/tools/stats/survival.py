from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class SurvivalParams(BaseModel):
    time_column: str = Field(description="Column with follow-up time (numeric)")
    event_column: str = Field(
        description=(
            "Column indicating whether the event occurred (1/True = event, "
            "0/False = censored). For datasets with multiple event codes "
            "(e.g. 0=censored, 1=transplant, 2=death) specify the event_codes "
            "parameter to select which codes count as events."
        )
    )
    group_column: Optional[str] = Field(
        default=None,
        description="Optional categorical column to produce per-group KM curves. "
        "If provided, a log-rank test p-value is also returned.",
    )
    event_codes: Optional[list[int]] = Field(
        default=None,
        description=(
            "When the event column has multiple codes, list the integer codes "
            "that should be treated as events (e.g. [1, 2] for transplant and death). "
            "All other values are treated as censored."
        ),
    )
    timepoints: list[float] = Field(
        default=[365, 730, 1095, 1825],
        description="Time points (same unit as time_column) at which to report survival probability.",
    )


def _km_summary(T: pd.Series, E: pd.Series) -> dict:
    """Compute a Kaplan-Meier summary without lifelines for resilience."""
    try:
        from lifelines import KaplanMeierFitter  # type: ignore[import]

        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E)
        median_surv = kmf.median_survival_time_
        return {"median_survival": None if pd.isna(median_surv) else round(float(median_surv), 2), "_kmf": kmf}
    except ImportError:
        # Fallback: manual KM
        return {"median_survival": None, "_kmf": None}


def _survival_at(kmf, t: float) -> float | None:  # type: ignore[no-untyped-def]
    try:
        val = kmf.survival_function_at_times(t).iloc[0]
        return None if pd.isna(val) else round(float(val), 4)
    except Exception:
        return None


@tool(
    name="survival_analysis",
    description=(
        "Kaplan-Meier survival analysis. Returns median survival time and "
        "survival probabilities at specified timepoints. If group_column is "
        "provided, computes per-group curves and a log-rank test p-value. "
        "Handles multi-code event columns via event_codes parameter."
    ),
    params_model=SurvivalParams,
    category="stats",
)
def survival_analysis(params: SurvivalParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe.copy()
    n_before = len(df)

    for col in (params.time_column, params.event_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")
    if params.group_column and params.group_column not in df.columns:
        raise ValueError(f"Column '{params.group_column}' not found. Available: {ctx.column_names}")

    # Build event indicator
    if params.event_codes:
        df["_event"] = df[params.event_column].isin(params.event_codes).astype(int)
    else:
        df["_event"] = pd.to_numeric(df[params.event_column], errors="coerce").fillna(0).astype(int)

    # Track rows dropped by listwise deletion
    df = df[[params.time_column, "_event"] + ([params.group_column] if params.group_column else [])].dropna()
    rows_dropped_na = n_before - len(df)

    T = pd.to_numeric(df[params.time_column], errors="coerce")
    E = df["_event"]
    n_after_dropna = len(df)
    df = df[T > 0]
    rows_dropped_nonpositive = n_after_dropna - len(df)
    T = pd.to_numeric(df[params.time_column], errors="coerce")
    E = df["_event"]

    result: dict = {
        "time_column": params.time_column,
        "event_column": params.event_column,
        "n_total": int(len(T)),
        "n_before_dropna": n_before,
        "rows_dropped": rows_dropped_na + rows_dropped_nonpositive,
        "n_events": int(E.sum()),
        "n_censored": int((E == 0).sum()),
        "event_rate_pct": round(float(E.mean() * 100), 2),
        "listwise_deletion": {
            "rows_dropped_missing": rows_dropped_na,
            "rows_dropped_nonpositive_time": rows_dropped_nonpositive,
            "total_rows_dropped": rows_dropped_na + rows_dropped_nonpositive,
            "note": "Rows with missing time/event or non-positive time values were excluded.",
        },
    }

    # Overall KM
    summary = _km_summary(T, E)
    result["overall"] = {
        "median_survival": summary["median_survival"],
        "survival_at_timepoints": {},
    }
    if summary["_kmf"] is not None:
        for tp in params.timepoints:
            result["overall"]["survival_at_timepoints"][str(int(tp))] = _survival_at(summary["_kmf"], tp)

    # Per-group KM + log-rank
    if params.group_column:
        try:
            from lifelines.statistics import logrank_test, multivariate_logrank_test  # type: ignore[import]
        except ImportError:
            result["groups"] = {}
            result["logrank_p_value"] = None
            return result

        groups: dict = {}
        labels = sorted(df[params.group_column].dropna().unique())
        for label in labels:
            mask = df[params.group_column] == label
            Tg, Eg = T[mask], E[mask]
            gs = _km_summary(Tg, Eg)
            entry: dict = {
                "n": int(mask.sum()),
                "n_events": int(Eg.sum()),
                "median_survival": gs["median_survival"],
                "survival_at_timepoints": {},
            }
            if gs["_kmf"] is not None:
                for tp in params.timepoints:
                    entry["survival_at_timepoints"][str(int(tp))] = _survival_at(gs["_kmf"], tp)
            groups[str(label)] = entry

        result["groups"] = groups

        # Log-rank test
        try:
            mlr = multivariate_logrank_test(T, df[params.group_column], E)
            result["logrank_p_value"] = round(float(mlr.p_value), 6)
            result["logrank_significant"] = bool(mlr.p_value < 0.05)
        except Exception as exc:
            result["logrank_p_value"] = None
            result["logrank_error"] = str(exc)

    return result
