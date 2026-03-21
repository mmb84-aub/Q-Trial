from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class RegressionParams(BaseModel):
    model_type: str = Field(
        description=(
            "Type of regression: "
            "'linear' (OLS with HC3 robust SE), "
            "'logistic' (binary outcome, returns ORs), or "
            "'cox' (time-to-event, returns HRs)."
        )
    )
    outcome_column: str = Field(
        description=(
            "Outcome/dependent variable column. "
            "For cox: the event/status indicator column."
        )
    )
    predictor_columns: list[str] = Field(
        description="List of predictor/covariate column names."
    )
    time_column: Optional[str] = Field(
        default=None,
        description="For cox only: column containing follow-up time.",
    )
    event_codes: Optional[list[int]] = Field(
        default=None,
        description=(
            "For cox only: integer codes in outcome_column that count as events "
            "(e.g. [1, 2] for transplant and death). All others = censored."
        ),
    )


def _fit_linear(outcome: str, predictors: list[str], df: pd.DataFrame, ctx: AgentContext) -> dict:
    try:
        import statsmodels.api as sm
    except ImportError:
        raise RuntimeError(
            "statsmodels is required for linear regression. Run: poetry add statsmodels"
        )

    # Track rows before and after listwise deletion
    n_before = len(df)
    sub = df[[outcome] + predictors].dropna()
    rows_dropped = n_before - len(sub)

    if len(sub) < max(len(predictors) + 2, 10):
        raise ValueError(f"Too few complete observations: {len(sub)}")

    X = pd.get_dummies(sub[predictors], drop_first=True).astype(float)
    X_c = sm.add_constant(X)
    y = pd.to_numeric(sub[outcome], errors="coerce")

    model = sm.OLS(y, X_c).fit(cov_type="HC3")
    ci = model.conf_int(alpha=0.05)

    coefs = []
    for term in model.params.index:
        coefs.append(
            {
                "term": str(term),
                "coef": round(float(model.params[term]), 6),
                "ci_lower": round(float(ci.loc[term, 0]), 6),
                "ci_upper": round(float(ci.loc[term, 1]), 6),
                "std_err": round(float(model.bse[term]), 6),
                "p_value": round(float(model.pvalues[term]), 6),
            }
        )

    return {
        "model_type": "linear",
        "outcome": outcome,
        "predictors": list(X.columns.tolist()),
        "n_observations": int(len(sub)),
        "n_before_dropna": n_before,
        "rows_dropped": rows_dropped,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "aic": round(float(model.aic), 4),
        "bic": round(float(model.bic), 4),
        "f_statistic_p": round(float(model.f_pvalue), 6),
        "coefficients": coefs,
        "listwise_deletion": {
            "rows_dropped": rows_dropped,
            "note": "Rows with missing values in outcome or predictors were excluded.",
        },
    }


def _fit_logistic(outcome: str, predictors: list[str], df: pd.DataFrame, ctx: AgentContext) -> dict:
    try:
        import statsmodels.api as sm
    except ImportError:
        raise RuntimeError(
            "statsmodels is required for logistic regression. Run: poetry add statsmodels"
        )

    # Track rows before and after listwise deletion
    n_before = len(df)
    sub = df[[outcome] + predictors].dropna()
    rows_dropped = n_before - len(sub)

    if len(sub) < max(len(predictors) + 2, 10):
        raise ValueError(f"Too few complete observations: {len(sub)}")

    X = pd.get_dummies(sub[predictors], drop_first=True).astype(float)
    X_c = sm.add_constant(X)
    y = pd.to_numeric(sub[outcome], errors="coerce")

    model = sm.Logit(y, X_c).fit(disp=0)
    ci = model.conf_int(alpha=0.05)

    coefs = []
    for term in model.params.index:
        lo = float(np.exp(ci.loc[term, 0]))
        hi = float(np.exp(ci.loc[term, 1]))
        coefs.append(
            {
                "term": str(term),
                "log_odds": round(float(model.params[term]), 6),
                "odds_ratio": round(float(np.exp(model.params[term])), 6),
                "or_ci_lower": round(lo, 6),
                "or_ci_upper": round(hi, 6),
                "p_value": round(float(model.pvalues[term]), 6),
            }
        )

    return {
        "model_type": "logistic",
        "outcome": outcome,
        "predictors": list(X.columns.tolist()),
        "n_observations": int(len(sub)),
        "n_before_dropna": n_before,
        "rows_dropped": rows_dropped,
        "pseudo_r2_mcfadden": round(float(model.prsquared), 4),
        "aic": round(float(model.aic), 4),
        "bic": round(float(model.bic), 4),
        "coefficients": coefs,
        "note": "Odds ratio > 1 means higher odds of outcome for higher predictor values.",
        "listwise_deletion": {
            "rows_dropped": rows_dropped,
            "note": "Rows with missing values in outcome or predictors were excluded.",
        },
    }


def _fit_cox(params: RegressionParams, df: pd.DataFrame, ctx: AgentContext) -> dict:
    from lifelines import CoxPHFitter

    if not params.time_column:
        raise ValueError("time_column is required for Cox PH regression.")

    required = [params.time_column, params.outcome_column] + params.predictor_columns
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {ctx.column_names}")

    # Track rows before and after listwise deletion
    n_before = len(df)
    sub = df[required].dropna().copy()
    rows_dropped_na = n_before - len(sub)

    if params.event_codes:
        sub["_event"] = sub[params.outcome_column].isin(params.event_codes).astype(int)
    else:
        sub["_event"] = pd.to_numeric(sub[params.outcome_column], errors="coerce").fillna(0).astype(int)

    T = pd.to_numeric(sub[params.time_column], errors="coerce")
    n_after_dropna = len(sub)
    sub = sub[T > 0].copy()
    rows_dropped_nonpositive = n_after_dropna - len(sub)
    sub["_event"] = sub["_event"].values  # ensure no index misalignment after filter

    cox_df = sub[[params.time_column, "_event"] + params.predictor_columns].rename(
        columns={"_event": "event"}
    )

    # Dummy-encode any categorical predictors
    non_time_event = [c for c in cox_df.columns if c not in (params.time_column, "event")]
    cox_df = pd.get_dummies(cox_df, columns=[
        c for c in non_time_event if cox_df[c].dtype == object or str(cox_df[c].dtype) == "category"
    ], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col=params.time_column, event_col="event")

    summary = cph.summary
    coefs = []
    for cov in summary.index:
        entry: dict = {"covariate": str(cov)}
        for raw_col, key in [
            ("coef", "log_hr"),
            ("exp(coef)", "hazard_ratio"),
            ("se(coef)", "std_err"),
            ("z", "z_stat"),
            ("p", "p_value"),
            ("exp(coef) lower 95%", "hr_ci_lower"),
            ("exp(coef) upper 95%", "hr_ci_upper"),
        ]:
            if raw_col in summary.columns:
                val = summary.loc[cov, raw_col]
                entry[key] = round(float(val), 6) if pd.notna(val) else None
        coefs.append(entry)

    return {
        "model_type": "cox",
        "time_column": params.time_column,
        "event_column": params.outcome_column,
        "predictors": params.predictor_columns,
        "n_observations": int(len(sub)),
        "n_before_dropna": n_before,
        "rows_dropped": rows_dropped_na + rows_dropped_nonpositive,
        "n_events": int(sub["_event"].sum()),
        "concordance_index": round(float(cph.concordance_index_), 4),
        "log_likelihood": round(float(cph.log_likelihood_), 4),
        "coefficients": coefs,
        "note": "HR > 1 = higher hazard (worse survival) for higher predictor values.",
        "listwise_deletion": {
            "rows_dropped_missing": rows_dropped_na,
            "rows_dropped_nonpositive_time": rows_dropped_nonpositive,
            "total_rows_dropped": rows_dropped_na + rows_dropped_nonpositive,
            "note": "Rows with missing values or non-positive time were excluded.",
        },
    }


@tool(
    name="regression",
    description=(
        "Fit a regression model and return coefficients with confidence intervals and p-values. "
        "linear: OLS with HC3 heteroskedasticity-robust SE. Returns coefficients, CIs, R², AIC. "
        "logistic: Logistic regression. Returns log-odds, ORs, CIs, p-values, AIC, McFadden R². "
        "cox: Cox proportional hazards. Returns HRs, CIs, p-values, concordance index (C-statistic). "
        "Categorical predictors are automatically dummy-encoded."
    ),
    params_model=RegressionParams,
    category="stats",
)
def regression(params: RegressionParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe

    for col in [params.outcome_column] + params.predictor_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")

    mt = params.model_type.lower()
    if mt == "linear":
        return _fit_linear(params.outcome_column, params.predictor_columns, df, ctx)
    elif mt == "logistic":
        return _fit_logistic(params.outcome_column, params.predictor_columns, df, ctx)
    elif mt == "cox":
        return _fit_cox(params, df, ctx)
    else:
        raise ValueError(
            f"model_type must be 'linear', 'logistic', or 'cox'. Got: '{params.model_type}'"
        )
