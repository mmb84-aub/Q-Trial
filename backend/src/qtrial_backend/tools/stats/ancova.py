"""
Input: DataFrame, outcome column, treatment column, list of baseline covariates
Output: adjusted treatment effect, marginal means, covariate coefficients
Purpose: Control for prognostic baseline variables to increase precision and reduce
  confounding. ICH E9 guideline recommends ANCOVA for primary analysis in RCTs when
  baseline measurements of the outcome are available.
Reference: ICH E9 Statistical Principles for Clinical Trials (1998);
  Senn (2006) Change from baseline and analysis of covariance revisited.
  Statistics in Medicine, 25(24), 4334–4344.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class ANCOVAParams(BaseModel):
    outcome_col: str = Field(description="Numeric outcome (dependent variable) column.")
    treatment_col: str = Field(
        description="Column containing treatment group assignment (binary or multi-level)."
    )
    covariates: list[str] = Field(
        description=(
            "Baseline variables to adjust for (e.g. ['age', 'baseline_score']). "
            "Must be numeric."
        )
    )
    interaction: bool = Field(
        default=False,
        description=(
            "If True, add treatment × covariate interaction terms. "
            "Tests whether the treatment effect is homogeneous across covariate levels."
        ),
    )


def _ancova_logic(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariates: list[str],
    interaction: bool = False,
) -> dict:
    """Core ANCOVA logic — callable directly with a DataFrame for programmatic use."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise RuntimeError(
            "statsmodels is required for ANCOVA. Run: pip install statsmodels"
        )

    # Build OLS formula with HC3 robust SE
    cov_terms = " + ".join(covariates)
    formula = f"{outcome_col} ~ C({treatment_col}) + {cov_terms}"

    if interaction:
        interaction_terms = " + ".join(
            f"C({treatment_col}):{cov}" for cov in covariates
        )
        formula = f"{formula} + {interaction_terms}"

    required_cols = [outcome_col, treatment_col] + covariates
    n_before = len(df)
    df_clean = df[required_cols].dropna().copy()
    n_dropped = n_before - len(df_clean)

    if len(df_clean) < max(4, len(covariates) + 2):
        raise ValueError(
            f"Insufficient observations after listwise deletion: {len(df_clean)} rows."
        )

    model = smf.ols(formula, data=df_clean)
    result = model.fit(cov_type="HC3")

    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    conf = result.conf_int()

    # Identify treatment coefficient(s) — those starting with "C({treatment_col})"
    treatment_prefix = f"C({treatment_col})"
    treatment_terms = [t for t in params.index if t.startswith(treatment_prefix) and ":" not in t]

    if not treatment_terms:
        raise ValueError(
            f"No treatment coefficient found for '{treatment_col}' in model output. "
            "Check that the treatment column has at least two unique values."
        )

    # Report the first treatment coefficient (vs. reference level)
    trt_term = treatment_terms[0]
    adj_treatment = {
        "term": trt_term,
        "coefficient": round(float(params[trt_term]), 4),
        "se": round(float(bse[trt_term]), 4),
        "p_value": round(float(pvalues[trt_term]), 6),
        "ci_lower": round(float(conf.loc[trt_term, 0]), 4),
        "ci_upper": round(float(conf.loc[trt_term, 1]), 4),
        "significant": bool(pvalues[trt_term] < 0.05),
    }

    # Marginal (adjusted) means per treatment group
    cov_means = {cov: float(df_clean[cov].mean()) for cov in covariates}
    treatment_groups = df_clean[treatment_col].unique().tolist()
    marginal_means: list[dict] = []
    for group in treatment_groups:
        pred_data = pd.DataFrame(
            {treatment_col: [group], **{c: [cov_means[c]] for c in covariates}}
        )
        try:
            pred = result.get_prediction(pred_data)
            adj_mean = float(pred.predicted_mean.iloc[0])
            adj_se = float(pred.se_mean.iloc[0])
        except Exception:
            adj_mean = float(result.predict(pred_data).iloc[0])
            adj_se = float("nan")
        marginal_means.append(
            {
                "group": str(group),
                "adjusted_mean": round(adj_mean, 4),
                "se": round(adj_se, 4) if not np.isnan(adj_se) else None,
            }
        )

    # Covariate coefficients
    cov_results = []
    for cov in covariates:
        if cov in params.index:
            cov_results.append(
                {
                    "name": cov,
                    "coefficient": round(float(params[cov]), 4),
                    "p_value": round(float(pvalues[cov]), 6),
                }
            )

    return {
        "model": "ANCOVA",
        "formula": formula,
        "adjusted_treatment_effect": adj_treatment,
        "marginal_means": marginal_means,
        "covariates": cov_results,
        "r_squared": round(float(result.rsquared), 4),
        "adj_r_squared": round(float(result.rsquared_adj), 4),
        "f_statistic": round(float(result.fvalue), 4),
        "model_p_value": round(float(result.f_pvalue), 6),
        "n_observations": int(len(df_clean)),
        "n_dropped_listwise": int(n_dropped),
    }


@tool(
    name="ancova",
    description=(
        "Analysis of Covariance (ANCOVA): adjust the treatment effect for baseline covariates. "
        "Fits OLS with HC3 robust standard errors. "
        "Returns the adjusted treatment coefficient, marginal (least-squares) means per group, "
        "covariate effects, R², and F-statistic. "
        "ICH E9 recommends ANCOVA as the primary analysis when baseline outcome measurements "
        "are available in an RCT."
    ),
    params_model=ANCOVAParams,
    category="stats",
)
def ancova(params: ANCOVAParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in [params.outcome_col, params.treatment_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")
    missing_covs = [c for c in params.covariates if c not in df.columns]
    if missing_covs:
        raise ValueError(
            f"Covariate columns not found: {missing_covs}. Available: {ctx.column_names}"
        )
    return _ancova_logic(
        df,
        params.outcome_col,
        params.treatment_col,
        params.covariates,
        params.interaction,
    )
