"""
Input: DataFrame with outcome, time, treatment, optional subject ID and covariates
Output: MMRM fixed effects, interaction term, AIC/BIC
Purpose: Account for within-subject correlation in longitudinal data. FDA Guidance (2021)
  recommends MMRM as primary analysis for trials with repeated measurements. Simple t-tests
  are invalid for longitudinal data because they assume independence of observations.
Reference: FDA Guidance for Industry (2021) — Adjusting for Covariates in Randomized
  Clinical Trials; Mallinckrodt et al. (2008) MMRM in clinical trials.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class MMRMParams(BaseModel):
    outcome_col: str = Field(description="Numeric outcome / dependent variable column.")
    time_col: str = Field(
        description=(
            "Column representing time / visit (numeric or categorical). "
            "Encode as category prior if the values are labels like 'Week 0', 'Week 4'."
        )
    )
    treatment_col: str = Field(
        description="Column containing treatment group assignment (binary or multi-level)."
    )
    subject_col: Optional[str] = Field(
        default=None,
        description=(
            "Column identifying the subject / participant. "
            "If null, the DataFrame row index is used as subject ID."
        ),
    )
    covariates: Optional[list[str]] = Field(
        default=None,
        description="Additional baseline covariates to include as fixed effects.",
    )


def _mmrm_logic(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    treatment_col: str,
    subject_col: Optional[str] = None,
    covariates: Optional[list[str]] = None,
) -> dict:
    """Core MMRM logic — callable directly with a DataFrame for programmatic use."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise RuntimeError(
            "statsmodels is required for MMRM. Run: pip install statsmodels"
        )

    df_work = df.copy()

    # Use index as subject if no subject column provided
    if subject_col is None:
        df_work = df_work.reset_index(drop=False)
        subject_col = "_subject_id"
        df_work[subject_col] = df_work.index.astype(str)
        subject_was_none = True
    else:
        subject_was_none = False

    # Build formula
    formula_parts = [f"{outcome_col} ~ {time_col} * {treatment_col}"]
    if covariates:
        for cov in covariates:
            formula_parts.append(f" + {cov}")
    formula = "".join(formula_parts)

    # Listwise deletion
    required_cols = [outcome_col, time_col, treatment_col, subject_col]
    if covariates:
        required_cols += covariates
    required_cols = list(dict.fromkeys(required_cols))  # deduplicate preserving order

    n_before = len(df_work)
    df_clean = df_work[required_cols].dropna().copy()
    n_dropped = n_before - len(df_clean)

    if len(df_clean) < 4:
        return {
            "error": "Insufficient observations after listwise deletion",
            "fallback": "use hypothesis_test for unpaired group comparison",
        }

    n_subjects = int(df_clean[subject_col].nunique())
    n_observations = len(df_clean)

    try:
        model = smf.mixedlm(
            formula,
            data=df_clean,
            groups=df_clean[subject_col],
        )
        result = model.fit(reml=True, disp=False)
    except Exception as exc:
        return {
            "error": f"MMRM failed to converge: {exc}",
            "fallback": "use hypothesis_test for unpaired group comparison",
        }

    # Extract fixed effects table
    fe_params = result.params
    fe_pvalues = result.pvalues
    fe_bse = result.bse
    fe_tvalues = result.tvalues if hasattr(result, "tvalues") else (fe_params / fe_bse)

    fixed_effects = []
    treatment_time_interaction: dict = {}

    for term in fe_params.index:
        if term == "Group Var":
            continue  # skip random effects variance
        coef = float(fe_params[term])
        se = float(fe_bse[term]) if term in fe_bse.index else float("nan")
        z = float(fe_tvalues[term]) if term in fe_tvalues.index else (coef / se if se != 0 else float("nan"))
        pval = float(fe_pvalues[term]) if term in fe_pvalues.index else float("nan")
        sig = bool(pval < 0.05)

        entry = {
            "term": term,
            "coefficient": round(coef, 4),
            "se": round(se, 4),
            "z": round(z, 4),
            "p_value": round(pval, 6),
            "significant": sig,
        }
        fixed_effects.append(entry)

        # Identify treatment × time interaction (any term containing ':')
        if ":" in term and not treatment_time_interaction:
            treatment_time_interaction = {
                "p_value": round(pval, 6),
                "significant": sig,
                "coefficient": round(coef, 4),
            }

    # AIC and BIC
    aic = float(result.aic) if hasattr(result, "aic") else float("nan")
    bic = float(result.bic) if hasattr(result, "bic") else float("nan")

    # Clinical interpretation
    if treatment_time_interaction:
        ix_p = treatment_time_interaction["p_value"]
        if treatment_time_interaction["significant"]:
            interpretation = (
                f"Treatment effect changes significantly over time (p={ix_p:.3f}), "
                "indicating a differential treatment trajectory."
            )
        else:
            interpretation = (
                f"No significant treatment × time interaction detected (p={ix_p:.3f}); "
                "treatment effect is consistent across time points."
            )
    else:
        interpretation = "No interaction term found in model output."

    return {
        "model": "MMRM (Mixed Model for Repeated Measures)",
        "formula": formula,
        "n_subjects": n_subjects,
        "n_observations": n_observations,
        "n_dropped_listwise": n_dropped,
        "fixed_effects": fixed_effects,
        "treatment_time_interaction": treatment_time_interaction,
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "reml": True,
        "clinical_interpretation": interpretation,
    }


@tool(
    name="mixed_model_repeated_measures",
    description=(
        "Fit a Mixed Model for Repeated Measures (MMRM) for longitudinal clinical trial data. "
        "Models the within-subject correlation across time points using a random intercept. "
        "Extracts fixed effects, treatment × time interaction (key regulatory endpoint), "
        "and AIC/BIC. FDA (2021) recommends MMRM as the primary analysis method for trials "
        "with repeated measurements. Use when data has a time/visit column and subject IDs."
    ),
    params_model=MMRMParams,
    category="stats",
)
def mixed_model_repeated_measures(params: MMRMParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in [params.outcome_col, params.time_col, params.treatment_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")
    if params.subject_col and params.subject_col not in df.columns:
        raise ValueError(
            f"subject_col '{params.subject_col}' not found. Available: {ctx.column_names}"
        )
    if params.covariates:
        missing = [c for c in params.covariates if c not in df.columns]
        if missing:
            raise ValueError(f"Covariate columns not found: {missing}")

    return _mmrm_logic(
        df,
        params.outcome_col,
        params.time_col,
        params.treatment_col,
        params.subject_col,
        params.covariates,
    )
