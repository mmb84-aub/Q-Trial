"""
Input: DataFrame with outcome, time, treatment, optional subject ID
Output: cLDA fixed effects, treatment-by-time estimates, model fit statistics
Purpose: Constrained Longitudinal Data Analysis — models ALL time points
  (including baseline) as repeated measures.  The treatment main effect is
  omitted from the model formula, which constrains the treatment groups to
  have equal expected means at the baseline (reference) time point.
  cLDA is recommended when baseline values should be treated as part of the
  longitudinal process rather than as a covariate (as in ANCOVA/MMRM).
Reference: Liang & Zeger (2000); Liu et al. (2009) cLDA for clinical trials
  with longitudinal outcomes. Statistics in Medicine, 28(5), 747–763.
Limitation: Uses compound-symmetry (random-intercept) covariance structure
  via statsmodels.mixedlm.  True cLDA literature recommends unstructured
  covariance (available in SAS PROC MIXED, R nlme/mmrm).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _clda_logic(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    treatment_col: str,
    subject_col: Optional[str] = None,
) -> dict:
    """Constrained Longitudinal Data Analysis (cLDA).

    Key difference from standard MMRM
    ----------------------------------
    MMRM:  ``outcome ~ time * treatment``  (treatment main effect included)
    cLDA:  ``outcome ~ C(time) + C(time):C(treatment)``  (NO treatment main
           effect — constrains baseline treatment difference to zero)

    Parameters
    ----------
    df : DataFrame in long format (one row per subject per time point).
    outcome_col : numeric outcome column.
    time_col : time / visit column (treated as categorical factor).
    treatment_col : treatment-arm column.
    subject_col : subject ID; if *None*, row index is used.

    Returns
    -------
    dict with model results, treatment-by-time effects, and limitations.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise RuntimeError(
            "statsmodels is required for cLDA.  Run: pip install statsmodels"
        )

    df_work = df.copy()

    # Handle subject column ---------------------------------------------------
    if subject_col is None:
        df_work = df_work.reset_index(drop=False)
        subject_col = "_subject_id"
        df_work[subject_col] = df_work.index.astype(str)

    # Validate required columns -----------------------------------------------
    required_cols = [outcome_col, time_col, treatment_col, subject_col]
    required_cols = list(dict.fromkeys(required_cols))  # deduplicate

    n_before = len(df_work)
    df_clean = df_work[required_cols].dropna().copy()
    n_dropped = n_before - len(df_clean)

    if len(df_clean) < 6:
        return {
            "error": "Insufficient observations after listwise deletion for cLDA",
            "n_remaining": len(df_clean),
        }

    n_subjects = int(df_clean[subject_col].nunique())
    n_observations = len(df_clean)
    n_timepoints = int(df_clean[time_col].nunique())

    if n_timepoints < 2:
        return {
            "error": "cLDA requires at least 2 time points (including baseline)",
            "n_timepoints": n_timepoints,
        }

    # Determine baseline: smallest numeric value or first sorted level --------
    time_vals = df_clean[time_col].unique()
    if pd.api.types.is_numeric_dtype(df_clean[time_col]):
        baseline_val = min(time_vals)
    else:
        baseline_val = sorted(time_vals, key=str)[0]

    # Build cLDA formula ------------------------------------------------------
    # outcome ~ C(time, Treatment(ref=baseline)) +
    #           C(time, Treatment(ref=baseline)):C(treatment)
    #
    # No treatment main effect → treatment effect at baseline is exactly 0.
    time_ref = f"Treatment(reference={repr(baseline_val)})"
    formula = (
        f"{outcome_col} ~ "
        f"C({time_col}, {time_ref}) + "
        f"C({time_col}, {time_ref}):C({treatment_col})"
    )

    # Fit mixed-effects model (random intercept per subject) ------------------
    try:
        model = smf.mixedlm(
            formula,
            data=df_clean,
            groups=df_clean[subject_col],
        )
        result = model.fit(reml=True, disp=False)
    except Exception as exc:
        return {
            "error": f"cLDA model failed to converge: {exc}",
            "formula": formula,
        }

    # Extract fixed effects ---------------------------------------------------
    fe_params = result.params
    fe_pvalues = result.pvalues
    fe_bse = result.bse

    fixed_effects: list[dict] = []
    treatment_by_time: list[dict] = []

    for term in fe_params.index:
        if term == "Group Var":
            continue
        coef = float(fe_params[term])
        se = float(fe_bse[term]) if term in fe_bse.index else float("nan")
        pval = float(fe_pvalues[term]) if term in fe_pvalues.index else float("nan")

        entry = {
            "term": term,
            "coefficient": round(coef, 4),
            "se": round(se, 4),
            "p_value": round(pval, 6),
            "significant": bool(pval < 0.05),
        }
        fixed_effects.append(entry)

        # Interaction terms (treatment-by-time) contain ":"
        if ":" in term:
            treatment_by_time.append(entry)

    # Top-level p_value: smallest treatment-by-time p (primary interest) ------
    top_p: float | None = None
    if treatment_by_time:
        top_p = min(t["p_value"] for t in treatment_by_time)
    # Fallback: any treatment-related term
    if top_p is None:
        for fe in fixed_effects:
            if treatment_col.lower() in fe["term"].lower():
                top_p = fe["p_value"]
                break

    # Model fit ---------------------------------------------------------------
    aic = float(result.aic) if hasattr(result, "aic") else float("nan")
    bic = float(result.bic) if hasattr(result, "bic") else float("nan")

    # Clinical interpretation -------------------------------------------------
    if treatment_by_time:
        sig_times = [t for t in treatment_by_time if t["significant"]]
        if sig_times:
            interpretation = (
                f"cLDA detected significant treatment effects at {len(sig_times)} of "
                f"{len(treatment_by_time)} post-baseline time point(s). "
                "Treatment groups are constrained to equal means at baseline."
            )
        else:
            interpretation = (
                f"cLDA: no significant treatment-by-time effects at any of "
                f"{len(treatment_by_time)} post-baseline time point(s) (all p >= 0.05). "
                "Treatment groups are constrained to equal means at baseline."
            )
    else:
        interpretation = "No treatment-by-time interaction terms found in model output."

    return {
        "model": "cLDA (constrained Longitudinal Data Analysis)",
        "formula": formula,
        "baseline_time": str(baseline_val),
        "n_subjects": n_subjects,
        "n_observations": n_observations,
        "n_timepoints": n_timepoints,
        "n_dropped_listwise": n_dropped,
        "fixed_effects": fixed_effects,
        "treatment_by_time_effects": treatment_by_time,
        "p_value": top_p,
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "reml": True,
        "clinical_interpretation": interpretation,
        "limitations": [
            "Uses compound-symmetry (random-intercept) covariance structure via "
            "statsmodels.mixedlm. True cLDA literature recommends unstructured "
            "covariance (available in SAS PROC MIXED, R nlme/mmrm package).",
            "Baseline constraint is imposed by omitting the treatment main effect "
            "from the model formula (standard parameterisation-based approach).",
        ],
    }
