"""
Input: DataFrame, outcome, treatment column, list of subgroup columns, outcome type
Output: effect sizes per subgroup level + interaction test per subgroup variable
Purpose: Assess whether treatment effect is consistent across patient subgroups.
  Significant interaction (p<0.05) indicates heterogeneous treatment effect — a key
  regulatory and clinical finding. Required by FDA for Phase III trial submissions.
Reference: FDA Guidance for Industry — Enrichment Strategies for Clinical Trials (2019);
  Rothwell (2005) Subgroup analysis in randomised controlled trials: importance,
  indications, and interpretation. The Lancet, 365(9454), 176–186.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class SubgroupParams(BaseModel):
    outcome_col: str = Field(description="Outcome / dependent variable column.")
    treatment_col: str = Field(
        description="Column containing treatment group assignment (binary)."
    )
    subgroup_cols: list[str] = Field(
        description=(
            "Columns defining patient subgroups "
            "(e.g. ['sex', 'age_group', 'disease_stage'])."
        )
    )
    outcome_type: str = Field(
        default="continuous",
        description=(
            "'continuous' (t-test + Cohen's d) or "
            "'binary' (chi-square + Odds Ratio)."
        ),
    )


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _cohen_d_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot_vals = [
        _cohen_d(
            rng.choice(a, size=len(a), replace=True),
            rng.choice(b, size=len(b), replace=True),
        )
        for _ in range(n_boot)
    ]
    boot_vals_clean = [v for v in boot_vals if not np.isnan(v)]
    if not boot_vals_clean:
        return float("nan"), float("nan")
    return float(np.percentile(boot_vals_clean, 2.5)), float(
        np.percentile(boot_vals_clean, 97.5)
    )


def _odds_ratio_ci(ct: np.ndarray) -> tuple[float, float, float]:
    """Return (OR, ci_lower, ci_upper) for 2×2 table using Woolf CI."""
    a, b, c, d = ct[0, 0], ct[0, 1], ct[1, 0], ct[1, 1]
    # Haldane-Anscombe correction for zero cells
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    if b == 0 or c == 0:
        return float("nan"), float("nan"), float("nan")
    or_val = (a * d) / (b * c)
    log_or = np.log(or_val)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lo = float(np.exp(log_or - 1.96 * se_log_or))
    ci_hi = float(np.exp(log_or + 1.96 * se_log_or))
    return float(or_val), ci_lo, ci_hi


def _interaction_p_continuous(
    df: pd.DataFrame, outcome_col: str, treatment_col: str, subgroup_col: str
) -> float:
    """Likelihood-ratio test for treatment × subgroup interaction in OLS."""
    try:
        import statsmodels.formula.api as smf

        formula_full = (
            f"{outcome_col} ~ C({treatment_col}) + C({subgroup_col}) "
            f"+ C({treatment_col}):C({subgroup_col})"
        )
        formula_reduced = (
            f"{outcome_col} ~ C({treatment_col}) + C({subgroup_col})"
        )
        df_c = df[[outcome_col, treatment_col, subgroup_col]].dropna()
        fit_full = smf.ols(formula_full, data=df_c).fit()
        fit_red = smf.ols(formula_reduced, data=df_c).fit()
        # F-test for additional terms
        from statsmodels.stats.anova import anova_lm  # type: ignore

        aov = anova_lm(fit_red, fit_full)
        # Second row (full model) has F and Pr(>F)
        p_val = float(aov["Pr(>F)"].iloc[1])
        return p_val
    except Exception:
        return float("nan")


def _interaction_p_binary(
    df: pd.DataFrame, outcome_col: str, treatment_col: str, subgroup_col: str
) -> float:
    """Likelihood-ratio test for treatment × subgroup interaction in logistic regression."""
    try:
        import statsmodels.formula.api as smf

        formula_full = (
            f"{outcome_col} ~ C({treatment_col}) + C({subgroup_col}) "
            f"+ C({treatment_col}):C({subgroup_col})"
        )
        formula_reduced = f"{outcome_col} ~ C({treatment_col}) + C({subgroup_col})"
        df_c = df[[outcome_col, treatment_col, subgroup_col]].dropna()
        fit_full = smf.logit(formula_full, data=df_c).fit(disp=False)
        fit_red = smf.logit(formula_reduced, data=df_c).fit(disp=False)
        lr_stat = 2 * (fit_full.llf - fit_red.llf)
        df_diff = fit_full.df_model - fit_red.df_model
        from scipy.stats import chi2 as chi2_dist  # type: ignore

        p_val = float(1 - chi2_dist.cdf(lr_stat, df=max(1, int(df_diff))))
        return p_val
    except Exception:
        return float("nan")


def _subgroup_logic(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    subgroup_cols: list[str],
    outcome_type: str = "continuous",
) -> dict:
    """Core subgroup analysis logic — callable directly with a DataFrame."""
    outcome_lower = outcome_type.lower()
    if outcome_lower not in ("continuous", "binary"):
        raise ValueError("outcome_type must be 'continuous' or 'binary'.")

    trt_vals = df[treatment_col].dropna().unique().tolist()
    if len(trt_vals) < 2:
        raise ValueError(
            f"Treatment column '{treatment_col}' needs at least 2 unique values."
        )
    # Use first two groups as treatment (index 1) and control (index 0)
    control_label = str(trt_vals[0])
    treatment_label = str(trt_vals[1])

    subgroup_results: list[dict] = []
    small_n_warning = False

    for sg_col in subgroup_cols:
        levels = df[sg_col].dropna().unique().tolist()
        level_results: list[dict] = []

        for level in levels:
            mask = df[sg_col].astype(str) == str(level)
            df_level = df[mask].copy()

            mask_trt = df_level[treatment_col].astype(str) == treatment_label
            mask_ctl = df_level[treatment_col].astype(str) == control_label

            df_trt = df_level[mask_trt][outcome_col].dropna().to_numpy(dtype=float)
            df_ctl = df_level[mask_ctl][outcome_col].dropna().to_numpy(dtype=float)

            n_trt = len(df_trt)
            n_ctl = len(df_ctl)

            if n_trt < 5 or n_ctl < 5:
                small_n_warning = True

            if outcome_lower == "continuous":
                if n_trt < 2 or n_ctl < 2:
                    continue
                _, p_value = stats.ttest_ind(df_trt, df_ctl, equal_var=False)
                effect = _cohen_d(df_trt, df_ctl)
                ci_lo, ci_hi = _cohen_d_bootstrap_ci(df_trt, df_ctl)

            else:  # binary
                if n_trt < 1 or n_ctl < 1:
                    continue
                ct = pd.crosstab(
                    df_level[treatment_col].astype(str),
                    df_level[outcome_col].astype(int),
                )
                if ct.shape != (2, 2):
                    continue
                ct_arr = ct.to_numpy()
                try:
                    _, p_value, _, _ = stats.chi2_contingency(ct_arr, correction=False)
                except Exception:
                    p_value = 1.0
                # Reorder so rows = [control, treatment] for consistent OR direction
                try:
                    effect, ci_lo, ci_hi = _odds_ratio_ci(ct_arr)
                except Exception:
                    effect, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

            level_results.append(
                {
                    "level": str(level),
                    "n_treatment": int(n_trt),
                    "n_control": int(n_ctl),
                    "effect_size": round(float(effect), 4) if not np.isnan(effect) else None,
                    "ci_lower": round(float(ci_lo), 4) if not np.isnan(ci_lo) else None,
                    "ci_upper": round(float(ci_hi), 4) if not np.isnan(ci_hi) else None,
                    "p_value": round(float(p_value), 6),
                    "significant": bool(p_value < 0.05),
                }
            )

        # Interaction test for this subgroup variable
        if outcome_lower == "continuous":
            ix_p = _interaction_p_continuous(df, outcome_col, treatment_col, sg_col)
        else:
            ix_p = _interaction_p_binary(df, outcome_col, treatment_col, sg_col)

        sig_ix = bool(not np.isnan(ix_p) and ix_p < 0.05)
        if sig_ix:
            interp = (
                f"Significant treatment × {sg_col} interaction (p={ix_p:.3f}) — "
                "treatment effect is heterogeneous across subgroups."
            )
        else:
            p_str = f"p={ix_p:.3f}" if not np.isnan(ix_p) else "p=N/A"
            interp = (
                f"No significant treatment × {sg_col} interaction ({p_str}) — "
                "treatment effect appears consistent across subgroups."
            )

        subgroup_results.append(
            {
                "subgroup_column": sg_col,
                "levels": level_results,
                "interaction_p_value": round(float(ix_p), 6) if not np.isnan(ix_p) else None,
                "significant_interaction": sig_ix,
                "interpretation": interp,
            }
        )

    warning = (
        "Small subgroup sizes (<5 per arm in some levels) may reduce reliability of "
        "effect size estimates and interaction tests."
        if small_n_warning
        else None
    )

    return {
        "subgroups": subgroup_results,
        "overall_n": int(len(df)),
        "warning": warning,
    }


@tool(
    name="subgroup_analysis",
    description=(
        "Subgroup effect analysis with interaction testing (forest plot data). "
        "For each subgroup column, reports the treatment effect (Cohen's d or OR with 95% CI) "
        "at each subgroup level, plus an interaction test (p-value for treatment × subgroup). "
        "Significant interaction (p<0.05) indicates heterogeneous treatment effect — "
        "required reporting for FDA Phase III submissions."
    ),
    params_model=SubgroupParams,
    category="stats",
)
def subgroup_analysis(params: SubgroupParams, ctx: AgentContext) -> dict:
    df = ctx.dataframe
    for col in [params.outcome_col, params.treatment_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {ctx.column_names}")
    missing_sg = [c for c in params.subgroup_cols if c not in df.columns]
    if missing_sg:
        raise ValueError(
            f"Subgroup columns not found: {missing_sg}. Available: {ctx.column_names}"
        )
    return _subgroup_logic(
        df,
        params.outcome_col,
        params.treatment_col,
        params.subgroup_cols,
        params.outcome_type,
    )
