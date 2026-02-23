from __future__ import annotations

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool

# ── Decision tree rules ───────────────────────────────────────────────────────
_CONTINUOUS_2_GROUPS = {
    "parametric": {
        "test": "Independent t-test (Welch's)",
        "assumption": "Normality in each group or n >= 30 per group",
        "effect_size": "Cohen's d",
        "tool": "hypothesis_test",
    },
    "non_parametric": {
        "test": "Mann-Whitney U",
        "assumption": "No normality assumption; ordinal or continuous data",
        "effect_size": "Cliff's delta",
        "tool": "hypothesis_test",
    },
    "paired": {
        "test": "Paired t-test or Wilcoxon signed-rank",
        "assumption": "Paired observations (same subject, two timepoints)",
        "effect_size": "Cohen's d (paired)",
        "tool": "hypothesis_test (or note: paired not yet supported)",
    },
}

_CONTINUOUS_3PLUS_GROUPS = {
    "non_parametric": {
        "test": "Kruskal-Wallis H-test (overall) + pairwise Mann-Whitney U with Bonferroni",
        "tool": "pairwise_group_test",
    },
    "parametric": {
        "test": "One-way ANOVA (overall) + post-hoc Tukey HSD",
        "note": "Use parametric only if all groups pass normality_test",
        "tool": "pairwise_group_test (Kruskal-Wallis version available)",
    },
}

_BINARY_2_GROUPS = {
    "default": {
        "test": "Chi-square test of independence",
        "fallback": "Fisher's exact test when any expected cell count < 5",
        "effect_size": "Odds Ratio, Risk Ratio",
        "tool": "cross_tabulation",
    },
    "paired": {
        "test": "McNemar's test",
        "note": "For matched pairs or repeated binary measurement",
    },
}

_SURVIVAL = {
    "2_groups": {
        "test": "Log-rank test",
        "model": "Kaplan-Meier curves per group",
        "multivariable": "Cox proportional hazards regression",
        "ph_check": "Schoenfeld residuals (check proportional hazards assumption)",
        "tool": "survival_analysis + regression(model_type='cox')",
    },
    "1_group": {
        "test": "Kaplan-Meier curve only",
        "tool": "survival_analysis",
    },
}


class StatTestSelectorParams(BaseModel):
    outcome_type: str = Field(
        description=(
            "Type of outcome variable: "
            "'continuous', 'binary', 'survival', 'count', or 'ordinal'."
        )
    )
    n_groups: int = Field(
        description="Number of groups being compared (1, 2, or 3+)."
    )
    paired: bool = Field(
        default=False,
        description="True if observations are paired/matched (e.g. before-after, matched controls).",
    )
    n_per_group: int = Field(
        default=50,
        description="Approximate sample size per group (used to assess normality assumption feasibility).",
    )
    design: str = Field(
        default="rct",
        description=(
            "Study design: 'rct' (randomised controlled trial), "
            "'observational', 'crossover', or 'repeated_measures'."
        ),
    )
    additional_context: str = Field(
        default="",
        description="Any extra context (e.g. 'outcome is highly skewed', 'small sample n<30').",
    )


@tool(
    name="stat_test_selector",
    description=(
        "Recommend the most appropriate statistical test(s) for a given analysis scenario. "
        "Provide the outcome type, number of groups, whether data is paired, sample size, and study design. "
        "Returns recommended test, alternatives, required assumptions, effect size measures, "
        "and which existing tools to use. Use this before running hypothesis_test or pairwise_group_test "
        "to ensure the correct test is applied."
    ),
    params_model=StatTestSelectorParams,
    category="stats",
)
def stat_test_selector(params: StatTestSelectorParams, ctx: AgentContext) -> dict:
    outcome = params.outcome_type.lower()
    n_groups = params.n_groups
    paired = params.paired
    n = params.n_per_group
    design = params.design.lower()
    context = params.additional_context

    recommendations: dict = {
        "scenario": {
            "outcome_type": outcome,
            "n_groups": n_groups,
            "paired": paired,
            "n_per_group": n,
            "design": design,
        },
        "warnings": [],
        "recommended": {},
        "alternatives": [],
        "effect_size": [],
        "tools_to_use": [],
        "assumptions_to_verify": [],
    }

    # ── Continuous outcome ────────────────────────────────────────────
    if outcome == "continuous":
        if n_groups == 1:
            recommendations["recommended"] = {
                "test": "Descriptive statistics only (mean, SD, median, IQR)",
                "note": "One group — no comparison possible without a reference.",
            }
        elif n_groups == 2:
            use_param = n >= 30
            rec = _CONTINUOUS_2_GROUPS["paired" if paired else ("parametric" if use_param else "non_parametric")]
            recommendations["recommended"] = rec
            alt_key = "non_parametric" if use_param else "parametric"
            recommendations["alternatives"] = [_CONTINUOUS_2_GROUPS[alt_key]]
            recommendations["effect_size"] = ["Cohen's d", "Cliff's delta"]
            recommendations["tools_to_use"] = ["normality_test", "hypothesis_test", "effect_size"]
            recommendations["assumptions_to_verify"] = [
                "Run normality_test on both groups first",
                "Check for outliers with outlier_detection",
                "Verify equal/unequal variance (Welch's handles unequal variance)",
            ]
            if n < 30:
                recommendations["warnings"].append(
                    f"Small n ({n}) per group — normality assumption is hard to verify; prefer Mann-Whitney U"
                )
        else:
            rec = _CONTINUOUS_3PLUS_GROUPS["non_parametric"]
            recommendations["recommended"] = rec
            recommendations["alternatives"] = [_CONTINUOUS_3PLUS_GROUPS["parametric"]]
            recommendations["effect_size"] = ["Epsilon-squared (rank-based)", "Pairwise Cohen's d with correction"]
            recommendations["tools_to_use"] = ["normality_test", "pairwise_group_test", "effect_size"]
            recommendations["assumptions_to_verify"] = [
                "Run normality_test on each group",
                "Apply multiple_testing_correction on pairwise p-values",
            ]

    # ── Binary outcome ────────────────────────────────────────────────
    elif outcome == "binary":
        if n_groups == 2:
            rec = _BINARY_2_GROUPS["paired" if paired else "default"]
            recommendations["recommended"] = rec
            recommendations["effect_size"] = ["Odds Ratio", "Risk Ratio", "Number Needed to Treat"]
            recommendations["tools_to_use"] = ["cross_tabulation", "effect_size"]
            recommendations["assumptions_to_verify"] = [
                "Check expected cell counts — if any < 5, use Fisher's exact (cross_tabulation handles this automatically)",
                "Ensure independence of observations",
            ]
        else:
            recommendations["recommended"] = {
                "test": "Chi-square test with multiple groups",
                "note": "Consider collapsing rare categories",
            }
            recommendations["tools_to_use"] = ["cross_tabulation"]

    # ── Survival / time-to-event ──────────────────────────────────────
    elif outcome == "survival":
        rec = _SURVIVAL["2_groups" if n_groups >= 2 else "1_group"]
        recommendations["recommended"] = rec
        recommendations["effect_size"] = ["Hazard Ratio (from Cox PH)", "Median survival difference"]
        recommendations["tools_to_use"] = [
            "survival_analysis",
            "regression (model_type='cox')",
        ]
        recommendations["assumptions_to_verify"] = [
            "Proportional hazards assumption — plot log(-log(S(t))) vs log(t); should be parallel lines",
            "Check for informative censoring",
            "Verify event rate is sufficient (rule of thumb: ≥ 10 events per predictor in Cox)",
        ]
        if n < 30:
            recommendations["warnings"].append(
                f"Small sample (n≈{n}) — KM curve confidence intervals will be wide; interpret cautiously"
            )

    # ── Count outcome ─────────────────────────────────────────────────
    elif outcome == "count":
        recommendations["recommended"] = {
            "test": "Negative binomial or Poisson regression (check for overdispersion)",
            "note": "Use Poisson if variance ≈ mean; negative binomial if variance >> mean",
            "tool": "regression (model_type='linear' as approximation; true count models not yet available)",
        }
        recommendations["warnings"].append(
            "True count regression (Poisson/NB) is not yet available as a tool — "
            "use linear regression as an approximation for large counts, or log-transform."
        )

    # ── Ordinal outcome ───────────────────────────────────────────────
    elif outcome == "ordinal":
        recommendations["recommended"] = {
            "test": "Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups)",
            "note": "Treat ordinal as numeric ranks; do NOT use parametric tests assuming equal intervals",
        }
        recommendations["tools_to_use"] = [
            "hypothesis_test (2 groups)",
            "pairwise_group_test (3+ groups)",
        ]
        recommendations["effect_size"] = ["Cliff's delta (rank-based)"]

    else:
        recommendations["warnings"].append(
            f"Unknown outcome_type '{outcome}'. "
            "Choose from: continuous, binary, survival, count, ordinal."
        )

    # ── Design-specific additions ─────────────────────────────────────
    if design in ("crossover", "repeated_measures") and "paired" not in str(recommendations.get("recommended", {})):
        recommendations["warnings"].append(
            f"Design '{design}' involves correlated measurements — ensure paired/mixed-effects approach."
        )

    if design == "rct":
        recommendations["tools_to_use"] = list(
            dict.fromkeys(["baseline_balance"] + recommendations.get("tools_to_use", []))
        )
        recommendations["assumptions_to_verify"] = [
            "Check baseline balance with baseline_balance tool first",
        ] + recommendations.get("assumptions_to_verify", [])

    if context:
        recommendations["additional_context_noted"] = context

    return recommendations
