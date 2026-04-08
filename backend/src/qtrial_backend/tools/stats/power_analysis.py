"""
Input: standardized effect size, n per group, alpha, test type
Output: achieved power, required n for 80% and 90% power
Purpose: Distinguish statistical significance from clinical meaningfulness. A significant
  p-value in an underpowered study may be a false positive. Power < 80% must be disclosed.
Reference: Cohen (1988) Statistical Power Analysis for the Behavioral Sciences, 2nd ed.;
  ICH E9 Statistical Principles for Clinical Trials.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.tools.registry import tool


class PowerAnalysisParams(BaseModel):
    effect_size: float = Field(
        description=(
            "Standardised effect size (Cohen's d, Cohen's h, or similar). "
            "Typical thresholds: small=0.2, medium=0.5, large=0.8."
        )
    )
    n_per_group: int = Field(
        description="Observed (or planned) sample size per group.", ge=2
    )
    alpha: float = Field(
        default=0.05,
        description="Significance level (Type I error rate). Default 0.05.",
        gt=0.0,
        lt=1.0,
    )
    test_type: str = Field(
        default="two-sample",
        description=(
            "'two-sample' (independent groups, uses TTestIndPower), "
            "'one-sample' (single group vs. null, uses TTestPower), "
            "'paired' (paired observations, uses TTestPower)."
        ),
    )


def _classify_power(achieved_power: float) -> str:
    if achieved_power >= 0.80:
        return "Adequate (≥80%)"
    if achieved_power >= 0.60:
        return "Moderate (60–79%) — underpowered"
    return "Low (<60%) — results unreliable"


def _power_logic(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05,
    test_type: str = "two-sample",
) -> dict:
    """Core power analysis logic — callable directly for programmatic use."""
    try:
        from statsmodels.stats.power import TTestIndPower, TTestPower  # type: ignore
    except ImportError:
        raise RuntimeError(
            "statsmodels is required for power analysis. Run: pip install statsmodels"
        )

    test_lower = test_type.lower()

    if test_lower == "two-sample":
        analysis = TTestIndPower()
        achieved_power = float(
            analysis.power(effect_size, nobs1=n_per_group, alpha=alpha)
        )
        n_80 = int(
            round(analysis.solve_power(effect_size, power=0.80, alpha=alpha))
        )
        n_90 = int(
            round(analysis.solve_power(effect_size, power=0.90, alpha=alpha))
        )
    elif test_lower in ("one-sample", "paired"):
        analysis = TTestPower()
        achieved_power = float(
            analysis.power(effect_size, nobs=n_per_group, alpha=alpha)
        )
        n_80 = int(
            round(analysis.solve_power(effect_size, power=0.80, alpha=alpha))
        )
        n_90 = int(
            round(analysis.solve_power(effect_size, power=0.90, alpha=alpha))
        )
    else:
        raise ValueError(
            f"test_type must be 'two-sample', 'one-sample', or 'paired'. Got: '{test_type}'"
        )

    classification = _classify_power(achieved_power)
    adequate = achieved_power >= 0.80

    interpretation = (
        f"With n={n_per_group}/group and d={effect_size:.2f}, "
        f"achieved power is {achieved_power * 100:.0f}% — "
        f"{'study is adequately powered.' if adequate else 'study is underpowered.'}"
    )

    return {
        "effect_size_input": effect_size,
        "n_per_group_observed": n_per_group,
        "alpha": alpha,
        "test_type": test_type,
        "achieved_power": round(achieved_power, 4),
        "power_classification": classification,
        "n_required_80pct_power": n_80,
        "n_required_90pct_power": n_90,
        "adequately_powered": adequate,
        "interpretation": interpretation,
    }


def batch_power_analysis(findings: list[dict]) -> list[dict]:
    """Run power analysis for multiple findings.

    Each finding dict must contain: 'finding_id', 'effect_size', 'n_per_group', 'alpha'.
    Returns each dict extended with power analysis results.
    """
    results = []
    for finding in findings:
        effect_size = float(finding.get("effect_size", 0.0))
        n_per_group = int(finding.get("n_per_group", 2))
        alpha = float(finding.get("alpha", 0.05))
        test_type = str(finding.get("test_type", "two-sample"))

        try:
            power_result = _power_logic(effect_size, n_per_group, alpha, test_type)
        except Exception as exc:
            power_result = {"error": str(exc)}

        results.append({**finding, **power_result})
    return results


@tool(
    name="power_analysis",
    description=(
        "Compute achieved statistical power and required sample size thresholds. "
        "Uses the t-test power framework (TTestIndPower for two-sample, TTestPower for one-sample/paired). "
        "Reports achieved power, 80%- and 90%-power sample size requirements, and a "
        "classification (adequate/moderate/low). "
        "A significant p-value from an underpowered study (<80% power) may be a false positive."
    ),
    params_model=PowerAnalysisParams,
    category="stats",
)
def power_analysis(params: PowerAnalysisParams, ctx: AgentContext) -> dict:  # noqa: ARG001
    return _power_logic(
        params.effect_size,
        params.n_per_group,
        params.alpha,
        params.test_type,
    )
