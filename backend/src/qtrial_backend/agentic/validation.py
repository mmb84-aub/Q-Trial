from __future__ import annotations

import re

from qtrial_backend.agentic.schemas import GroundedFinding, SynthesisOutput


def validate_synthesis_output(
    synthesis: SynthesisOutput,
    grounded_findings: list[GroundedFinding],
) -> list[str]:
    """
    Deterministic structural validation of a SynthesisOutput.
    Returns a list of error strings; an empty list means the output is valid.
    """
    errors: list[str] = []

    # Check 1 — Research question grounding
    for rq in synthesis.research_questions:
        sf = (rq.source_finding or "").strip()
        if not sf or len(sf.split()) < 4:
            errors.append(
                f"Research question '{rq.question[:60]}' has a missing or trivial source_finding."
            )

    # Check 2 — Citation enforcement
    for gf in grounded_findings:
        if gf.grounding_status in ("Supported", "Contradicted") and not gf.literature_skipped:
            if not gf.citations:
                errors.append(
                    f"Finding '{gf.finding_text[:60]}' is marked {gf.grounding_status} but has no citations."
                )

    # Check 3 — Recommendation specificity
    _num_pattern = re.compile(r"\d+\.?\d*")
    for rec in synthesis.endpoint_improvement_recommendations:
        if not _num_pattern.search(rec):
            errors.append(
                f"Recommendation lacks quantitative specifics: '{rec[:60]}'"
            )

    # Check 4 — Hypothesis non-triviality
    hypothesis = (synthesis.future_trial_hypothesis or "").strip()
    if len(hypothesis.split()) <= 10 or hypothesis == "Further investigation required.":
        errors.append("future_trial_hypothesis is empty or a fallback placeholder.")

    # Check 5 — Sample size non-triviality
    sample_size = (synthesis.recommended_sample_size or "").strip()
    if not _num_pattern.search(sample_size) and len(sample_size) <= 20:
        errors.append("recommended_sample_size is not populated meaningfully.")

    return errors


def build_retry_prompt(errors: list[str], original_user_prompt: str) -> str:
    """
    Build a re-prompt string for the LLM when synthesis validation fails.
    """
    numbered = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(errors))
    return (
        "The previous synthesis output failed structural validation with the following errors:\n\n"
        f"{numbered}\n\n"
        "Please regenerate the synthesis output as valid JSON, correcting each issue above. "
        "All other fields should remain consistent with the original analysis.\n\n"
        "Original prompt for reference:\n"
        f"{original_user_prompt}"
    )
