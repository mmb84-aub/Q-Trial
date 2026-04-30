import pandas as pd

from qtrial_backend.agentic.report_curation import curate_user_facing_report_sections, report_contains_banned_user_facing_text
from qtrial_backend.agentic.schemas import (
    FinalReportSchema,
    GroundedFinding,
    GroundedFindingsSchema,
    InsightSynthesisOutput,
    PlanSchema,
    PlanStep,
)


def _minimal_report(**kwargs) -> FinalReportSchema:
    final_insights = kwargs.pop(
        "final_insights",
        InsightSynthesisOutput(
            key_findings=[],
            risks_and_bias_signals=[],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
    )
    return FinalReportSchema(
        provider="gemini",
        model="test-model",
        plan=PlanSchema(
            dataset_summary="x",
            steps=[
                PlanStep(
                    step_number=1,
                    name="Stub",
                    goal="Stub",
                    inputs_used=["dataframe"],
                    expected_output_keys=["analysis_report"],
                    agent_to_call="DataQualityAgent",
                )
            ],
        ),
        agent_runs=[],
        unknowns={"ranked_unknowns": [], "explicit_assumptions": [], "required_documents": [], "summary": ""},  # type: ignore[arg-type]
        final_insights=final_insights,
        **kwargs,
    )


def test_curation_removes_banned_agent_phrases_everywhere() -> None:
    report = _minimal_report(
        prior_analysis_report="Now I have all the necessary information.\n> Generated 2026-01-01 · Fully deterministic, no LLM\n",
        final_insights=InsightSynthesisOutput(
            key_findings=["Let me compile the comprehensive final report."],
            risks_and_bias_signals=["Next step: run analyze"],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
        grounded_findings=GroundedFindingsSchema(
            findings=[
                GroundedFinding(
                    finding_text="Now I have all the necessary information.",
                    finding_text_plain="Now I have all the necessary information.",
                    finding_category="statistical_note",
                    grounding_status="QC Observation",
                    literature_skipped=True,
                ),
            ]
        ),
    )

    curated = curate_user_facing_report_sections(report)
    assert report_contains_banned_user_facing_text(curated) == []


def test_curation_caps_qc_notes_and_dedupes() -> None:
    qc = [
        GroundedFinding(
            finding_text=f"Missingness note {i}: 10% missing",
            finding_text_plain=f"Missingness note {i}: 10% missing",
            finding_category="statistical_note",
            grounding_status="QC Observation",
            literature_skipped=True,
        )
        for i in range(30)
    ]
    # Add duplicates and a high-signal note that must remain.
    qc.append(qc[0])
    qc.append(
        GroundedFinding(
            finding_text="Digit preference / rounding detected in lab values.",
            finding_text_plain="Digit preference / rounding detected in lab values.",
            finding_category="data_quality",
            grounding_status="Data Quality Note",
            literature_skipped=True,
        )
    )

    report = _minimal_report(
        grounded_findings=GroundedFindingsSchema(findings=qc),
    )
    curated = curate_user_facing_report_sections(report)
    findings = curated.grounded_findings.findings  # type: ignore[union-attr]
    qc_notes = [f for f in findings if f.finding_category in {"statistical_note", "data_quality"}]
    assert len(qc_notes) <= 12
    assert any("digit preference" in (f.finding_text_plain or "").lower() for f in qc_notes)


def test_curation_makes_recommendations_cautious_in_observational_context() -> None:
    report = _minimal_report(
        study_context="Retrospective observational cohort dataset.",
        final_insights=InsightSynthesisOutput(
            key_findings=[],
            risks_and_bias_signals=[],
            required_metadata_or_questions=[],
            recommended_next_analyses=[
                {
                    "rank": 1,
                    "analysis": "Test an intervention",
                    "rationale": "This will improve survival based on observed associations.",
                    "evidence_citation": "",
                }
            ],  # type: ignore[arg-type]
        ),
    )
    curated = curate_user_facing_report_sections(report)
    rationale = curated.final_insights.recommended_next_analyses[0].rationale  # type: ignore[union-attr]
    assert "will improve survival" not in rationale.lower()
    assert "hypothesis-generating" in rationale.lower()
