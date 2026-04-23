import pandas as pd

from qtrial_backend.agentic.orchestrator import run_agentic_insights
from qtrial_backend.agentic.schemas import GroundedFinding, ResearchQuestion, SynthesisOutput


def test_qc_findings_remain_visible_in_grounded_findings(monkeypatch) -> None:
    monkeypatch.setattr(
        "qtrial_backend.agentic.orchestrator.translate_findings_to_cst",
        lambda findings, study_context, provider: [],
    )

    class _StubPipeline:
        def __init__(self, provider):  # noqa: ANN001
            self.query_records = []

        def validate(self, csts):  # noqa: ANN001
            return [
                GroundedFinding(
                    finding_text="Age was significantly associated with mortality.",
                    finding_text_plain="Age was significantly associated with mortality.",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                )
            ]

    monkeypatch.setattr(
        "qtrial_backend.agentic.orchestrator.LiteratureValidatorPipeline",
        _StubPipeline,
    )
    monkeypatch.setattr(
        "qtrial_backend.agentic.orchestrator.run_synthesis_call",
        lambda analysis_report, grounded_findings, study_context, provider: (
            SynthesisOutput(
                future_trial_hypothesis="Test hypothesis",
                endpoint_improvement_recommendations=[],
                recommended_sample_size="100",
                variables_to_control=[],
                research_questions=[ResearchQuestion(question="Q?", source_finding="Age finding")],
                narrative_summary="Summary.",
            ),
            "Summary.",
        ),
    )

    df = pd.DataFrame({"age": [50, 60], "DEATH_EVENT": [0, 1]})
    report = run_agentic_insights(
        df=df,
        provider="gemini",
        analysis_report="Results: Age was significantly associated with mortality.",
        study_context="Heart failure prognosis study.",
        clinical_analysis={
            "stage_3_corrections": {
                "corrected_findings": [
                    {
                        "finding_id": "age",
                        "finding_text_plain": "Age was significantly associated with mortality.",
                        "finding_category": "analytical",
                        "claim_type": "association_claim",
                        "significant_after_correction": True,
                        "adjusted_p_value": 0.01,
                        "effect_size": 0.4,
                    },
                    {
                        "finding_id": "duplicates",
                        "finding_text_plain": "Key-column duplicates were detected during integrity checks.",
                        "finding_category": "data_quality",
                        "claim_type": "data_quality_claim",
                    },
                ]
            }
        },
    )

    findings = report.grounded_findings.findings  # type: ignore[union-attr]
    assert len(findings) == 2
    assert any(f.finding_category == "analytical" for f in findings)
    assert any(f.finding_category == "data_quality" for f in findings)
    qc = next(f for f in findings if f.finding_category == "data_quality")
    assert qc.grounding_status == "Data Quality Note"
