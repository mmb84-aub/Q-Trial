import pandas as pd

from qtrial_backend.agentic.orchestrator import _sanitize_final_report, run_agentic_insights
from qtrial_backend.agentic.schemas import (
    FinalReportSchema,
    GroundedFinding,
    GroundedFindingsSchema,
    InsightSynthesisOutput,
    PlanSchema,
    PlanStep,
    ResearchQuestion,
    SynthesisOutput,
    UnknownsOutput,
)


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
    assert len(findings) == 1
    assert not any(f.finding_category == "analytical" for f in findings)
    assert any(f.finding_category in {"data_quality", "statistical_note"} for f in findings)
    qc = next(f for f in findings if f.finding_category in {"data_quality", "statistical_note"})
    assert qc.grounding_status in {"Data Quality Note", "QC Observation"}


def test_final_report_sanitizer_removes_raw_chi_square_from_primary_analytical_outputs() -> None:
    report = FinalReportSchema(
        provider="gemini",
        model="test-model",
        plan=PlanSchema(
            dataset_summary="Test dataset",
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
        unknowns=UnknownsOutput(
            ranked_unknowns=[],
            explicit_assumptions=[],
            required_documents=[],
            summary="",
        ),
        final_insights=InsightSynthesisOutput(
            key_findings=[
                "`time`: χ²=38.4916, p=0.0000",
                "Hazard Ratios (HR with 95% CI):",
                "Smoking was not significantly associated with mortality.",
            ],
            risks_and_bias_signals=["`smoking`: χ²=1387.4548, p=0.0000", "Test_Selection_Rationale:"],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
        grounded_findings=GroundedFindingsSchema(
            findings=[
                GroundedFinding(
                    finding_text="`time`: χ²=38.4916, p=0.0000",
                    finding_text_plain="`time`: χ²=38.4916, p=0.0000",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
                GroundedFinding(
                    finding_text="Smoking was not significantly associated with mortality.",
                    finding_text_plain="Smoking was not significantly associated with mortality.",
                    finding_category="analytical",
                    claim_type="negative_association",
                    grounding_status="Supported",
                ),
                GroundedFinding(
                    finding_text="Effect Sizes (Cohen's d with 95% bootstrap CI):",
                    finding_text_plain="Effect Sizes (Cohen's d with 95% bootstrap CI):",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
            ]
        ),
        clinical_analysis={
            "stage_3_corrections": {
                "corrected_findings": [
                    {
                        "finding_id": "smoking",
                        "finding_text_plain": "`smoking`: χ²=1387.4548, p=0.0000",
                        "finding_category": "analytical",
                    },
                    {
                        "finding_id": "platelets",
                        "finding_text_plain": "Platelets did not show statistically significant association with mortality.",
                        "finding_category": "analytical",
                    },
                ]
            }
        },
    )

    sanitized = _sanitize_final_report(report)

    assert sanitized.final_insights.key_findings == [
        "Smoking was not significantly associated with mortality."
    ]
    assert sanitized.final_insights.risks_and_bias_signals == []

    primary_findings = [
        finding for finding in sanitized.grounded_findings.findings
        if finding.finding_category == "analytical"
    ]
    note_findings = [
        finding for finding in sanitized.grounded_findings.findings
        if finding.finding_category == "statistical_note"
    ]
    assert [finding.finding_text for finding in primary_findings] == [
        "Smoking was not significantly associated with mortality."
    ]
    assert note_findings
    assert all("χ²" not in finding.finding_text for finding in primary_findings)

    corrected = sanitized.clinical_analysis["stage_3_corrections"]["corrected_findings"]
    excluded = sanitized.clinical_analysis["stage_3_corrections"]["artifact_excluded_findings"]
    assert len(corrected) == 1
    assert corrected[0]["finding_id"] == "platelets"
    assert excluded[0]["finding_id"] == "smoking"


def test_final_report_sanitizer_drops_malformed_and_demotes_contextual_findings() -> None:
    report = FinalReportSchema(
        provider="gemini",
        model="test-model",
        plan=PlanSchema(
            dataset_summary="Test dataset",
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
        unknowns=UnknownsOutput(
            ranked_unknowns=[],
            explicit_assumptions=[],
            required_documents=[],
            summary="",
        ),
        final_insights=InsightSynthesisOutput(
            key_findings=[
                "Deaths occurred much earlier (median 44.5 days vs.",
                "The dataset includes 8 predictor variables selected via QUBO feature selection.",
                "Higher serum creatinine was associated with higher mortality risk.",
            ],
            risks_and_bias_signals=[],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
        grounded_findings=GroundedFindingsSchema(
            findings=[
                GroundedFinding(
                    finding_text="Deaths occurred much earlier (median 44.5 days vs.",
                    finding_text_plain="Deaths occurred much earlier (median 44.5 days vs.",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
                GroundedFinding(
                    finding_text="The dataset includes 8 predictor variables selected via QUBO feature selection.",
                    finding_text_plain="The dataset includes 8 predictor variables selected via QUBO feature selection.",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
                GroundedFinding(
                    finding_text="Higher serum creatinine was associated with higher mortality risk.",
                    finding_text_plain="Higher serum creatinine was associated with higher mortality risk.",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
                GroundedFinding(
                    finding_text="- **Interpretation:** No association with mortality (41.7% diabetic in both groups).",
                    finding_text_plain="- **Interpretation:** No association with mortality (41.7% diabetic in both groups).",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
                GroundedFinding(
                    finding_text="**Event rate:** 32.1% (96 deaths out of 299 patients).",
                    finding_text_plain="**Event rate:** 32.1% (96 deaths out of 299 patients).",
                    finding_category="analytical",
                    claim_type="association_claim",
                    grounding_status="Supported",
                ),
            ]
        ),
        clinical_analysis={
            "stage_3_corrections": {
                "corrected_findings": [
                    {
                        "finding_id": "bad",
                        "finding_text_plain": "1.18 mg/dL (p<0.001, adjusted p=0.00014)",
                        "finding_category": "analytical",
                    },
                    {
                        "finding_id": "qubo",
                        "finding_text_plain": "The dataset includes 8 predictor variables selected via QUBO feature selection.",
                        "finding_category": "analytical",
                    },
                ]
            }
        },
    )

    sanitized = _sanitize_final_report(report)

    assert sanitized.final_insights.key_findings == [
        "Higher serum creatinine was associated with higher mortality risk."
    ]
    grounded_texts = [finding.finding_text for finding in sanitized.grounded_findings.findings]
    assert "Deaths occurred much earlier (median 44.5 days vs." not in grounded_texts
    assert "- **Interpretation:** No association with mortality (41.7% diabetic in both groups)." not in grounded_texts
    assert "**Event rate:** 32.1% (96 deaths out of 299 patients)." not in grounded_texts
    qubo = next(f for f in sanitized.grounded_findings.findings if "QUBO" in f.finding_text)
    assert qubo.finding_category == "preprocessing"
    assert qubo.grounding_status == "Preprocessing Observation"
    corrected = sanitized.clinical_analysis["stage_3_corrections"]["corrected_findings"]
    assert corrected == [
        {
            "finding_id": "qubo",
            "finding_text_plain": "The dataset includes 8 predictor variables selected via QUBO feature selection.",
            "finding_category": "preprocessing",
            "claim_type": "metadata_claim",
        }
    ]
