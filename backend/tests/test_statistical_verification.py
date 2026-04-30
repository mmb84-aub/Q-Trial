import pandas as pd

from qtrial_backend.agentic.schemas import (
    ComparableFinding,
    ComparisonMetrics,
    ComparisonReport,
    FinalReportSchema,
    GroundedFindingsSchema,
    InsightSynthesisOutput,
    MetadataInput,
    PlanSchema,
    PlanStep,
    StatisticalVerificationMetrics,
    StatisticalVerificationReport,
    UnknownsOutput,
    VerifiedClaim,
)
from qtrial_backend.agentic.statistical_verification import build_statistical_verification_report


def test_metadata_input_accepts_statistical_verification_fields() -> None:
    metadata = MetadataInput.model_validate(
        {
            "primary_endpoint": "DEATH_EVENT",
            "time_column": "followup_time",
            "event_column": "status",
            "event_codes": [2],
            "group_column": "allocation",
            "status_mapping": {"0": "censored", "2": "death"},
        }
    )

    assert metadata.primary_endpoint == "DEATH_EVENT"
    assert metadata.time_column == "followup_time"
    assert metadata.event_column == "status"
    assert metadata.event_codes == [2]
    assert metadata.group_column == "allocation"
    assert metadata.status_mapping == {"0": "censored", "2": "death"}


def _make_report(
    comparison_report: ComparisonReport | None = None,
    statistical_verification_report: StatisticalVerificationReport | None = None,
) -> FinalReportSchema:
    return FinalReportSchema(
        provider="test",
        model="test-model",
        plan=PlanSchema(
            dataset_summary="Synthetic dataset",
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
            key_findings=[],
            risks_and_bias_signals=[],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
        grounded_findings=GroundedFindingsSchema(findings=[]),
        comparison_report=comparison_report,
        statistical_verification_report=statistical_verification_report,
    )


def test_verified_binary_association_claim() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "DEATH_EVENT": [0] * 12 + [1] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with DEATH_EVENT (p=0.001).",
        analyst_report_name="analyst.txt",
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.test_used in {"Fisher's exact", "Chi-square"}
    assert claim.variable == "trt"
    assert claim.endpoint == "DEATH_EVENT"
    assert claim.recomputed_p_value is not None
    assert claim.recomputed_p_value < 0.05
    assert report.metrics.verified_count == 1
    assert report.metrics.verification_rate == 1.0


def test_raw_chi_square_artifact_is_not_extracted_for_verification() -> None:
    df = pd.DataFrame(
        {
            "platelets": [100, 110, 120, 130],
            "DEATH_EVENT": [0, 0, 1, 1],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="platelets: χ²=2232.8060, p=0.0000",
        analyst_report_name="analyst.txt",
    )

    assert report.claims == []
    assert report.metrics.total_claims == 0


def test_header_wrapper_is_not_extracted_for_verification() -> None:
    df = pd.DataFrame(
        {
            "age": [50, 60, 70, 80],
            "DEATH_EVENT": [0, 0, 1, 1],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Hazard Ratios (HR with 95% CI):\nTest_Selection_Rationale:",
        analyst_report_name="analyst.txt",
    )

    assert report.claims == []
    assert report.metrics.total_claims == 0


def test_contradicted_significance_claim() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 0, 0, 0, 1, 1, 1, 1],
            "DEATH_EVENT": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with DEATH_EVENT (p=0.01).",
    )

    claim = report.claims[0]
    assert claim.label == "contradicted"
    assert claim.recomputed_p_value is not None
    assert claim.recomputed_p_value >= 0.05
    assert report.metrics.contradicted_count == 1
    assert report.metrics.contradiction_rate == 1.0


def test_vague_clinical_claim_is_not_verifiable() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 1, 0, 1],
            "DEATH_EVENT": [0, 1, 0, 1],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="The treatment appears clinically promising and useful.",
    )

    assert report.claims[0].label == "not_verifiable"
    assert report.claims[0].test_used is None
    assert report.metrics.not_verifiable_count == 1


def test_verified_kaplan_meier_logrank_claim() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "DEATH_EVENT": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in survival by log-rank test (p=0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.test_used == "kaplan_meier_logrank"
    assert claim.endpoint == "DEATH_EVENT"
    assert claim.recomputed_p_value is not None
    assert claim.recomputed_p_value < 0.05


def test_final_report_serializes_comparison_and_statistical_verification_reports() -> None:
    statistical_report = StatisticalVerificationReport(
        summary="Verified 1 of 1 extracted claims.",
        metrics=StatisticalVerificationMetrics(
            total_claims=1,
            verified_count=1,
            verification_rate=1.0,
        ),
        claims=[
            VerifiedClaim(
                claim_id="analyst_1",
                source_text="Treatment was associated with DEATH_EVENT.",
                label="verified",
                variable="trt",
                endpoint="DEATH_EVENT",
                test_used="Fisher's exact",
                recomputed_p_value=0.01,
                rationale="Claim agrees with recomputed result.",
            )
        ],
    )
    comparison_report = ComparisonReport(
        analyst_report_name="analyst.txt",
        summary="No comparison issues.",
        metrics=ComparisonMetrics(total_qtrial_findings=1, total_human_findings=1),
    )

    report = _make_report(
        comparison_report=comparison_report,
        statistical_verification_report=statistical_report,
    )

    payload = report.model_dump(mode="json")
    assert payload["comparison_report"]["analyst_report_name"] == "analyst.txt"
    assert payload["statistical_verification_report"]["metrics"]["verification_rate"] == 1.0
    assert payload["statistical_verification_report"]["claims"][0]["label"] == "verified"


def test_qtrial_findings_are_included_with_analyst_claims() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "DEATH_EVENT": [0] * 12 + [1] * 12,
        }
    )
    qtrial_finding = ComparableFinding(
        finding_id="qtrial_mortality",
        source="qtrial",
        source_label="Q-Trial",
        finding_text="Treatment was significantly associated with DEATH_EVENT (p<0.001).",
        normalized_text="treatment associated death_event",
        variable="trt",
        endpoint="DEATH_EVENT",
        significant=True,
        significance="significant",
        p_value=0.001,
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[qtrial_finding],
        analyst_report_text="The treatment appears clinically promising and useful.",
    )

    claim_ids = {claim.claim_id for claim in report.claims}
    assert "analyst_1" in claim_ids
    assert "qtrial_1" in claim_ids
    qtrial_claim = next(claim for claim in report.claims if claim.claim_id == "qtrial_1")
    assert qtrial_claim.metadata["source"] == "qtrial"
    assert qtrial_claim.label == "verified"


def test_qtrial_dict_finding_text_raw_is_evaluated() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "DEATH_EVENT": [0] * 12 + [1] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[
            {
                "finding_text_raw": "Treatment was significantly associated with DEATH_EVENT (p<0.001).",
            }
        ],
        analyst_report_text="Administrative note only with no statistical claim.",
    )

    assert any(claim.claim_id == "qtrial_1" for claim in report.claims)
    assert report.claims[-1].label == "verified"


def test_directionally_wrong_significant_claim_is_contradicted() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "DEATH_EVENT": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment increased DEATH_EVENT risk significantly (p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "contradicted"
    assert "direction" in claim.rationale


def test_or_below_one_supports_lower_ejection_fraction_higher_mortality_claim() -> None:
    df = pd.DataFrame(
        {
            "ejection_fraction": [
                60, 58, 56, 54, 52, 50, 48, 46, 44, 42,
                40, 38, 36, 34, 32, 30, 28, 26, 24, 22,
                59, 55, 51, 47, 43, 39, 35, 31, 27, 23,
            ] * 2,
            "DEATH_EVENT": [
                0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            ] * 2,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text=(
            "Lower ejection_fraction was associated with higher mortality risk "
            "(OR=0.90, p<0.05)."
        ),
    )

    claim = report.claims[0]
    assert claim.label in {"verified", "partial"}
    assert claim.test_used == "logistic_regression"
    assert claim.effect_size is not None
    assert claim.effect_size < 1
    assert claim.effect_agreement != "conflicts"
    assert "direction" not in claim.rationale.lower() or "conflicts" not in claim.rationale.lower()


def test_p_greater_than_threshold_is_not_treated_as_significant() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 0, 0, 0, 1, 1, 1, 1],
            "DEATH_EVENT": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment association with DEATH_EVENT had p > 0.05.",
    )

    claim = report.claims[0]
    assert claim.metadata["claimed_significant"] is False
    assert claim.metadata["reported_p_operator"] == ">"
    assert claim.label == "verified"


def test_p_less_than_scientific_threshold_can_verify() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 20 + [1] * 20,
            "DEATH_EVENT": [0] * 20 + [1] * 20,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with DEATH_EVENT (p < 1e-3).",
    )

    claim = report.claims[0]
    assert claim.metadata["reported_p_operator"] == "<"
    assert claim.reported_p_value == 0.001
    assert claim.label == "verified"


def test_adjusted_or_independent_predictor_claim_is_not_verified_univariably() -> None:
    df = pd.DataFrame(
        {
            "age": list(range(30, 70)),
            "DEATH_EVENT": [0] * 20 + [1] * 20,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Age was an independent predictor of DEATH_EVENT.",
    )

    claim = report.claims[0]
    assert claim.label == "unsupported"
    assert "univariable" in claim.rationale


def test_ambiguous_survival_event_coding_is_not_verifiable() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 0, 0, 1, 1, 1],
            "time": [5, 6, 7, 8, 9, 10],
            "status": [0, 1, 2, 0, 1, 2],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in survival by log-rank test.",
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert "event coding is ambiguous" in claim.rationale


def test_explicit_endpoint_mention_beats_metadata_primary_endpoint() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "DEATH_EVENT": [0] * 12 + [1] * 12,
            "ALT_OUTCOME": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with DEATH_EVENT (p<0.001).",
        metadata={"primary_endpoint": "ALT_OUTCOME"},
    )

    claim = report.claims[0]
    assert claim.endpoint == "DEATH_EVENT"
    assert claim.metadata["column_resolutions"][0]["source"] == "explicit_text"
    assert claim.label == "verified"


def test_invalid_metadata_endpoint_does_not_force_verification() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 1, 0, 1],
            "lab_value": [1.0, 2.0, 1.5, 2.5],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was associated with the primary endpoint.",
        metadata={"primary_endpoint": "missing_endpoint"},
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert "missing_endpoint" in " ".join(claim.confidence_warnings)


def test_multiple_endpoint_like_columns_are_not_guessed() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 0, 1, 1],
            "time": [5, 6, 7, 8],
            "DEATH_EVENT": [0, 1, 0, 1],
            "TRANSPLANT_EVENT": [1, 0, 1, 0],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed in survival by log-rank test.",
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert any("Multiple plausible endpoint columns" in warning for warning in claim.confidence_warnings)


def test_multiple_time_like_columns_are_not_guessed() -> None:
    df = pd.DataFrame(
        {
            "trt": [0, 0, 1, 1],
            "followup_time": [5, 6, 7, 8],
            "lab_time": [1, 2, 3, 4],
            "DEATH_EVENT": [0, 1, 0, 1],
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed in survival by log-rank test.",
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert any("Multiple plausible time columns" in warning for warning in claim.confidence_warnings)


def test_unique_data_dictionary_alias_resolves_endpoint() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "outcome_a": [0] * 12 + [1] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with relapse (p<0.001).",
        column_dict={"outcome_a": "relapse event indicator"},
    )

    claim = report.claims[0]
    assert claim.endpoint == "outcome_a"
    assert claim.metadata["column_resolutions"][0]["source"] == "data_dictionary"
    assert claim.label == "verified"


def test_ambiguous_data_dictionary_alias_refuses_endpoint_resolution() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "outcome_a": [0] * 12 + [1] * 12,
            "outcome_b": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with relapse (p<0.001).",
        column_dict={
            "outcome_a": "relapse event indicator",
            "outcome_b": "relapse event adjudication",
        },
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert any("Data dictionary matched multiple endpoint candidates" in warning for warning in claim.confidence_warnings)


def test_survival_with_explicit_status_mapping_verifies_safely() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "status": [2] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in mortality survival by log-rank test (p<0.001).",
        metadata={"status_mapping": {"0": "censored", "1": "transplant", "2": "death"}},
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.test_used == "kaplan_meier_logrank"


def test_same_survival_claim_without_status_mapping_is_not_verifiable() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "status": [2] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in mortality survival by log-rank test (p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert "event coding is ambiguous" in claim.rationale


def test_metadata_time_column_resolves_multiple_time_like_columns() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "followup_time": list(range(20, 32)) + list(range(80, 92)),
            "lab_time": list(range(1, 25)),
            "DEATH_EVENT": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in mortality survival by log-rank test (p<0.001).",
        metadata={"time_column": "followup_time"},
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.metadata["time_column"] == "followup_time"
    assert any(
        resolution["role"] == "time" and resolution["source"] == "metadata"
        for resolution in claim.metadata["column_resolutions"]
    )


def test_metadata_event_column_resolves_ambiguous_endpoint_candidates() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "DEATH_EVENT": [1] * 12 + [0] * 12,
            "TRANSPLANT_EVENT": [0] * 12 + [1] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in survival by log-rank test (p<0.001).",
        metadata={"event_column": "DEATH_EVENT"},
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.endpoint == "DEATH_EVENT"
    assert any("metadata.event_column=DEATH_EVENT" in warning for warning in claim.confidence_warnings)


def test_metadata_group_column_used_for_logrank_grouping() -> None:
    df = pd.DataFrame(
        {
            "allocation": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "DEATH_EVENT": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Study groups differed significantly in mortality survival by log-rank test (p<0.001).",
        metadata={"group_column": "allocation"},
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.metadata["group_column"] == "allocation"
    assert any(
        resolution["role"] == "group" and resolution["source"] == "metadata"
        for resolution in claim.metadata["column_resolutions"]
    )


def test_non_binary_event_status_verifies_with_event_codes_metadata() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "status": [2] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in mortality survival by log-rank test (p<0.001).",
        metadata={"event_codes": [2]},
    )

    claim = report.claims[0]
    assert claim.label == "verified"
    assert claim.test_used == "kaplan_meier_logrank"


def test_invalid_metadata_column_returns_not_verifiable_with_warning() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "DEATH_EVENT": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in mortality survival by log-rank test (p<0.001).",
        metadata={"time_column": "missing_time"},
    )

    claim = report.claims[0]
    assert claim.label == "not_verifiable"
    assert "time/follow-up column" in claim.rationale
    assert any("missing_time" in warning for warning in claim.confidence_warnings)


def test_or_opposite_side_of_null_is_contradicted() -> None:
    df = pd.DataFrame(
        {
            "age": list(range(20, 70)),
            "DEATH_EVENT": [0] * 20 + [1, 0, 1, 1, 0] * 6,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Age was a significant predictor of DEATH_EVENT with OR=0.5 (p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "contradicted"
    assert claim.reported_effect_size_label == "odds_ratio"
    assert claim.effect_size is not None and claim.effect_size > 1
    assert claim.effect_agreement == "conflicts"


def test_hr_ci_mismatch_downgrades_or_contradicts_claim() -> None:
    df = pd.DataFrame(
        {
            "age": list(range(30, 70)),
            "time": list(range(10, 50)),
            "DEATH_EVENT": [0, 1] * 20,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Age was a significant predictor of mortality survival with HR=2.0 (95% CI 1.2 to 3.1, p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label in {"partial", "contradicted"}
    assert claim.reported_effect_size_label == "hazard_ratio"
    assert claim.reported_ci_lower == 1.2
    assert claim.reported_ci_upper == 3.1
    assert claim.effect_agreement in {"partial", "conflicts"}


def test_correlation_opposite_sign_effect_is_contradicted() -> None:
    df = pd.DataFrame(
        {
            "creatinine": list(range(1, 31)),
            "ejection_fraction": list(range(30, 0, -1)),
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Creatinine and ejection_fraction were positively correlated (r=0.8, p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "contradicted"
    assert claim.effect_size is not None and claim.effect_size < 0
    assert claim.effect_agreement == "conflicts"


def test_continuous_effect_opposite_sign_is_contradicted() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 20 + [1] * 20,
            "biomarker": list(range(80, 100)) + list(range(20, 40)),
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment group had higher biomarker with Cohen's d=1.0 (p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "contradicted"
    assert claim.reported_effect_size_label == "cohen_d"
    assert claim.effect_agreement == "conflicts"


def test_p_value_agrees_but_or_materially_differs_is_partial() -> None:
    df = pd.DataFrame(
        {
            "risk_score": [0] * 25 + [2] * 25,
            "DEATH_EVENT": [0] * 20 + [1] * 5 + [0] * 5 + [1] * 20,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="risk_score was a significant predictor of DEATH_EVENT with OR=1.1 (p<0.001).",
    )

    claim = report.claims[0]
    assert claim.label == "partial"
    assert claim.effect_agreement == "partial"
    assert "materially differs" in claim.rationale


def test_fisher_categorical_without_or_ci_does_not_fake_effect_agreement() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 6 + [1] * 6,
            "DEATH_EVENT": [0] * 6 + [1] * 6,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment was significantly associated with DEATH_EVENT (p<0.01).",
    )

    claim = report.claims[0]
    assert claim.test_used == "Fisher's exact"
    assert claim.reported_effect_size is None
    assert claim.effect_agreement == "not_assessed"


def test_km_logrank_remains_p_value_only_for_effect_agreement() -> None:
    df = pd.DataFrame(
        {
            "trt": [0] * 12 + [1] * 12,
            "time": list(range(20, 32)) + list(range(80, 92)),
            "DEATH_EVENT": [1] * 12 + [0] * 12,
        }
    )

    report = build_statistical_verification_report(
        df=df,
        qtrial_findings=[],
        analyst_report_text="Treatment groups differed significantly in mortality survival by log-rank test (p<0.001).",
    )

    claim = report.claims[0]
    assert claim.test_used == "kaplan_meier_logrank"
    assert claim.effect_size is None
    assert claim.effect_agreement == "not_assessed"
