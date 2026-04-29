from qtrial_backend.agentic.report_comparison import (
    _build_metrics,
    build_comparison_report,
    parse_human_report_text,
)
from qtrial_backend.agentic.schemas import (
    ComparableFinding,
    FinalReportSchema,
    FindingMatch,
    GroundedFinding,
    GroundedFindingsSchema,
    InsightSynthesisOutput,
    PlanSchema,
    PlanStep,
    UnknownsOutput,
)


def _make_report(
    corrected_findings: list[dict],
    grounded_findings: list[GroundedFinding] | None = None,
) -> FinalReportSchema:
    return FinalReportSchema(
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
            key_findings=[],
            risks_and_bias_signals=[],
            recommended_next_analyses=[],
            required_metadata_or_questions=[],
        ),
        grounded_findings=GroundedFindingsSchema(findings=grounded_findings or []),
        clinical_analysis={
            "stage_3_corrections": {
                "corrected_findings": corrected_findings,
            }
        },
    )


def test_parse_human_report_uses_current_section_header() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        """
        Results:
        - Mortality was statistically significant with p=0.01 [1].
        Discussion:
        - Survival was not significant after correction (p=0.20).
        """,
    )

    assert len(parsed.findings) == 2
    assert parsed.findings[0].section == "results"
    assert parsed.findings[1].section == "discussion"


def test_report_title_and_variable_heading_are_not_extracted_as_claims() -> None:
    parsed = parse_human_report_text(
        "heart_failure.md",
        """
        # Clinical Analysis Report: Ejection fraction and mortality

        Results:
        Ejection fraction and mortality
        - Lower ejection fraction was associated with higher mortality risk (OR=0.94, p=8e-6).
        """,
        known_variables={"ejection_fraction"},
    )

    assert len(parsed.findings) == 1
    assert parsed.findings[0].finding_text.startswith("Lower ejection fraction")
    assert "Clinical Analysis Report" not in parsed.findings[0].finding_text


def test_build_comparison_report_handles_empty_human_findings() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "adjusted_p_value": 0.01,
                "effect_size": 0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="notes.txt",
        analyst_report_text="Overview\nMethods were reviewed separately.\nAdministrative note only.",
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 1
    assert comparison.metrics.total_human_findings == 0
    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.qtrial_only_count == 1
    assert comparison.metrics.human_only_count == 0
    assert comparison.matched_findings == []


def test_build_comparison_report_detects_agreement_contradiction_and_evidence_upgrade() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "endpoint": "mortality",
                "adjusted_p_value": 0.01,
                "effect_size": 0.42,
                "significant_after_correction": True,
            },
            {
                "finding_id": "smoking",
                "endpoint": "mortality",
                "adjusted_p_value": 0.20,
                "effect_size": 0.11,
                "significant_after_correction": False,
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="""
        Results:
        - Age was statistically significant for mortality with p=0.01 [1].
        - Smoking was statistically significant for mortality with p=0.02.
        - Albumin increased in the treatment arm.
        """,
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 2
    assert comparison.metrics.total_human_findings == 2
    assert comparison.metrics.matched_pairs == 2
    assert comparison.metrics.qtrial_only_count == 0
    assert comparison.metrics.human_only_count == 0
    assert comparison.metrics.agreement_count == 1
    assert comparison.metrics.contradiction_count == 1
    assert comparison.metrics.partial_agreement_count == 0
    assert comparison.metrics.recall_against_human == 1.0
    assert comparison.metrics.evidence_upgrade_rate == 0.5
    assert {match.relation for match in comparison.matched_findings} == {"agree", "contradict"}
    assert len(comparison.contradictions) == 1
    assert "smoking" in comparison.contradictions[0].rationale.lower()


def test_structured_matching_links_human_claim_to_qtrial_variable() -> None:
    report = _make_report(
        [
            {
                "finding_id": "ejection_fraction",
                "adjusted_p_value": 0.001,
                "effect_size": -0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Lower ejection fraction predicts mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    match = comparison.matched_findings[0]
    assert match.relation == "agree"
    assert match.matched_by in {"variable", "variable+endpoint"}
    assert match.qtrial_finding.variable == "ejection_fraction"
    assert match.human_finding.variable == "ejection_fraction"
    assert match.human_finding.endpoint == "mortality"
    assert match.text_used_for_matching["qtrial"]
    assert match.text_used_for_matching["human"] == "Lower ejection fraction predicts mortality."
    assert "ejection fraction" in match.rationale.lower()


def test_or_below_one_matches_lower_predictor_higher_mortality_claim() -> None:
    report = _make_report(
        [
            {
                "finding_id": "ejection_fraction",
                "finding_text_plain": (
                    "Higher ejection fraction was associated with lower odds of mortality "
                    "(OR 0.945, p=8e-6), consistent with lower ejection fraction indicating higher mortality risk."
                ),
                "endpoint": "mortality",
                "direction": "negative",
                "effect_size": 0.945,
                "effect_size_label": "odds_ratio",
                "odds_ratio": 0.945,
                "adjusted_p_value": 0.000008,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Lower ejection fraction was associated with higher mortality risk.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    match = comparison.matched_findings[0]
    assert match.relation == "agree"
    assert match.qtrial_finding.direction == "negative"
    assert match.human_finding.direction == "negative"


def test_structured_matching_detects_significance_contradiction() -> None:
    report = _make_report(
        [
            {
                "finding_id": "smoking",
                "adjusted_p_value": 0.21,
                "effect_size": 0.05,
                "significant_after_correction": False,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Smoking strongly predicts mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert comparison.metrics.contradiction_count == 1
    assert comparison.matched_findings[0].relation == "contradict"
    assert comparison.matched_findings[0].human_finding.variable == "smoking"
    assert "human report states that smoking predicts mortality" in comparison.matched_findings[0].rationale.lower()


def test_structured_matching_keeps_unmatched_human_claim_human_only() -> None:
    report = _make_report(
        [
            {
                "finding_id": "ejection_fraction",
                "adjusted_p_value": 0.001,
                "effect_size": -0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Platelets show no relationship with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.human_only_count == 1
    assert comparison.human_only_findings[0].variable == "platelets"


def test_different_non_significant_variables_do_not_partially_match() -> None:
    report = _make_report(
        [
            {
                "finding_id": "creatinine_phosphokinase",
                "finding_text_plain": "Creatinine phosphokinase did not show a statistically significant association with mortality.",
                "endpoint": "mortality",
                "adjusted_p_value": 0.41,
                "effect_size": 0.02,
                "significant_after_correction": False,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Platelet levels did not show a strong relationship with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.qtrial_only_count == 1
    assert comparison.metrics.human_only_count == 1
    assert comparison.human_only_findings[0].variable == "platelets"
    assert "did_not" not in str(comparison.human_only_findings[0].variable)


def test_human_claim_variable_normalization_strips_levels_and_verb_fragments() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        """
        Platelet levels did not show a strong relationship with mortality.
        Serum creatinine levels did not show a statistically significant association with mortality.
        """,
    )

    assert [finding.variable for finding in parsed.findings] == ["platelets", "serum_creatinine"]
    assert all("did_not" not in str(finding.variable) for finding in parsed.findings)


def test_same_endpoint_and_significance_variable_mismatch_prevents_match() -> None:
    report = _make_report(
        [
            {
                "finding_id": "diabetes",
                "finding_text_plain": "Diabetes was not significantly associated with mortality.",
                "endpoint": "mortality",
                "adjusted_p_value": 0.85,
                "effect_size": 0.01,
                "significant_after_correction": False,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Smoking was not significantly associated with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.qtrial_only_count == 1
    assert comparison.metrics.human_only_count == 1


def test_non_significant_finding_text_yields_not_significant_label() -> None:
    report = _make_report(
        grounded_findings=[
            GroundedFinding(
                finding_text="Serum creatinine did not show a statistically significant association with mortality.",
                finding_text_plain="Serum creatinine did not show a statistically significant association with mortality.",
                grounding_status="Supported",
            )
        ],
        corrected_findings=[],
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Serum creatinine did not show a statistically significant association with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    match = comparison.matched_findings[0]
    assert match.qtrial_finding.significance == "not_significant"
    assert match.qtrial_finding.significant is False


def test_comparison_prefers_plain_english_qtrial_text_when_available() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001; effect size=0.42; significant after correction",
                "finding_text_plain": "Age was significantly associated with the outcome.",
                "comparison_claim_text": "Older age was associated with higher mortality risk.",
                "endpoint": "mortality",
                "direction": "positive",
                "adjusted_p_value": 0.0001,
                "effect_size": 0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Older age was associated with higher mortality risk.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert comparison.matched_findings[0].qtrial_finding.finding_text == "Older age was associated with higher mortality risk."
    assert comparison.matched_findings[0].text_used_for_matching["qtrial"] == "Older age was associated with higher mortality risk."


def test_directional_age_comparison_claim_matches_human_age_statement() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "comparison_claim_text": "Older age was associated with higher mortality risk.",
                "finding_text_plain": "Age was significantly associated with mortality.",
                "endpoint": "mortality",
                "direction": "positive",
                "adjusted_p_value": 0.0001,
                "effect_size": 0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Older patients had a higher probability of death during follow-up.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert comparison.matched_findings[0].relation == "agree"


def test_comparison_falls_back_to_raw_qtrial_text_when_plain_text_missing() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_raw": "age; adjusted p=0.0001; effect size=0.42; significant after correction",
                "adjusted_p_value": 0.0001,
                "effect_size": 0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was significantly associated with the outcome.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert "adjusted p=0.0001" in comparison.matched_findings[0].text_used_for_matching["qtrial"]


def test_same_variable_with_incomplete_endpoint_alignment_is_partial_agree() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was significantly associated with the outcome.",
                "adjusted_p_value": 0.0001,
                "effect_size": 0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was significantly associated with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert comparison.matched_findings[0].relation in {"agree", "partial_agree"}
    if comparison.matched_findings[0].relation == "partial_agree":
        assert "only one report explicitly links it to mortality" in comparison.matched_findings[0].rationale.lower()


def test_qc_findings_are_excluded_from_comparison_metrics() -> None:
    report = _make_report(
        corrected_findings=[],
        grounded_findings=[
            GroundedFinding(
                finding_text="Age was significantly associated with mortality.",
                finding_text_plain="Age was significantly associated with mortality.",
                finding_category="analytical",
                grounding_status="Supported",
            ),
            GroundedFinding(
                finding_text="Duplicate rows were detected during the data integrity check.",
                finding_text_plain="Duplicate rows were detected during the data integrity check.",
                finding_category="data_quality",
                grounding_status="Data Quality Note",
                literature_skipped=True,
                literature_skip_note="Excluded from literature grounding.",
            ),
        ],
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was significantly associated with mortality.",
        provider="gemini",
    )

    assert len(report.grounded_findings.findings) == 2  # type: ignore[union-attr]
    assert comparison.metrics.total_qtrial_findings == 1
    assert comparison.metrics.matched_pairs == 1
    assert comparison.metrics.qtrial_only_count == 0
    assert all(
        match.qtrial_finding.finding_category == "analytical"
        for match in comparison.matched_findings
    )
    assert any(
        finding.finding_category == "data_quality"
        for finding in report.grounded_findings.findings  # type: ignore[union-attr]
    )


def test_serum_creatinine_matches_plain_creatinine_wording() -> None:
    report = _make_report(
        [
            {
                "finding_id": "serum_creatinine",
                "finding_text_plain": "Serum creatinine was significantly associated with mortality.",
                "adjusted_p_value": 0.002,
                "effect_size": 0.33,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Creatinine was significantly associated with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert comparison.matched_findings[0].relation == "agree"
    assert comparison.matched_findings[0].qtrial_finding.variable == "serum_creatinine"
    assert comparison.matched_findings[0].human_finding.variable == "serum_creatinine"


def test_survival_primary_note_does_not_match_clinical_variable_claim() -> None:
    report = _make_report(
        [
            {
                "finding_id": "survival_primary",
                "finding_text_plain": "Survival was significantly associated with mortality.",
                "endpoint": "mortality",
                "finding_category": "statistical_note",
                "claim_type": "statistical_note",
                "adjusted_p_value": 0.01,
                "effect_size": 0.0,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Mortality was significantly associated with the survival outcome.",
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 0
    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.total_human_findings == 0
    assert comparison.metrics.human_only_count == 0


def test_duplicate_key_qc_finding_cannot_match_human_mortality_statement() -> None:
    report = _make_report(
        corrected_findings=[],
        grounded_findings=[
            GroundedFinding(
                finding_text="Key-column duplicates (age + time + DEATH_EVENT) suggest repeated-row integrity issues.",
                finding_text_plain="Key-column duplicates (age + time + DEATH_EVENT) suggest repeated-row integrity issues.",
                finding_category="data_quality",
                grounding_status="Data Quality Note",
                literature_skipped=True,
                literature_skip_note="Excluded from literature grounding.",
            ),
        ],
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Mortality was significantly associated with age.",
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 0
    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.human_only_count == 1


def test_framework_artifact_finding_is_excluded_from_comparison() -> None:
    report = _make_report(
        corrected_findings=[],
        grounded_findings=[
            GroundedFinding(
                finding_text="ANCOVA: adjusted treatment p=0.04 after baseline adjustment.",
                finding_text_plain="ANCOVA: adjusted treatment p=0.04 after baseline adjustment.",
                finding_category="qc_note",
                grounding_status="QC Observation",
                literature_skipped=True,
                literature_skip_note="Excluded from literature grounding.",
            ),
        ],
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was significantly associated with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 0
    assert comparison.metrics.matched_pairs == 0


def test_only_association_claims_enter_comparison_pool() -> None:
    report = _make_report(
        corrected_findings=[],
        grounded_findings=[
            GroundedFinding(
                finding_text="Age was significantly associated with mortality.",
                finding_text_plain="Age was significantly associated with mortality.",
                finding_category="analytical",
                claim_type="association_claim",
                grounding_status="Supported",
            ),
            GroundedFinding(
                finding_text="The event rate was 32.1% during follow-up.",
                finding_text_plain="The event rate was 32.1% during follow-up.",
                finding_category="endpoint_result",
                claim_type="descriptive_claim",
                grounding_status="Supported",
            ),
            GroundedFinding(
                finding_text="Digit preference was detected in age values.",
                finding_text_plain="Digit preference was detected in age values.",
                finding_category="data_quality",
                claim_type="data_quality_claim",
                grounding_status="Data Quality Note",
                literature_skipped=True,
            ),
        ],
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="""
        Age was significantly associated with mortality.
        The event rate was 30%.
        """,
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 1
    assert comparison.metrics.total_human_findings == 1
    assert comparison.metrics.matched_pairs == 1
    assert comparison.metrics.qtrial_only_count == 0
    assert comparison.metrics.human_only_count == 0


def test_descriptive_and_metadata_human_claims_do_not_match_result_findings() -> None:
    report = _make_report(
        [
            {
                "finding_id": "ejection_fraction",
                "finding_text_plain": "Ejection fraction was significantly associated with mortality.",
                "adjusted_p_value": 0.001,
                "effect_size": -0.42,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="""
        The dataset comprises 299 patients with chronic heart failure.
        Overall median survival was 209 days.
        """,
        provider="gemini",
    )

    assert comparison.metrics.total_human_findings == 0
    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.qtrial_only_count == 1


def test_followup_time_context_is_not_an_analytical_human_claim() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was significantly associated with mortality.",
                "endpoint": "mortality",
                "adjusted_p_value": 0.001,
                "effect_size": 0.4,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Median follow-up time was 18.4 months.",
        provider="gemini",
    )

    assert comparison.metrics.total_human_findings == 0
    assert comparison.metrics.matched_pairs == 0
    assert comparison.metrics.qtrial_only_count == 1


def test_mcc_is_near_perfect_for_perfect_binary_significance_agreement() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was significantly associated with mortality.",
                "adjusted_p_value": 0.001,
                "effect_size": 0.4,
                "significant_after_correction": True,
            },
            {
                "finding_id": "smoking",
                "finding_text_plain": "Smoking did not show a statistically significant association with mortality.",
                "adjusted_p_value": 0.4,
                "effect_size": 0.02,
                "significant_after_correction": False,
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="""
        Age was significantly associated with mortality.
        Smoking did not show a statistically significant association with mortality.
        """,
        provider="gemini",
    )

    assert comparison.metrics.mcc == 1.0
    assert comparison.metrics.mcc_interpretation == "near-perfect agreement"


def test_mcc_returns_zero_for_balanced_mixed_agreement_and_contradiction() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was significantly associated with mortality.",
                "adjusted_p_value": 0.001,
                "effect_size": 0.4,
                "significant_after_correction": True,
            },
            {
                "finding_id": "smoking",
                "finding_text_plain": "Smoking did not show a statistically significant association with mortality.",
                "adjusted_p_value": 0.4,
                "effect_size": 0.02,
                "significant_after_correction": False,
            },
            {
                "finding_id": "anaemia",
                "finding_text_plain": "Anaemia was significantly associated with mortality.",
                "adjusted_p_value": 0.01,
                "effect_size": 0.2,
                "significant_after_correction": True,
            },
            {
                "finding_id": "diabetes",
                "finding_text_plain": "Diabetes did not show a statistically significant association with mortality.",
                "adjusted_p_value": 0.3,
                "effect_size": 0.03,
                "significant_after_correction": False,
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="""
        Age was significantly associated with mortality.
        Smoking did not show a statistically significant association with mortality.
        Anaemia did not show a statistically significant association with mortality.
        Diabetes was significantly associated with mortality.
        """,
        provider="gemini",
    )

    assert comparison.metrics.mcc == 0.0
    assert comparison.metrics.mcc_interpretation == "poor agreement"


def test_mcc_returns_null_for_degenerate_case() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was significantly associated with mortality.",
                "adjusted_p_value": 0.001,
                "effect_size": 0.4,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was significantly associated with mortality.",
        provider="gemini",
    )

    assert comparison.metrics.mcc is None
    assert comparison.metrics.mcc_interpretation is None
    assert comparison.metrics.mcc_explanation is not None
    assert "true negatives" in comparison.metrics.mcc_explanation


def test_partial_agree_is_excluded_from_mcc() -> None:
    partial_match = FindingMatch(
        qtrial_finding=ComparableFinding(
            finding_id="q_age",
            source="qtrial",
            source_label="clinical_analysis",
            finding_text="Age was significantly associated with mortality.",
            normalized_text="age significantly associated mortality",
            variable="age",
            endpoint="mortality",
            significant=True,
            significance="significant",
        ),
        human_finding=ComparableFinding(
            finding_id="h_age",
            source="human",
            source_label="analyst.txt",
            finding_text="Age was significantly associated with progression-free survival.",
            normalized_text="age significantly associated progression free survival",
            variable="age",
            endpoint="progression_free_survival",
            significant=True,
            significance="significant",
        ),
        relation="partial_agree",
        match_score=0.7,
    )

    metrics = _build_metrics(
        qtrial_findings=[partial_match.qtrial_finding],
        human_findings=[partial_match.human_finding],
        matches=[partial_match],
        qtrial_only=[],
        human_only=[],
    )

    assert metrics.partial_agreement_count == 1
    assert metrics.mcc is None
    assert metrics.mcc_interpretation is None
