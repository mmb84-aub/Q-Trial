from qtrial_backend.agentic.report_comparison import (
    _candidate_match_score,
    _compare_statistical_evidence,
    _build_metrics,
    build_comparison_report,
    normalize_qtrial_findings,
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
    StatisticalEvidence,
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


def _finding_with_evidence(
    finding_id: str,
    evidence: StatisticalEvidence,
    source: str = "qtrial",
) -> ComparableFinding:
    return ComparableFinding(
        finding_id=finding_id,
        source=source,  # type: ignore[arg-type]
        source_label=source,
        finding_text=f"{finding_id} statistical finding",
        normalized_text=f"{finding_id} statistical finding",
        variable=evidence.variable,
        endpoint=evidence.endpoint,
        significant=evidence.significant,
        significance="significant" if evidence.significant is True else "not_significant" if evidence.significant is False else "unclear",
        p_value=evidence.p_value,
        effect_size=evidence.effect_size,
        effect_size_label=evidence.effect_size_label,
        statistical_evidence=evidence,
    )


def test_parse_human_report_uses_current_section_header() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        """
        Results:
        - Age was statistically significant for mortality with p=0.01 [1].
        Discussion:
        - Smoking was not significantly associated with mortality (p=0.20).
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


def test_human_parser_extracts_or_p_value_and_ci() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        "Age was associated with mortality (OR 2.1, 95% CI 1.4-3.2, p = 0.01).",
        known_variables={"age"},
    )

    evidence = parsed.findings[0].statistical_evidence
    assert evidence is not None
    assert evidence.effect_size == 2.1
    assert evidence.effect_size_label == "odds_ratio"
    assert evidence.p_value == 0.01
    assert evidence.ci_lower == 1.4
    assert evidence.ci_upper == 3.2
    assert evidence.significant is True


def test_human_parser_extracts_hr_ci_and_bounded_p_value() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        "Serum creatinine predicted mortality (HR 0.72, 95% CI 0.55 to 0.93, p<0.001).",
        known_variables={"serum_creatinine"},
    )

    evidence = parsed.findings[0].statistical_evidence
    assert evidence is not None
    assert evidence.effect_size == 0.72
    assert evidence.effect_size_label == "hazard_ratio"
    assert evidence.p_operator == "<"
    assert evidence.p_value == 0.001
    assert evidence.ci_lower == 0.55
    assert evidence.ci_upper == 0.93


def test_human_parser_extracts_cohens_d_and_correlation() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        """
        Age differed between treatment groups (Cohen's d = -0.60).
        Serum sodium correlated with mortality risk (Spearman ρ=-0.54).
        """,
        known_variables={"age", "serum_sodium"},
    )

    labels = [finding.statistical_evidence.effect_size_label for finding in parsed.findings if finding.statistical_evidence]
    values = [finding.statistical_evidence.effect_size for finding in parsed.findings if finding.statistical_evidence]
    assert "cohen_d" in labels
    assert "correlation" in labels
    assert -0.60 in values
    assert -0.54 in values


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


def test_pairing_confidence_does_not_use_p_values() -> None:
    qfinding = ComparableFinding(
        finding_id="q_age",
        source="qtrial",
        source_label="qtrial",
        finding_text="Age was associated with mortality.",
        normalized_text="age associated mortality",
        variable="age",
        endpoint="mortality",
        claim_type="association_claim",
        p_value=0.000001,
    )
    hfinding = ComparableFinding(
        finding_id="h_age",
        source="human",
        source_label="analyst.txt",
        finding_text="Age was associated with mortality.",
        normalized_text="age associated mortality",
        variable="age",
        endpoint="mortality",
        claim_type="association_claim",
        p_value=0.93,
    )
    same_score = _candidate_match_score(qfinding, hfinding)

    hfinding.p_value = 0.000001
    assert _candidate_match_score(qfinding, hfinding) == same_score

    hfinding.variable = "smoking"
    hfinding.p_value = 0.000001
    assert _candidate_match_score(qfinding, hfinding) == 0.0


def test_paired_statistical_comparison_marks_close_ors_as_strong_agreement() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was associated with mortality (OR 2.28, p=2.9e-5).",
                "endpoint": "mortality",
                "odds_ratio": 2.28,
                "adjusted_p_value": 2.9e-5,
                "effect_size_label": "odds_ratio",
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was associated with mortality (OR 2.1, p = 0.01).",
        provider="gemini",
    )

    match = comparison.matched_findings[0]
    assert match.pairing_confidence == match.match_score
    assert match.relation == "agree"
    assert match.statistical_comparison is not None
    assert match.statistical_comparison.available is True
    assert match.statistical_comparison.agreement_label == "strong"
    assert match.statistical_comparison.effect_size_agreement == "close"
    assert match.statistical_comparison.p_value_agreement == "different_strength"
    assert match.statistical_comparison.p_value_log_delta is not None
    assert match.statistical_comparison.statistical_agreement_score == match.statistical_comparison.overall_statistical_agreement_score
    assert match.statistical_comparison.statistical_agreement_coverage == match.statistical_comparison.coverage_score


def test_paired_statistical_comparison_marks_opposite_or_side_as_contradiction() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was associated with mortality (OR 0.70, p=0.01).",
                "endpoint": "mortality",
                "odds_ratio": 0.70,
                "adjusted_p_value": 0.01,
                "effect_size_label": "odds_ratio",
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was associated with mortality (OR 2.1, p = 0.01).",
        provider="gemini",
    )

    match = comparison.matched_findings[0]
    assert match.relation == "contradict"
    assert match.statistical_comparison is not None
    assert match.statistical_comparison.agreement_label == "contradiction"
    assert match.statistical_comparison.effect_size_agreement == "opposite"


def test_or_opposite_sides_of_null_are_statistical_contradiction() -> None:
    qev = StatisticalEvidence(
        variable="biomarker",
        endpoint="mortality",
        effect_size=0.7,
        effect_size_label="odds_ratio",
        significant=True,
    )
    hev = StatisticalEvidence(
        variable="biomarker",
        endpoint="mortality",
        effect_size=1.4,
        effect_size_label="odds_ratio",
        significant=True,
    )

    comparison = _compare_statistical_evidence(
        _finding_with_evidence("q_biomarker", qev),
        _finding_with_evidence("h_biomarker", hev, source="human"),
    )

    assert comparison.agreement_label == "contradiction"
    assert comparison.effect_size_agreement == "opposite"


def test_same_significance_but_far_effect_size_becomes_partial_agreement() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was associated with mortality (HR 2.8, p=1e-5).",
                "endpoint": "mortality",
                "effect_size": 2.8,
                "effect_size_label": "hazard_ratio",
                "adjusted_p_value": 1e-5,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was associated with mortality (HR 1.1, p = 0.04).",
        provider="gemini",
    )

    match = comparison.matched_findings[0]
    assert match.relation == "partial_agree"
    assert match.statistical_comparison is not None
    assert match.statistical_comparison.effect_size_agreement == "far"
    assert match.statistical_comparison.significance_agreement == "agree"


def test_missing_human_numbers_keeps_semantic_match_but_statistical_not_assessed() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was significantly associated with mortality.",
                "endpoint": "mortality",
                "effect_size": 0.4,
                "adjusted_p_value": 0.001,
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

    match = comparison.matched_findings[0]
    assert match.relation == "agree"
    assert match.statistical_comparison is not None
    assert match.statistical_comparison.available is False
    assert match.statistical_comparison.agreement_label == "not_assessed"
    assert "human report did not provide numeric" in match.statistical_comparison.reason_if_unavailable


def test_p_value_comparison_uses_log_scale() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was associated with mortality (OR 2.0, p=0.001).",
                "endpoint": "mortality",
                "odds_ratio": 2.0,
                "adjusted_p_value": 0.001,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was associated with mortality (OR 2.0, p = 0.04).",
        provider="gemini",
    )

    stat = comparison.matched_findings[0].statistical_comparison
    assert stat is not None
    assert stat.p_value_agreement == "different_strength"
    assert stat.p_value_log_delta is not None
    assert stat.p_value_log_delta > 1.0


def test_missing_human_effect_size_and_ci_lower_statistical_coverage_without_fake_agreement() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was associated with mortality (OR 2.0, 95% CI 1.2-3.1, p=0.01).",
                "endpoint": "mortality",
                "odds_ratio": 2.0,
                "adjusted_p_value": 0.01,
                "significant_after_correction": True,
                "metadata": {"ci_lower": 1.2, "ci_upper": 3.1},
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was significantly associated with mortality (p = 0.02).",
        provider="gemini",
    )

    stat = comparison.matched_findings[0].statistical_comparison
    assert stat is not None
    assert stat.available is True
    assert stat.effect_size_agreement == "not_assessed"
    assert stat.ci_agreement == "not_assessed"
    assert 0 < stat.statistical_agreement_coverage < 1


def test_ci_overlap_and_null_exclusion_logic_is_reported() -> None:
    report = _make_report(
        [
            {
                "finding_id": "age",
                "finding_text_plain": "Age was associated with mortality (OR 2.0, 95% CI 1.2-3.1, p=0.01).",
                "endpoint": "mortality",
                "odds_ratio": 2.0,
                "adjusted_p_value": 0.01,
                "significant_after_correction": True,
                "metadata": {"ci_lower": 1.2, "ci_upper": 3.1},
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Age was associated with mortality (OR 2.2, 95% CI 1.5 to 2.9, p = 0.02).",
        provider="gemini",
    )

    stat = comparison.matched_findings[0].statistical_comparison
    assert stat is not None
    assert stat.ci_agreement == "agree"
    assert stat.ci_overlap is True


def test_ci_null_side_disagreement_is_partial_not_fake_agreement() -> None:
    qev = StatisticalEvidence(
        variable="age",
        endpoint="mortality",
        effect_size=2.0,
        effect_size_label="odds_ratio",
        ci_lower=1.2,
        ci_upper=3.1,
        significant=True,
    )
    hev = StatisticalEvidence(
        variable="age",
        endpoint="mortality",
        effect_size=1.4,
        effect_size_label="odds_ratio",
        ci_lower=0.8,
        ci_upper=2.0,
        significant=True,
    )

    comparison = _compare_statistical_evidence(
        _finding_with_evidence("q_age", qev),
        _finding_with_evidence("h_age", hev, source="human"),
    )

    assert comparison.ci_agreement == "partial"
    assert comparison.ci_overlap is True
    assert comparison.statistical_agreement_score is not None
    assert comparison.statistical_agreement_score < 1.0


def test_ci_non_overlap_same_null_side_is_weak() -> None:
    qev = StatisticalEvidence(
        variable="age",
        endpoint="mortality",
        effect_size=2.0,
        effect_size_label="odds_ratio",
        ci_lower=1.1,
        ci_upper=1.5,
        significant=True,
    )
    hev = StatisticalEvidence(
        variable="age",
        endpoint="mortality",
        effect_size=2.4,
        effect_size_label="odds_ratio",
        ci_lower=1.8,
        ci_upper=3.0,
        significant=True,
    )

    comparison = _compare_statistical_evidence(
        _finding_with_evidence("q_age", qev),
        _finding_with_evidence("h_age", hev, source="human"),
    )

    assert comparison.ci_agreement == "weak"
    assert comparison.ci_overlap is False


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


def test_inverse_direction_phrasings_normalize_to_same_endpoint_effect() -> None:
    report = _make_report(
        [
            {
                "finding_id": "ejection_fraction",
                "finding_text_plain": "Higher ejection fraction lowered mortality risk (OR 0.95, p=0.01).",
                "endpoint": "mortality",
                "direction": "unknown",
                "effect_size": 0.95,
                "effect_size_label": "odds_ratio",
                "adjusted_p_value": 0.01,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Lower ejection fraction increased mortality risk.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    match = comparison.matched_findings[0]
    assert match.relation == "agree"
    assert match.qtrial_finding.direction == "negative"
    assert match.human_finding.direction == "negative"
    assert match.qtrial_finding.statistical_evidence.direction_effect_on_endpoint == "decreases_endpoint_risk"
    assert match.human_finding.statistical_evidence.direction_effect_on_endpoint == "decreases_endpoint_risk"


def test_same_predictor_opposite_endpoint_effect_is_contradiction() -> None:
    report = _make_report(
        [
            {
                "finding_id": "biomarker_x",
                "finding_text_plain": "Higher biomarker x lowered endpoint risk (OR 0.70, p=0.01).",
                "endpoint": "primary_outcome",
                "direction": "unknown",
                "effect_size": 0.70,
                "effect_size_label": "odds_ratio",
                "adjusted_p_value": 0.01,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Higher biomarker x increased endpoint risk.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 1
    assert comparison.matched_findings[0].relation == "contradict"


def test_raw_chi_square_artifact_is_excluded_even_if_upstream_labeled_analytical() -> None:
    report = _make_report(
        [
            {
                "finding_id": "platelets",
                "finding_text_raw": "platelets: χ²=2232.8060, p=0.0000",
                "finding_text_plain": "platelets: χ²=2232.8060, p=0.0000",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "adjusted_p_value": 0.0,
                "effect_size": 2232.806,
                "significant_after_correction": True,
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Platelets were associated with the endpoint.",
        provider="gemini",
    )

    assert comparison.metrics.total_qtrial_findings == 0
    assert comparison.qtrial_only_findings == []


def test_raw_chi_square_artifacts_are_excluded_from_comparison_but_clinical_negatives_remain() -> None:
    report = _make_report(
        [
            {
                "finding_id": "time",
                "finding_text_raw": "`time`: χ²=38.4916, p=0.0000",
                "finding_text_plain": "`time`: χ²=38.4916, p=0.0000",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "adjusted_p_value": 0.0,
                "effect_size": 38.4916,
                "significant_after_correction": True,
            },
            {
                "finding_id": "smoking",
                "finding_text_raw": "`smoking`: χ²=1387.4548, p=0.0000",
                "finding_text_plain": "`smoking`: χ²=1387.4548, p=0.0000",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "adjusted_p_value": 0.0,
                "effect_size": 1387.4548,
                "significant_after_correction": True,
            },
            {
                "finding_id": "smoking",
                "finding_text_plain": "Smoking was not significantly associated with mortality (p=0.41).",
                "endpoint": "mortality",
                "finding_category": "analytical",
                "claim_type": "negative_association",
                "adjusted_p_value": 0.41,
                "effect_size": 0.02,
                "significant_after_correction": False,
            },
            {
                "finding_id": "platelets",
                "finding_text_plain": "Platelets did not show statistically significant association with mortality.",
                "endpoint": "mortality",
                "finding_category": "analytical",
                "claim_type": "negative_association",
                "adjusted_p_value": 0.72,
                "effect_size": 0.01,
                "significant_after_correction": False,
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Smoking was not significantly associated with mortality.",
        provider="gemini",
    )

    all_comparison_text = " ".join(
        [
            *(match.qtrial_finding.finding_text for match in comparison.matched_findings),
            *(match.human_finding.finding_text for match in comparison.matched_findings),
            *(finding.finding_text for finding in comparison.qtrial_only_findings),
            *(finding.finding_text for finding in comparison.human_only_findings),
        ]
    )
    assert "χ²" not in all_comparison_text
    assert "1387.4548" not in all_comparison_text
    assert comparison.metrics.total_qtrial_findings == 2
    assert comparison.metrics.matched_pairs == 1
    assert comparison.metrics.qtrial_only_count == 1
    assert comparison.matched_findings[0].qtrial_finding.variable == "smoking"
    assert comparison.qtrial_only_findings[0].variable == "platelets"


def test_header_wrapper_is_excluded_from_comparison_and_cannot_be_contradiction() -> None:
    report = _make_report(
        [
            {
                "finding_id": "hazard_ratios",
                "finding_text_plain": "Hazard Ratios (HR with 95% CI):",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "significant_after_correction": True,
                "adjusted_p_value": 0.001,
            },
            {
                "finding_id": "serum_creatinine",
                "finding_text_plain": "Higher serum creatinine was associated with higher mortality risk.",
                "endpoint": "mortality",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "direction": "positive",
                "significant_after_correction": True,
                "adjusted_p_value": 0.01,
                "effect_size": 1.7,
                "effect_size_label": "hazard_ratio",
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Older patients had significantly higher probability of death during follow-up.",
        provider="gemini",
    )

    all_comparison_text = " ".join(
        [
            *(match.qtrial_finding.finding_text for match in comparison.matched_findings),
            *(match.human_finding.finding_text for match in comparison.matched_findings),
            *(finding.finding_text for finding in comparison.qtrial_only_findings),
            *(finding.finding_text for finding in comparison.human_only_findings),
        ]
    )
    assert "Hazard Ratios" not in all_comparison_text
    assert comparison.contradictions == []
    assert all(
        "Hazard Ratios" not in match.qtrial_finding.finding_text
        for match in comparison.matched_findings
    )


def test_ratio_direction_parsing_uses_hr_below_one_as_lower_endpoint_risk() -> None:
    report = _make_report(
        [
            {
                "finding_id": "biomarker_x",
                "finding_text_plain": "Each 1% increase in biomarker x reduces mortality hazard by 4.7% (HR=0.953).",
                "endpoint": "mortality",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "direction": "unknown",
                "significant_after_correction": True,
                "effect_size": 0.953,
                "effect_size_label": "hazard_ratio",
            }
        ]
    )

    finding = normalize_qtrial_findings(report)[0]
    assert finding.direction == "negative"
    assert finding.statistical_evidence is not None
    assert finding.statistical_evidence.direction_effect_on_endpoint == "decreases_endpoint_risk"


def test_ratio_direction_parsing_uses_hr_above_one_as_higher_endpoint_risk() -> None:
    report = _make_report(
        [
            {
                "finding_id": "biomarker_x",
                "finding_text_plain": "Each unit increase in biomarker x was associated with mortality hazard (HR=1.476).",
                "endpoint": "mortality",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "direction": "unknown",
                "significant_after_correction": True,
                "effect_size": 1.476,
                "effect_size_label": "hazard_ratio",
            }
        ]
    )

    finding = normalize_qtrial_findings(report)[0]
    assert finding.direction == "positive"
    assert finding.statistical_evidence is not None
    assert finding.statistical_evidence.direction_effect_on_endpoint == "increases_endpoint_risk"


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
    assert comparison.matched_findings[0].text_used_for_matching["qtrial"] == (
        "Age was significantly associated with the outcome."
    )
    assert comparison.matched_findings[0].qtrial_finding.p_value == 0.0001


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


def test_qtrial_only_excludes_malformed_and_contextual_findings() -> None:
    report = _make_report(
        [
            {
                "finding_id": "bad_fragment",
                "finding_text_plain": "1.18 mg/dL (p<0.001, adjusted p=0.00014)",
                "finding_category": "analytical",
                "claim_type": "association_claim",
            },
            {
                "finding_id": "qubo",
                "finding_text_plain": "The dataset includes 8 predictor variables selected via QUBO feature selection.",
                "finding_category": "analytical",
                "claim_type": "association_claim",
            },
            {
                "finding_id": "serum_creatinine",
                "finding_text_plain": "Higher serum creatinine was associated with higher mortality risk.",
                "finding_category": "analytical",
                "claim_type": "association_claim",
                "variable": "serum_creatinine",
                "endpoint": "mortality",
                "significant_after_correction": True,
                "adjusted_p_value": 0.001,
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="",
        provider="gemini",
    )

    assert comparison.metrics.qtrial_only_count == 1
    assert comparison.qtrial_only_findings[0].variable == "serum_creatinine"
    assert all("QUBO" not in finding.finding_text for finding in comparison.qtrial_only_findings)


def test_qtrial_only_excludes_continuation_wrappers_and_context_stats() -> None:
    report = _make_report(
        [
            {
                "finding_id": "fragment",
                "finding_text_plain": "1.18 mg/dL, 56% increase), indicating renal dysfunction is strongly associated with mortality.",
                "finding_category": "analytical",
                "claim_type": "association_claim",
            },
            {
                "finding_id": "interpretation",
                "finding_text_plain": "- **Interpretation:** No association with mortality (41.7% diabetic in both groups).",
                "finding_category": "analytical",
                "claim_type": "association_claim",
            },
            {
                "finding_id": "event_rate",
                "finding_text_plain": "**Event rate:** 32.1% (96 deaths out of 299 patients).",
                "finding_category": "analytical",
                "claim_type": "association_claim",
            },
            {
                "finding_id": "smoking",
                "finding_text_plain": "Smoking was not significantly associated with mortality.",
                "finding_category": "analytical",
                "claim_type": "negative_association",
                "variable": "smoking",
                "endpoint": "mortality",
                "significant_after_correction": False,
                "adjusted_p_value": 0.41,
            },
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="",
        provider="gemini",
    )

    assert [finding.finding_text for finding in comparison.qtrial_only_findings] == [
        "Smoking was not significantly associated with mortality."
    ]


def test_fragment_cannot_be_matched_to_age_finding() -> None:
    report = _make_report(
        [
            {
                "finding_id": "fragment",
                "finding_text_plain": "1.18 mg/dL, 56% increase), indicating renal dysfunction is strongly associated with mortality.",
                "finding_category": "analytical",
                "claim_type": "association_claim",
            }
        ]
    )

    comparison = build_comparison_report(
        final_report=report,
        analyst_report_name="analyst.txt",
        analyst_report_text="Older patients had significantly higher probability of death during follow-up.",
        provider="gemini",
    )

    assert comparison.metrics.matched_pairs == 0
    assert comparison.qtrial_only_findings == []


def test_human_parser_drops_generic_variable_artifacts() -> None:
    parsed = parse_human_report_text(
        "analyst.txt",
        "These variables were significantly associated with mortality (p<0.01).",
    )

    assert parsed.findings == []


def test_build_metrics_counts_partial_agreement() -> None:
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
    assert metrics.agreement_count == 0
    assert metrics.contradiction_count == 0
