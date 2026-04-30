from qtrial_backend.agentic.finding_categories import (
    classify_claim_type,
    classify_finding_category,
    is_user_facing_clinical_finding_eligible,
    is_non_finding_header_artifact,
    is_malformed_finding_fragment,
    is_raw_stat_artifact_finding,
    is_user_facing_nonfinding_artifact,
    neutral_status_for_category,
)


def test_duplicate_row_finding_is_data_quality() -> None:
    assert (
        classify_finding_category("Duplicate rows were detected during the data integrity check.")
        == "data_quality_note"
    )


def test_imputation_note_is_preprocessing() -> None:
    assert (
        classify_finding_category("MICE imputation was applied to missing laboratory values before analysis.")
        == "preprocessing"
    )


def test_variable_outcome_finding_remains_analytical() -> None:
    assert (
        classify_finding_category(
            "Age was significantly associated with mortality.",
            variable="age",
            endpoint="mortality",
            analysis_type="association",
        )
        == "analytical"
    )


def test_qc_categories_map_to_neutral_status_labels() -> None:
    assert neutral_status_for_category("data_quality") == "Data Quality Note"
    assert neutral_status_for_category("preprocessing") == "Preprocessing Observation"
    assert neutral_status_for_category("pipeline_warning") == "Pipeline Warning"
    assert neutral_status_for_category("qc_note") == "QC Observation"


def test_key_duplicate_finding_is_data_quality() -> None:
    assert (
        classify_finding_category("Key-column duplicates (age + time + DEATH_EVENT) suggest repeated-row or record-integrity issues.")
        == "data_quality_note"
    )


def test_framework_artifact_is_not_analytical() -> None:
    assert (
        classify_finding_category("ANCOVA: adjusted treatment p=0.04 after baseline adjustment.")
        == "qc_note"
    )


def test_association_claim_type_for_variable_finding() -> None:
    assert classify_claim_type(
        "Age was significantly associated with mortality.",
        finding_category="analytical",
        variable="age",
        endpoint="mortality",
        significant=True,
        p_value=0.01,
    ) == "analytical_association"


def test_descriptive_claim_type_for_event_rate() -> None:
    assert classify_claim_type("The event rate was 32.1% during follow-up.") == "descriptive_context"


def test_descriptive_claim_type_for_median_survival() -> None:
    assert classify_claim_type("Overall median survival was 209 days.") == "descriptive_context"


def test_descriptive_claim_type_for_colon_median_survival_summary() -> None:
    assert classify_claim_type("Overall median survival: 209.0") == "descriptive_context"


def test_data_quality_claim_type_for_digit_preference() -> None:
    assert classify_claim_type("Digit preference was detected in age values.", finding_category="data_quality_note") == "data_quality_note"


def test_setup_claim_type_for_survival_setup_line() -> None:
    assert classify_claim_type(
        "Survival Analysis (time=`time`, event=`DEATH_EVENT`).",
        finding_category="pipeline_warning",
    ) == "setup_claim"


def test_metadata_claim_type_for_cohort_description() -> None:
    assert classify_claim_type("The dataset comprises 299 patients with chronic heart failure.") == "metadata_claim"


def test_followup_time_association_is_statistical_note() -> None:
    assert (
        classify_finding_category(
            "time was significantly associated with mortality.",
            variable="time",
            endpoint="mortality",
            analysis_type="association",
        )
        == "statistical_note"
    )


def test_raw_chi_square_artifacts_are_statistical_notes() -> None:
    assert classify_finding_category("time: χ²=38.4916, p=0.0000") == "statistical_note"
    assert classify_finding_category("smoking: χ²=0.0074, p=0.9316") == "statistical_note"
    assert classify_finding_category("platelets: chi-square=14.2, p=0.002") == "statistical_note"
    assert is_raw_stat_artifact_finding("`time`: χ²=38.4916, p=0.0000")
    assert is_raw_stat_artifact_finding("`smoking`: χ²=1387.4548, p=0.0000")
    assert is_raw_stat_artifact_finding("creatinine_phosphokinase χ²=69.9298, p=0.0000")


def test_clinical_negative_finding_is_not_raw_chi_square_artifact() -> None:
    assert (
        classify_finding_category(
            "Smoking was not significantly associated with mortality (χ² p=0.932).",
            variable="smoking",
            endpoint="mortality",
            analysis_type="association",
        )
        == "analytical"
    )
    assert not is_raw_stat_artifact_finding("Smoking was not significantly associated with mortality (p=0.41).")
    assert not is_raw_stat_artifact_finding("Platelets did not show statistically significant association with mortality.")


def test_statistical_headers_and_wrappers_are_non_finding_artifacts() -> None:
    assert is_non_finding_header_artifact("Hazard Ratios (HR with 95% CI):")
    assert is_non_finding_header_artifact("Effect Sizes (Cohen's d with 95% bootstrap CI):")
    assert is_non_finding_header_artifact("Test_Selection_Rationale:")
    assert is_non_finding_header_artifact("All continuous variables are non-normal (p<0.0001)")
    assert is_non_finding_header_artifact("**Binary outcome:**")
    assert is_user_facing_nonfinding_artifact("Hazard Ratios (HR with 95% CI):")


def test_real_clinical_findings_are_not_header_artifacts() -> None:
    assert not is_non_finding_header_artifact("Higher serum creatinine was associated with higher mortality risk.")
    assert not is_non_finding_header_artifact("Higher ejection fraction was associated with lower mortality risk.")
    assert not is_non_finding_header_artifact("Platelets did not show statistically significant association with mortality.")
    assert not is_non_finding_header_artifact("Smoking was not significantly associated with mortality.")


def test_malformed_fragments_are_user_facing_artifacts() -> None:
    assert is_malformed_finding_fragment("Deaths occurred much earlier (median 44.5 days vs.")
    assert is_malformed_finding_fragment("1.18 mg/dL (p<0.001, adjusted p=0.00014)")
    assert is_malformed_finding_fragment("Compared with the control group,")
    assert is_user_facing_nonfinding_artifact("Deaths occurred much earlier (median 44.5 days vs.")


def test_real_clinical_findings_are_not_malformed_fragments() -> None:
    assert not is_malformed_finding_fragment("Higher serum creatinine was associated with higher mortality risk.")
    assert not is_malformed_finding_fragment("Platelets did not show statistically significant association with mortality.")


def test_metadata_and_qc_context_are_demoted_from_analytical() -> None:
    assert (
        classify_finding_category("The dataset includes 8 predictor variables selected via QUBO feature selection.")
        == "preprocessing"
    )
    assert (
        classify_finding_category("All continuous variables rejected normality (p<0.001).")
        == "statistical_note"
    )
    assert classify_finding_category("The study design was retrospective observational.") == "qc_note"


def test_strict_clinical_finding_eligibility_rejects_fragments_wrappers_and_context() -> None:
    assert not is_user_facing_clinical_finding_eligible(
        "1.18 mg/dL, 56% increase), indicating renal dysfunction is strongly associated with mortality."
    )
    assert not is_user_facing_clinical_finding_eligible(
        "- **Interpretation:** No association with mortality (41.7% diabetic in both groups)."
    )
    assert not is_user_facing_clinical_finding_eligible(
        "**Event rate:** 32.1% (96 deaths out of 299 patients)."
    )
    assert not is_user_facing_clinical_finding_eligible("Primary analysis results:")
    assert not is_user_facing_clinical_finding_eligible("Characteristics by survival status:")
    assert not is_user_facing_clinical_finding_eligible("Prevalence of risk factors:")
    assert not is_user_facing_clinical_finding_eligible("Independent predictors:")


def test_strict_clinical_finding_eligibility_keeps_standalone_findings() -> None:
    assert is_user_facing_clinical_finding_eligible(
        "Serum creatinine was higher in patients who died (1.84 vs 1.18 mg/dL, p<0.001)."
    )
    assert is_user_facing_clinical_finding_eligible(
        "Higher serum creatinine was associated with higher mortality."
    )
    assert is_user_facing_clinical_finding_eligible(
        "Higher ejection fraction was associated with lower mortality."
    )
    assert is_user_facing_clinical_finding_eligible(
        "Smoking was not significantly associated with mortality."
    )
    assert is_user_facing_clinical_finding_eligible(
        "Platelets did not show a statistically significant association with mortality."
    )


def test_survival_primary_is_not_analytical_category() -> None:
    assert (
        classify_finding_category(
            "survival_primary was significantly associated with mortality.",
            variable="survival_primary",
            endpoint="mortality",
            analysis_type="association",
        )
        == "statistical_note"
    )


def test_endpoint_self_association_is_artifact_excluded() -> None:
    assert (
        classify_finding_category(
            "DEATH_EVENT was significantly associated with mortality.",
            variable="DEATH_EVENT",
            endpoint="DEATH_EVENT",
            analysis_type="association",
        )
        == "artifact_excluded"
    )
