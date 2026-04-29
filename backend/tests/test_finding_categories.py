from qtrial_backend.agentic.finding_categories import (
    classify_claim_type,
    classify_finding_category,
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
