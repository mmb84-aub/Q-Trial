from qtrial_backend.agentic.finding_categories import (
    classify_claim_type,
    classify_finding_category,
    neutral_status_for_category,
)


def test_duplicate_row_finding_is_data_quality() -> None:
    assert (
        classify_finding_category("Duplicate rows were detected during the data integrity check.")
        == "data_quality"
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
        == "data_quality"
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
    ) == "association_claim"


def test_descriptive_claim_type_for_event_rate() -> None:
    assert classify_claim_type("The event rate was 32.1% during follow-up.") == "descriptive_claim"


def test_descriptive_claim_type_for_median_survival() -> None:
    assert classify_claim_type("Overall median survival was 209 days.") == "descriptive_claim"


def test_descriptive_claim_type_for_colon_median_survival_summary() -> None:
    assert classify_claim_type("Overall median survival: 209.0") == "descriptive_claim"


def test_data_quality_claim_type_for_digit_preference() -> None:
    assert classify_claim_type("Digit preference was detected in age values.", finding_category="data_quality") == "data_quality_claim"


def test_setup_claim_type_for_survival_setup_line() -> None:
    assert classify_claim_type(
        "Survival Analysis (time=`time`, event=`DEATH_EVENT`).",
        finding_category="pipeline_warning",
    ) == "setup_claim"


def test_metadata_claim_type_for_cohort_description() -> None:
    assert classify_claim_type("The dataset comprises 299 patients with chronic heart failure.") == "metadata_claim"
