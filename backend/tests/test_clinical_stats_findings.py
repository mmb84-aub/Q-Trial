import pandas as pd

from qtrial_backend.tools.stats.clinical_stats import run_clinical_analysis


def _heart_failure_regression_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [
                42, 45, 47, 50, 52, 54, 55, 57, 58, 60, 61, 63,
                66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
            ],
            "anaemia": [
                0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            "creatinine_phosphokinase": [
                110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
                340, 360, 380, 420, 460, 500, 560, 620, 700, 820, 940, 1080,
            ],
            "diabetes": [
                0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
            ],
            "ejection_fraction": [
                62, 60, 59, 57, 56, 55, 54, 53, 52, 50, 49, 48,
                38, 36, 34, 32, 30, 28, 26, 24, 22, 21, 20, 18,
            ],
            "high_blood_pressure": [
                0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            "platelets": [
                280000, 275000, 290000, 285000, 270000, 265000, 300000, 295000, 288000, 282000, 276000, 271000,
                268000, 264000, 260000, 255000, 250000, 246000, 242000, 238000, 234000, 230000, 226000, 222000,
            ],
            "serum_creatinine": [
                0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2,
                1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            ],
            "serum_sodium": [
                143, 142, 142, 141, 141, 140, 140, 139, 139, 138, 138, 137,
                136, 135, 135, 134, 134, 133, 132, 132, 131, 130, 129, 128,
            ],
            "sex": [
                1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            ],
            "smoking": [
                0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            ],
            "time": [
                240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152,
                144, 132, 120, 108, 96, 84, 72, 60, 48, 36, 24, 12,
            ],
            "DEATH_EVENT": [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
        }
    )


def test_clinical_analysis_emits_structured_finding_without_hardcoded_plain_text() -> None:
    df = pd.DataFrame(
        {
            "age": [60, 61, 70, 72, 80, 82],
            "death_event": [0, 0, 1, 1, 1, 1],
        }
    )
    result = run_clinical_analysis(
        df,
        {
            "primary_endpoints": ["death_event"],
            "outcome_type": "binary",
            "alpha": 0.05,
        },
    )

    findings = result.get("stage_3_corrections", {}).get("corrected_findings", [])
    assert isinstance(findings, list)
    if findings:
        finding = findings[0]
        assert "finding_text_raw" in finding
        assert finding.get("finding_text_plain") is None
        assert finding.get("finding_category") in {"analytical", "survival_result"}
        assert "variable" in finding
        assert "significant_after_correction" in finding


def test_survival_analysis_uses_configured_event_column_for_endpoint() -> None:
    df = pd.DataFrame(
        {
            "time": [5, 12, 19, 30, 44, 60],
            "DEATH_EVENT": [0, 0, 1, 0, 1, 1],
            "anaemia": [0, 1, 0, 1, 0, 1],
        }
    )
    result = run_clinical_analysis(
        df,
        {
            "time_col": "time",
            "event_col": "DEATH_EVENT",
            "outcome_type": "survival",
            "alpha": 0.05,
        },
    )

    findings = result.get("stage_3_corrections", {}).get("corrected_findings", [])
    assert isinstance(findings, list)
    if findings:
        assert all(f.get("endpoint") == "mortality" for f in findings)
        assert all(f.get("finding_id") != "DEATH_EVENT" for f in findings)


def test_serum_creatinine_generates_corrected_finding_for_survival_outcome() -> None:
    df = pd.DataFrame(
        {
            "time": [5, 8, 12, 18, 25, 33, 40, 48, 56, 64],
            "DEATH_EVENT": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "age": [50, 52, 54, 56, 58, 70, 72, 74, 76, 78],
            "serum_creatinine": [0.8, 0.9, 1.0, 0.9, 1.1, 2.1, 2.3, 2.4, 2.5, 2.6],
            "ejection_fraction": [45, 47, 48, 46, 49, 20, 22, 24, 25, 23],
            "anaemia": [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            "creatinine_phosphokinase": [100, 110, 120, 115, 130, 140, 135, 145, 150, 142],
        }
    )
    result = run_clinical_analysis(
        df,
        {
            "time_col": "time",
            "event_col": "DEATH_EVENT",
            "outcome_type": "survival",
            "primary_endpoints": [
                "age",
                "anaemia",
                "creatinine_phosphokinase",
                "ejection_fraction",
                "serum_creatinine",
            ],
            "alpha": 0.05,
        },
    )

    findings = result.get("stage_3_corrections", {}).get("corrected_findings", [])
    assert isinstance(findings, list)
    serum_finding = next((f for f in findings if f.get("finding_id") == "serum_creatinine"), None)
    assert serum_finding is not None
    assert serum_finding.get("endpoint") == "mortality"


def test_survival_logrank_p_value_is_harvested_into_corrected_findings() -> None:
    df = pd.DataFrame(
        {
            "time": [5, 8, 12, 16, 20, 24, 28, 32],
            "DEATH_EVENT": [1, 1, 1, 1, 0, 0, 0, 0],
            "trt": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    result = run_clinical_analysis(
        df,
        {
            "time_col": "time",
            "event_col": "DEATH_EVENT",
            "treatment_col": "trt",
            "outcome_type": "survival",
            "alpha": 0.05,
        },
    )

    primary_analysis = result.get("stage_2_analysis", {}).get("primary_analysis", {})
    assert primary_analysis.get("logrank_p_value") is not None

    findings = result.get("stage_3_corrections", {}).get("corrected_findings", [])
    survival_finding = next((f for f in findings if f.get("finding_id") == "survival_primary"), None)

    assert survival_finding is not None
    assert survival_finding.get("raw_p_value") == primary_analysis.get("logrank_p_value")
    assert survival_finding.get("finding_category") == "survival_result"


def test_survival_predictors_are_marked_significant_and_event_is_not_a_predictor() -> None:
    df = pd.DataFrame(
        {
            "time": [5, 8, 12, 18, 25, 33, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112],
            "DEATH_EVENT": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "age": [48, 50, 52, 54, 55, 56, 57, 58, 72, 74, 76, 78, 80, 82, 84, 86],
            "serum_creatinine": [0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
            "ejection_fraction": [55, 54, 53, 52, 51, 50, 49, 48, 25, 24, 23, 22, 21, 20, 19, 18],
            "serum_sodium": [142, 141, 141, 140, 140, 139, 139, 138, 134, 133, 132, 131, 130, 129, 128, 127],
            "anaemia": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    result = run_clinical_analysis(
        df,
        {
            "time_col": "time",
            "event_col": "DEATH_EVENT",
            "outcome_type": "survival",
            "primary_endpoints": [
                "DEATH_EVENT",
                "age",
                "ejection_fraction",
                "serum_creatinine",
                "serum_sodium",
                "anaemia",
            ],
            "alpha": 0.05,
        },
    )

    findings = result.get("stage_3_corrections", {}).get("corrected_findings", [])
    assert isinstance(findings, list)
    finding_map = {f.get("finding_id"): f for f in findings}

    assert "DEATH_EVENT" not in finding_map
    assert finding_map["age"]["significant_after_correction"] is True
    assert finding_map["ejection_fraction"]["significant_after_correction"] is True
    assert finding_map["serum_creatinine"]["significant_after_correction"] is True
    assert finding_map["serum_sodium"]["significant_after_correction"] is True


def test_heart_failure_survival_regression_uses_death_event_and_surfaces_known_predictors() -> None:
    df = _heart_failure_regression_fixture()

    result = run_clinical_analysis(
        df,
        {
            "time_col": "time",
            "event_col": "anaemia",
            "outcome_type": "survival",
            "primary_endpoints": ["age", "anaemia", "ejection_fraction", "serum_creatinine", "serum_sodium"],
            "alpha": 0.05,
        },
    )

    primary_analysis = result.get("stage_2_analysis", {}).get("primary_analysis", {})
    assert primary_analysis.get("resolved_event_column") == "DEATH_EVENT"

    findings = result.get("stage_3_corrections", {}).get("corrected_findings", [])
    assert isinstance(findings, list)
    finding_map = {f.get("finding_id"): f for f in findings}

    assert "DEATH_EVENT" not in finding_map
    assert finding_map["age"]["significant_after_correction"] is True
    assert finding_map["ejection_fraction"]["significant_after_correction"] is True
    assert finding_map["serum_creatinine"]["significant_after_correction"] is True
    assert finding_map["serum_sodium"]["significant_after_correction"] is True
