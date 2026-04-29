from __future__ import annotations

import re
from typing import Literal


FindingCategory = Literal[
    "analytical",
    "clinical_association",
    "negative_association",
    "survival_result",
    "endpoint_result",
    "statistical_note",
    "data_quality",
    "data_quality_note",
    "preprocessing",
    "pipeline_warning",
    "qc_note",
    "artifact_excluded",
]

ClaimType = Literal[
    "association_claim",
    "analytical_association",
    "negative_association",
    "descriptive_claim",
    "descriptive_context",
    "statistical_note",
    "data_quality_claim",
    "data_quality_note",
    "setup_claim",
    "metadata_claim",
    "recommendation",
    "artifact",
]

GroundingStatusLabel = Literal[
    "Supported",
    "Contradicted",
    "Novel",
    "Data Quality Note",
    "Preprocessing Observation",
    "Pipeline Warning",
    "QC Observation",
]

ANALYTICAL_FINDING_CATEGORIES = {
    "analytical",
    "clinical_association",
    "negative_association",
}

COMPARISON_EXCLUDED_CATEGORIES = {
    "survival_result",
    "endpoint_result",
    "statistical_note",
    "data_quality",
    "data_quality_note",
    "preprocessing",
    "pipeline_warning",
    "qc_note",
    "artifact_excluded",
}

COMPARISON_INCLUDED_CLAIM_TYPES = {
    "association_claim",
    "analytical_association",
    "negative_association",
}

_DATA_QUALITY_PATTERNS = (
    "duplicate row",
    "duplicate rows",
    "duplicate key",
    "duplicate keys",
    "key-column duplicate",
    "key-column duplicates",
    "key column duplicate",
    "key column duplicates",
    "repeated-row",
    "repeated row",
    "repeated rows",
    "duplicate record",
    "duplicate records",
    "record integrity",
    "record-integrity",
    "data integrity",
    "integrity check",
    "digit preference",
    "duplicate count",
    "duplicate counts",
    "constant column",
    "constant columns",
    "outlier",
    "outliers",
    "baseline imbalance",
    "mcar",
    "missingness",
    "missing data",
)

_PREPROCESSING_PATTERNS = (
    "imputation",
    "imputed",
    "mice",
    "rubin",
    "listwise deletion",
    "rows dropped",
    "excluded from primary analysis",
    "preprocess",
    "pre-processing",
)

_PIPELINE_WARNING_PATTERNS = (
    "warning",
    "warnings",
    "failed",
    "failure",
    "error",
    "skipped",
    "could not",
    "not provided",
    "requires",
    "insufficient configuration",
    "interpret with caution",
    "median survival: inf",
    "median survival = inf",
    "overall median survival: inf",
    "overall median survival = inf",
    "infinite median survival",
    "censoring",
    "censored",
    "survival analysis (time=",
    "event=",
)

_QC_NOTE_PATTERNS = (
    "gate is open",
    "gate is closed",
    "hierarchical gate",
    "housekeeping",
    "setup",
    "configuration",
    "not applicable",
    "applicable",
    "ancova",
    "mmrm",
    "clda",
    "adjusted treatment p",
    "treatment-effect",
    "treatment effect",
)

_METHODOLOGY_INSTRUCTION_PATTERNS = (
    r"^(?:please\s+)?(?:perform|run|conduct|use|apply)\b.*\b(?:analysis|regression|curve|curves|model|models|test|tests)\b",
    r"^(?:please\s+)?(?:perform|run|conduct|use|apply)\b.*\b(?:kaplan[- ]meier|cox regression|cox proportional hazards|log-rank)\b",
    r"^(?:survival analysis|kaplan[- ]meier(?: curves?)?|cox regression|cox proportional hazards regression)\b(?:\s*[:(,-]|\s+with\b|\s+including\b)",
)

_ANALYTICAL_PATTERNS = (
    "associated",
    "association",
    "predict",
    "predictor",
    "survival",
    "mortality",
    "death",
    "hazard ratio",
    "odds ratio",
    "risk ratio",
    "statistically significant",
    "significant",
    "not significant",
    "p=",
    "p <",
    "p<",
    "correlated",
)

_DESCRIPTIVE_PATTERNS = (
    "event rate",
    "event rates",
    "median survival",
    "median survival: inf",
    "overall median survival: inf",
    "overall survival",
    "median overall survival",
    "overall median survival",
    "median follow up",
    "median follow-up",
    "follow-up duration",
    "follow up duration",
    "mean age",
    "baseline characteristics",
    "cohort summary",
    "proportion",
    "rate of",
)

_SYNTHETIC_ENDPOINT_PATTERNS = (
    "survival_primary",
    "survival_status",
    "mortality_flag",
    "event_flag",
    "death_event",
    "death event",
    "primary_endpoint",
    "primary_outcome",
)

_FOLLOW_UP_PATTERNS = (
    "follow-up time",
    "follow up time",
    "followup time",
    "followup_time",
    "time-to-event",
    "time to event",
    "censoring time",
)

_RAW_STAT_ARTIFACT_RE = re.compile(
    r"^\s*`?[a-z][a-z0-9_\s-]{1,60}`?\s*[:;,-]\s*"
    r".*(?:χ²|χ2|chi\s*-?\s*square|chi2)\s*[=:]",
    re.IGNORECASE,
)

_METADATA_PATTERNS = (
    "dataset comprises",
    "dataset included",
    "cohort comprised",
    "patients were included",
    "subjects were included",
    "study included",
    "trial included",
    "baseline cohort",
    "cohort consisted",
    "patient characteristics",
)


def is_analytical_category(category: str | None) -> bool:
    return (category or "analytical") in ANALYTICAL_FINDING_CATEGORIES


def neutral_status_for_category(category: str | None) -> GroundingStatusLabel:
    mapping: dict[str, GroundingStatusLabel] = {
        "data_quality": "Data Quality Note",
        "data_quality_note": "Data Quality Note",
        "statistical_note": "QC Observation",
        "preprocessing": "Preprocessing Observation",
        "pipeline_warning": "Pipeline Warning",
        "qc_note": "QC Observation",
        "artifact_excluded": "QC Observation",
    }
    return mapping.get(category or "", "QC Observation")


def is_comparison_claim_type(claim_type: str | None) -> bool:
    return (claim_type or "association_claim") in COMPARISON_INCLUDED_CLAIM_TYPES


def is_methodology_instruction_text(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    return any(re.search(pattern, lowered) for pattern in _METHODOLOGY_INSTRUCTION_PATTERNS)


def classify_finding_category(
    text: str,
    *,
    variable: str | None = None,
    endpoint: str | None = None,
    analysis_type: str | None = None,
) -> FindingCategory:
    lowered = " ".join((text or "").lower().split())
    if _RAW_STAT_ARTIFACT_RE.search(text or "") and not any(
        phrase in lowered
        for phrase in (
            "associated with",
            "not associated",
            "not significantly associated",
            "was significantly associated",
            "were significantly associated",
        )
    ):
        return "statistical_note"

    if any(pattern in lowered for pattern in _PREPROCESSING_PATTERNS):
        return "preprocessing"
    if any(pattern in lowered for pattern in _DATA_QUALITY_PATTERNS):
        return "data_quality_note"
    if any(pattern in lowered for pattern in _PIPELINE_WARNING_PATTERNS):
        return "statistical_note"
    if any(pattern in lowered for pattern in _FOLLOW_UP_PATTERNS):
        return "statistical_note"
    if any(pattern in lowered for pattern in _SYNTHETIC_ENDPOINT_PATTERNS):
        if variable and endpoint and variable.lower() == endpoint.lower():
            return "artifact_excluded"
        return "statistical_note"
    if is_methodology_instruction_text(lowered):
        return "qc_note"
    if any(pattern in lowered for pattern in _QC_NOTE_PATTERNS):
        return "qc_note"

    if variable or analysis_type == "association":
        if variable and endpoint and variable.strip().lower() == endpoint.strip().lower():
            return "artifact_excluded"
        if variable and is_endpoint_like_variable(variable):
            return "artifact_excluded"
        if variable and is_followup_time_variable(variable):
            return "statistical_note"
        if endpoint in {"mortality", "primary_outcome"} and not variable:
            return "endpoint_result"
        return "analytical"

    if endpoint == "survival" or "survival" in lowered:
        return "survival_result"
    if endpoint in {"mortality", "primary_outcome"}:
        return "endpoint_result"
    if any(pattern in lowered for pattern in _ANALYTICAL_PATTERNS):
        return "analytical"

    return "qc_note"


def classify_claim_type(
    text: str,
    *,
    finding_category: str | None = None,
    variable: str | None = None,
    endpoint: str | None = None,
    significant: bool | None = None,
    p_value: float | None = None,
) -> ClaimType:
    lowered = " ".join((text or "").lower().split())
    category = finding_category or classify_finding_category(
        text,
        variable=variable,
        endpoint=endpoint,
        analysis_type="association" if variable else None,
    )

    if category == "artifact_excluded":
        return "artifact"
    if any(pattern in lowered for pattern in _METADATA_PATTERNS):
        return "metadata_claim"
    if any(pattern in lowered for pattern in _DESCRIPTIVE_PATTERNS):
        return "descriptive_context"
    if re.search(r"\b\d+(?:\.\d+)?%\b", lowered) and not any(
        token in lowered for token in ("significant", "associated", "predict", "hazard ratio", "odds ratio")
    ):
        return "descriptive_context"

    if category in {"data_quality", "data_quality_note", "preprocessing"}:
        return "data_quality_note"
    if category in {"statistical_note"}:
        return "statistical_note"
    if category in {"pipeline_warning", "qc_note"}:
        return "setup_claim"

    if variable and (significant is not None or p_value is not None):
        return "negative_association" if significant is False else "analytical_association"
    if any(pattern in lowered for pattern in _ANALYTICAL_PATTERNS):
        return "association_claim"

    return "descriptive_claim"


def is_endpoint_like_variable(variable: str | None, endpoint: str | None = None) -> bool:
    if not variable:
        return False
    var = " ".join(str(variable).strip().lower().replace("_", " ").split())
    if endpoint and var == " ".join(str(endpoint).strip().lower().replace("_", " ").split()):
        return True
    compact = var.replace(" ", "_")
    if compact in {
        "survival_primary",
        "survival_status",
        "survival_outcome",
        "the_survival_outcome",
        "mortality_flag",
        "event_flag",
        "death_event",
        "primary_endpoint",
        "primary_outcome",
    }:
        return True
    return any(token in compact for token in ("mortality_flag", "event_flag", "survival_status", "survival_outcome"))


def is_followup_time_variable(variable: str | None) -> bool:
    if not variable:
        return False
    compact = str(variable).strip().lower().replace("-", "_").replace(" ", "_")
    return compact in {
        "time",
        "follow_up",
        "followup",
        "follow_up_time",
        "followup_time",
        "survival_time",
        "time_to_event",
        "censoring_time",
    }
