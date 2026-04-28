from __future__ import annotations

import re
from typing import Literal


FindingCategory = Literal[
    "analytical",
    "survival_result",
    "endpoint_result",
    "data_quality",
    "preprocessing",
    "pipeline_warning",
    "qc_note",
]

ClaimType = Literal[
    "association_claim",
    "descriptive_claim",
    "data_quality_claim",
    "setup_claim",
    "metadata_claim",
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
    "survival_result",
    "endpoint_result",
}

COMPARISON_EXCLUDED_CATEGORIES = {
    "data_quality",
    "preprocessing",
    "pipeline_warning",
    "qc_note",
}

COMPARISON_INCLUDED_CLAIM_TYPES = {"association_claim"}

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
        "preprocessing": "Preprocessing Observation",
        "pipeline_warning": "Pipeline Warning",
        "qc_note": "QC Observation",
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

    if any(pattern in lowered for pattern in _PREPROCESSING_PATTERNS):
        return "preprocessing"
    if any(pattern in lowered for pattern in _DATA_QUALITY_PATTERNS):
        return "data_quality"
    if any(pattern in lowered for pattern in _PIPELINE_WARNING_PATTERNS):
        return "pipeline_warning"
    if is_methodology_instruction_text(lowered):
        return "qc_note"
    if any(pattern in lowered for pattern in _QC_NOTE_PATTERNS):
        return "qc_note"

    if variable or analysis_type == "association":
        if endpoint == "survival" or "survival" in lowered:
            return "survival_result"
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

    if any(pattern in lowered for pattern in _METADATA_PATTERNS):
        return "metadata_claim"
    if any(pattern in lowered for pattern in _DESCRIPTIVE_PATTERNS):
        return "descriptive_claim"
    if re.search(r"\b\d+(?:\.\d+)?%\b", lowered) and not any(
        token in lowered for token in ("significant", "associated", "predict", "hazard ratio", "odds ratio")
    ):
        return "descriptive_claim"

    if category in {"data_quality", "preprocessing"}:
        return "data_quality_claim"
    if category in {"pipeline_warning", "qc_note"}:
        return "setup_claim"

    if variable and (significant is not None or p_value is not None):
        return "association_claim"
    if any(pattern in lowered for pattern in _ANALYTICAL_PATTERNS):
        return "association_claim"

    return "descriptive_claim"
