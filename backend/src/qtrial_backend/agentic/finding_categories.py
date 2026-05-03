from __future__ import annotations

import re
from typing import Any, Literal


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
    "outlier summary",
    "outlier summaries",
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
    "qubo",
    "feature selection",
    "feature-selector",
    "selected via",
    "selected using",
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
    "normality",
    "shapiro",
    "kolmogorov",
    "anderson-darling",
    "test selection",
    "test-selection",
    "model selection",
    "study design",
    "study-design",
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
    r"^\s*(?:[-*•]|\d+[.)])?\s*"
    r"`?[a-z][a-z0-9_\s.-]{0,80}`?\s*(?::|;|,|-)?\s*"
    r"(?:χ\s*[²2]|chi\s*-?\s*square|chi2|x\s*\^?\s*2)\s*[=:]?\s*"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r".*\bp\s*(?:=|<|>|<=|>=|≤|≥)\s*"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?",
    re.IGNORECASE,
)
_RAW_STAT_ARTIFACT_TEXT_FIELDS = (
    "comparison_claim_text",
    "finding_text_plain",
    "finding_text",
    "source_finding_plain",
    "source_finding",
    "source_text",
    "plain",
    "finding_text_raw",
    "source_finding_raw",
    "raw",
)
_PRIMARY_ARTIFACT_TEXT_FIELDS = (
    "comparison_claim_text",
    "finding_text_plain",
    "finding_text",
    "source_finding_plain",
    "source_finding",
    "source_text",
    "plain",
    "finding_text_raw",
)
_CLINICAL_INTERPRETATION_PHRASES = (
    "associated with",
    "not associated",
    "not significantly associated",
    "did not show",
    "showed no",
    "was significantly associated",
    "were significantly associated",
    "predict",
    "predictor",
    "higher",
    "lower",
    "increased risk",
    "decreased risk",
    "mortality risk",
    "survival",
)
_NON_FINDING_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*•]\s*)?"
    r"(?:\*\*|__)?\s*"
    r"(?:"
    r"hazard\s+ratios?|odds\s+ratios?|risk\s+ratios?|effect\s+sizes?|"
    r"test[_\s-]*selection[_\s-]*rationale|follow[-\s]*up|binary\s+outcome|"
    r"columns?|imputation|median\s+survival|continuous\s+variables?|"
    r"categorical\s+variables?|cox\s+regression|logistic\s+regression|"
    r"model\s+summary|statistical\s+notes?|analytical\s+findings|"
    r"data\s+quality(?:\s+notes?)?|results?|methods?|summary"
    r")"
    r"(?:\s*\([^)]*\))?\s*(?:\*\*|__)?\s*:?\s*$",
    re.IGNORECASE,
)
_NON_FINDING_WRAPPER_PHRASES = (
    "all continuous variables are non-normal",
    "all continuous variables are non normal",
    "all continuous variables rejected normality",
    "all continuous variables reject normality",
    "continuous variables rejected normality",
    "continuous variables reject normality",
)

_METADATA_PATTERNS = (
    "dataset comprises",
    "dataset included",
    "dataset includes",
    "cohort comprised",
    "patients were included",
    "subjects were included",
    "study included",
    "trial included",
    "study design",
    "trial design",
    "baseline cohort",
    "cohort consisted",
    "patient characteristics",
    "predictor variables selected",
    "variables selected via",
    "variables selected using",
)

_DANGLING_ENDING_RE = re.compile(
    r"(?:\bvs\.?|\bversus|\bcompared\s+with|\bcompared\s+to|\brelative\s+to|"
    r"\bthan|\bbetween|\band|\bor|[(\[{])\s*$",
    re.IGNORECASE,
)
_BARE_NUMERIC_FRAGMENT_START_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*"
    r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|"
    r"(?:p|hr|or|rr|ci|auc|smd|χ\s*[²2]|chi\s*-?\s*square)\b)",
    re.IGNORECASE,
)
_STARTS_WITH_BARE_STAT_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*"
    r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?\s*"
    r"(?:%|mg/dl|mmol/l|g/dl|days?|years?|months?|ng/ml|u/l|iu/l|cm|mm|kg|bpm|mmhg)?|"
    r"(?:p|hr|or|rr|ci|auc|smd|χ\s*[²2]|chi\s*-?\s*square)\s*(?:=|<|>|≤|≥|:))",
    re.IGNORECASE,
)
_CONTINUATION_START_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*(?:vs\.?|versus|compared\s+(?:with|to)|"
    r"relative\s+to|than|and|or|respectively|whereas|while)\b",
    re.IGNORECASE,
)
_WRAPPER_LABEL_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*|__)?\s*"
    r"(?:interpretation|event\s+rate|primary\s+analysis\s+results?|"
    r"characteristics\s+by\b[^:]*|prevalence\s+of\b[^:]*|"
    r"independent\s+predictors?|effect\s+size|study\s+design|"
    r"dataset\s+summary|key\s+statistical\s+findings|"
    r"clinical\s+analysis\s+report|statistical\s+notes?|data\s+quality\s+notes?)"
    r"\s*(?:\*\*|__)?\s*:",
    re.IGNORECASE,
)
_P_VALUE_SIGNAL_RE = re.compile(r"\bp(?:\s*[- ]?\s*value)?\s*(?:=|<|>|<=|>=|≤|≥)", re.IGNORECASE)
_RELATIONSHIP_RE = re.compile(
    r"\b(?:associated|association|predicts?|predicted|correlated|correlation|"
    r"differ(?:ed|ence)?|increased|decreased|reduced|lowered|higher|lower|"
    r"greater|less|fewer|significant|not significant|show(?:ed|s)?|demonstrat(?:ed|es?)|"
    r"odds|hazard|risk|mortality|survival|outcome|endpoint)\b",
    re.IGNORECASE,
)
_SUBJECT_VERB_RE = re.compile(
    r"\b[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,5}\s+"
    r"(?:was|were|is|are|had|has|showed|shows|did|does|predicts?|predicted|"
    r"correlated|differed|increased|decreased|reduced|lowered)\b",
    re.IGNORECASE,
)
_CLINICAL_RELATION_RE = re.compile(
    r"\b(?:(?:significantly\s+|statistically\s+|strongly\s+)?associated\s+with|"
    r"associated\s+(?:significantly\s+|statistically\s+|strongly\s+)?with|"
    r"association\s+with|not\s+(?:significantly\s+)?associated|"
    r"did\s+not\s+show(?:\s+a)?(?:\s+statistically)?\s+significant\s+association|"
    r"showed\s+no(?:\s+statistically)?\s+significant\s+association|"
    r"show(?:s|ed)?\s+no(?:\s+(?:strong|statistically\s+significant|significant))?\s+relationship|"
    r"did\s+not\s+show(?:\s+a)?(?:\s+(?:strong|statistically\s+significant|significant))?\s+relationship|"
    r"no(?:\s+(?:strong|statistically\s+significant|significant))?\s+relationship\s+with|"
    r"predicts?|predicted|predictor(?:s)?\s+of|correlated\s+with|"
    r"positively\s+correlated|negatively\s+correlated|"
    r"differ(?:ed|s)?(?:\s+significantly)?\s+(?:between|in)|"
    r"was\s+(?:statistically\s+|significantly\s+)?significant\s+(?:for|with|in)|"
    r"were\s+(?:statistically\s+|significantly\s+)?significant\s+(?:for|with|in)|"
    r"was\s+higher|were\s+higher|was\s+lower|were\s+lower|"
    r"increased|decreased|reduces?|lowered|raises?|raised|"
    r"had\s+(?:a\s+)?(?:significantly\s+)?(?:higher|lower|increased|decreased|reduced))\b",
    re.IGNORECASE,
)
_OUTCOME_OR_GROUP_RE = re.compile(
    r"\b(?:mortality|death|deaths|survival|outcome|endpoint|risk|odds|hazard|event|"
    r"patients?\s+who\s+died|survivors?|non[-\s]?survivors?|groups?|treatment\s+arm|"
    r"control\s+group|compared\s+with|compared\s+to|between)\b",
    re.IGNORECASE,
)
_NO_VARIABLE_START_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*|__)?\s*"
    r"(?:no\s+association|association|relationship|difference|mortality|death|survival|outcome|endpoint)\b",
    re.IGNORECASE,
)
_VARIABLE_SUBJECT_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*|__)?\s*"
    r"(?:each\s+\d+(?:\.\d+)?%?\s+increase\s+in\s+|higher|lower|elevated|reduced|older|younger)?\s*"
    r"[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,5}\s+"
    r"(?:was|were|is|are|had|has|show|showed|shows|did|does|predicts?|predicted|"
    r"correlated|differed|increased|decreased|reduces?|lowered|raises?|raised)\b",
    re.IGNORECASE,
)
_GROUP_COMPARISON_WITH_VARIABLE_RE = re.compile(
    r"\bpatients?\s+who\s+died\b|\bsurvivors?\b|\bnon[-\s]?survivors?\b|\bgroups?\b",
    re.IGNORECASE,
)
_PASSIVE_ENDPOINT_VARIABLE_RE = re.compile(
    r"^\s*(?:mortality|death|survival|outcome|endpoint|risk|odds|hazard|event)\s+"
    r"(?:was|were|is|are)\s+(?:significantly\s+|statistically\s+|strongly\s+)?"
    r"(?:associated|related|linked)\s+with\s+"
    r"(?!p\s*(?:=|<|>|≤|≥))"
    r"[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,4}\b",
    re.IGNORECASE,
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


def is_raw_statistical_artifact_text(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    if not _RAW_STAT_ARTIFACT_RE.search(text or ""):
        return False
    return not any(phrase in lowered for phrase in _CLINICAL_INTERPRETATION_PHRASES)


def is_user_facing_clinical_finding_eligible(finding: Any) -> bool:
    """Strict final gate for standalone analytical clinical findings."""
    text = _first_text_field(finding, _PRIMARY_ARTIFACT_TEXT_FIELDS) if not isinstance(finding, str) else finding
    if not text:
        return False
    cleaned = _clean_header_candidate(text)
    if not cleaned:
        return False
    matching_text = cleaned.replace("_", " ")
    lowered = cleaned.lower()
    if is_user_facing_nonfinding_artifact(cleaned):
        return False
    if _WRAPPER_LABEL_RE.match(cleaned):
        return False
    if _BARE_NUMERIC_FRAGMENT_START_RE.match(cleaned):
        return False
    if _STARTS_WITH_BARE_STAT_RE.match(cleaned):
        return False
    passive_endpoint_variable = bool(_PASSIVE_ENDPOINT_VARIABLE_RE.search(matching_text))
    if _NO_VARIABLE_START_RE.match(cleaned) and not passive_endpoint_variable:
        return False
    if re.match(
        r"^\s*(?:[-*•]|\d+[.)])?\s*(?:this|these|that|those)\s+"
        r"(?:variable|variables|factor|factors|covariate|covariates|predictor|predictors)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return False
    if any(pattern in lowered for pattern in (*_METADATA_PATTERNS, *_PREPROCESSING_PATTERNS, *_QC_NOTE_PATTERNS)):
        return False
    if any(pattern in lowered for pattern in _DESCRIPTIVE_PATTERNS):
        if not _CLINICAL_RELATION_RE.search(cleaned):
            return False

    has_relation = bool(_CLINICAL_RELATION_RE.search(matching_text))
    has_outcome_or_group = bool(_OUTCOME_OR_GROUP_RE.search(matching_text))
    has_variable_subject = bool(_VARIABLE_SUBJECT_RE.search(cleaned))
    has_passive_variable_claim = passive_endpoint_variable
    has_group_variable_claim = bool(
        _GROUP_COMPARISON_WITH_VARIABLE_RE.search(matching_text)
        and re.search(r"\b(?:higher|lower|increased|decreased|reduced|elevated)\s+[a-z][a-z0-9_-]+", cleaned, re.IGNORECASE)
    )
    if not has_relation or not has_outcome_or_group:
        return False
    if not has_variable_subject and not has_group_variable_claim and not has_passive_variable_claim:
        return False
    return True


def is_malformed_finding_fragment_text(text: str) -> bool:
    """Return True for snippets that cannot stand alone as clinical findings."""
    cleaned = _clean_header_candidate(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if len(cleaned.split()) < 4:
        return True
    if any(
        pattern in lowered
        for pattern in (
            *_PREPROCESSING_PATTERNS,
            *_DATA_QUALITY_PATTERNS,
            *_QC_NOTE_PATTERNS,
            *_METADATA_PATTERNS,
        )
    ):
        return False
    if _DANGLING_ENDING_RE.search(cleaned):
        return True
    if cleaned.count("(") > cleaned.count(")") or cleaned.count("[") > cleaned.count("]"):
        return True
    if _CONTINUATION_START_RE.match(cleaned):
        return True
    if _BARE_NUMERIC_FRAGMENT_START_RE.match(cleaned) and len(cleaned.split()) <= 12 and not _SUBJECT_VERB_RE.search(cleaned):
        return True
    if _STARTS_WITH_BARE_STAT_RE.match(cleaned) and not _SUBJECT_VERB_RE.search(cleaned):
        return True

    has_p_value = bool(_P_VALUE_SIGNAL_RE.search(cleaned))
    has_claim_structure = bool(_SUBJECT_VERB_RE.search(cleaned) and _RELATIONSHIP_RE.search(cleaned))
    if has_p_value and not has_claim_structure:
        return True
    if (
        any(token in lowered for token in ("significant", "associated", "correlat"))
        or re.search(r"\bpredicts?\b|\bpredicted\b", lowered)
    ):
        return not has_claim_structure
    return False


def is_malformed_finding_fragment(finding: Any) -> bool:
    if isinstance(finding, str):
        return is_malformed_finding_fragment_text(finding)

    primary_text = _first_text_field(finding, _PRIMARY_ARTIFACT_TEXT_FIELDS)
    if primary_text:
        return is_malformed_finding_fragment_text(primary_text)

    return any(
        is_malformed_finding_fragment_text(text)
        for text in _iter_text_fields(finding, _RAW_STAT_ARTIFACT_TEXT_FIELDS)
    )


def is_raw_stat_artifact_finding(finding: Any) -> bool:
    """Hard final gate for raw variable-only statistical artifact findings.

    This intentionally ignores an upstream analytical category. A finding is
    excluded only when its user-facing text is a raw variable/test-statistic line
    such as "`time`: χ²=38.49, p=0.0000". Legitimate interpreted sentences such
    as "Smoking was not significantly associated with mortality" are preserved.
    """
    if isinstance(finding, str):
        return is_raw_statistical_artifact_text(finding)

    primary_text = _first_text_field(finding, _PRIMARY_ARTIFACT_TEXT_FIELDS)
    if primary_text:
        return is_raw_statistical_artifact_text(primary_text)

    return any(
        is_raw_statistical_artifact_text(text)
        for text in _iter_text_fields(finding, _RAW_STAT_ARTIFACT_TEXT_FIELDS)
    )


def is_non_finding_header_artifact_text(text: str) -> bool:
    cleaned = _clean_header_candidate(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if re.match(
        r"^\s*(?:this|these|that|those)\s+"
        r"(?:variable|variables|factor|factors|covariate|covariates|predictor|predictors)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    if _NON_FINDING_HEADER_RE.match(cleaned):
        return True
    if any(phrase in lowered for phrase in _NON_FINDING_WRAPPER_PHRASES):
        return True
    if re.search(
        r"\b(?:raw\s+p|adjusted\s+p|p)\s*(?:=|<|>|<=|>=|≤|≥)|"
        r"\b(?:effect\s+size|or|hr|rr)\s*=",
        lowered,
    ):
        return False
    if cleaned.endswith(":") and len(cleaned.rstrip(":").split()) <= 10:
        if not _has_interpretable_claim_verb(cleaned):
            return True
    if re.fullmatch(r"(?:\*\*|__)?[a-z0-9 _/\-()%'.,]+(?:\*\*|__)?", cleaned, re.IGNORECASE):
        if len(cleaned.split()) <= 8 and not _has_interpretable_claim_verb(cleaned):
            return any(
                token in lowered
                for token in (
                    "hazard",
                    "ratio",
                    "effect",
                    "cohen",
                    "bootstrap",
                    "follow",
                    "binary",
                    "columns",
                    "imputation",
                    "rationale",
                    "median survival",
                )
            )
    return False


def is_non_finding_header_artifact(finding: Any) -> bool:
    if isinstance(finding, str):
        return is_non_finding_header_artifact_text(finding)

    primary_text = _first_text_field(finding, _PRIMARY_ARTIFACT_TEXT_FIELDS)
    if primary_text:
        return is_non_finding_header_artifact_text(primary_text)

    return any(
        is_non_finding_header_artifact_text(text)
        for text in _iter_text_fields(finding, _RAW_STAT_ARTIFACT_TEXT_FIELDS)
    )


def is_user_facing_nonfinding_artifact(finding: Any) -> bool:
    return (
        is_malformed_finding_fragment(finding)
        or is_raw_stat_artifact_finding(finding)
        or is_non_finding_header_artifact(finding)
    )


def _clean_header_candidate(text: str) -> str:
    cleaned = " ".join((text or "").replace("\n", " ").split())
    cleaned = cleaned.strip(" \t-•")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = cleaned.strip()
    if (cleaned.startswith("**") and cleaned.endswith("**")) or (cleaned.startswith("__") and cleaned.endswith("__")):
        cleaned = cleaned[2:-2].strip()
    return cleaned


def _has_interpretable_claim_verb(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(was|were|is|are|had|has|showed|show|shows|associated|predicts?|predicted|"
            r"correlated|differed|increased|decreased|reduced|lowered|raises?|lowers?)\b",
            lowered,
        )
    )


def _first_text_field(finding: Any, fields: tuple[str, ...]) -> str | None:
    for text in _iter_text_fields(finding, fields):
        if text:
            return text
    return None


def _iter_text_fields(finding: Any, fields: tuple[str, ...]) -> list[str]:
    texts: list[str] = []
    if isinstance(finding, dict):
        for field in fields:
            value = finding.get(field)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        return texts
    for field in fields:
        value = getattr(finding, field, None)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def classify_finding_category(
    text: str,
    *,
    variable: str | None = None,
    endpoint: str | None = None,
    analysis_type: str | None = None,
) -> FindingCategory:
    lowered = " ".join((text or "").lower().split())
    if (text or "").strip() and is_non_finding_header_artifact_text(text):
        return "statistical_note"
    if is_raw_statistical_artifact_text(text):
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
    if any(pattern in lowered for pattern in _METADATA_PATTERNS):
        return "qc_note"
    if is_malformed_finding_fragment_text(text):
        return "artifact_excluded"

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
