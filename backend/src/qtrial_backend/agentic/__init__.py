# agentic package
from qtrial_backend.agentic.schemas import (
    DataQualityOutput,
    ClinicalSemanticsOutput,
    UnknownsOutput,
    InsightSynthesisOutput,
    FinalReportSchema,
    FailedClaim,
    RubricScores,
    JudgeOutput,
    MetadataInput,
    LabUnit,
    ToolCallRecord,
    # Task 4A reasoning schemas
    EvidenceSupportEntry,
    CandidateHypothesis,
    ClaimDraft,
    Contradiction,
    ReasoningStepLog,
    StopCondition,
    ConfidenceSummary,
    ReasoningState,
    ReasoningAction,
    ReasoningActionType,
    ReasoningStepType,
)
from qtrial_backend.agentic.judge import run_judge_agent
from qtrial_backend.agentic.orchestrator import run_agentic_insights
from qtrial_backend.agentic.reasoning import (
    build_valid_citation_keys,
    validate_claim_citations,
    check_endpoint_lockdown,
    check_treatment_effect_guard,
    check_high_confidence_guard,
    validate_claim,
    validate_all_claims,
    compute_confidence_summary,
    init_reasoning_state,
    append_reasoning_step,
    ValidationResult,
)

__all__ = [
    # existing agent output types
    "DataQualityOutput",
    "ClinicalSemanticsOutput",
    "UnknownsOutput",
    "InsightSynthesisOutput",
    "FinalReportSchema",
    "FailedClaim",
    "RubricScores",
    "JudgeOutput",
    "MetadataInput",
    "LabUnit",
    # upstream integration
    "ToolCallRecord",
    # Task 4A reasoning schemas
    "EvidenceSupportEntry",
    "CandidateHypothesis",
    "ClaimDraft",
    "Contradiction",
    "ReasoningStepLog",
    "ReasoningStepType",
    "StopCondition",
    "ConfidenceSummary",
    "ReasoningState",
    "ReasoningAction",
    "ReasoningActionType",
    # Task 4A validators + helpers
    "ValidationResult",
    "build_valid_citation_keys",
    "validate_claim_citations",
    "check_endpoint_lockdown",
    "check_treatment_effect_guard",
    "check_high_confidence_guard",
    "validate_claim",
    "validate_all_claims",
    "compute_confidence_summary",
    "init_reasoning_state",
    "append_reasoning_step",
    # pipeline runners
    "run_judge_agent",
    "run_agentic_insights",
]
