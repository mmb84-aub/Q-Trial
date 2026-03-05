from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Planner ───────────────────────────────────────────────────────────────────

AgentName = Literal["DataQualityAgent", "ClinicalSemanticsAgent", "UnknownsAgent", "InsightSynthesisAgent"]


class PlanStep(BaseModel):
    step_number: int = Field(..., ge=1)
    name: str
    goal: str
    inputs_used: list[str] = Field(
        ..., description="Which evidence/preview keys this step consumes"
    )
    expected_output_keys: list[str]
    agent_to_call: AgentName


class PlanSchema(BaseModel):
    dataset_summary: str = Field(..., description="One-line description inferred from preview")
    steps: list[PlanStep] = Field(..., min_length=1, max_length=7)


# ── DataQualityAgent ──────────────────────────────────────────────────────────

class QualityIssue(BaseModel):
    column: str | None = None
    issue_type: str
    severity: Literal["high", "medium", "low"]
    evidence_pointer: str = Field(
        ..., description="Which evidence metric triggered this (e.g. missingness_pct.bili)"
    )
    detail: str
    recommended_action: str


class DataQualityOutput(BaseModel):
    issues: list[QualityIssue]
    overall_quality_score: Literal["poor", "fair", "good", "excellent"]
    summary: str


# ── ClinicalSemanticsAgent ────────────────────────────────────────────────────

ColumnRole = Literal[
    "id", "time", "outcome", "covariate", "site", "arm",
    "demographic", "lab_value", "unknown"
]


class ColumnRoleAssignment(BaseModel):
    column: str
    inferred_role: ColumnRole
    confidence: Literal["high", "medium", "low"]
    rationale: str


class ClinicalSemanticsOutput(BaseModel):
    column_roles: list[ColumnRoleAssignment]
    clarifying_questions: list[str] = Field(
        ..., description="Open questions that need human/sponsor clarification"
    )
    trial_design_signals: list[str] = Field(
        ..., description="Inferred trial design clues from column names/values"
    )


# ── UnknownsAgent ─────────────────────────────────────────────────────────────

class RankedUnknown(BaseModel):
    rank: int
    question: str
    category: Literal[
        "protocol", "endpoint_definition", "data_provenance",
        "statistical_plan", "population", "regulatory", "other"
    ]
    impact: Literal["high", "medium", "low"]
    rationale: str


class ExplicitAssumption(BaseModel):
    assumption: str
    basis: str
    risk_if_wrong: Literal["high", "medium", "low"]


class RequiredDocument(BaseModel):
    document: str
    reason: str
    priority: Literal["essential", "recommended", "optional"]


class UnknownsOutput(BaseModel):
    ranked_unknowns: list[RankedUnknown]
    explicit_assumptions: list[ExplicitAssumption]
    required_documents: list[RequiredDocument]
    summary: str
    # Filled by the orchestrator after comparing unknowns to metadata answers
    unresolved_high_impact: list[str] = Field(
        default_factory=list,
        description=(
            "High-impact questions that remain unanswered after metadata "
            "resolution.  Populated by the orchestrator, NOT the LLM."
        ),
    )


# ── InsightSynthesisAgent ─────────────────────────────────────────────────────

class RankedAnalysis(BaseModel):
    rank: int
    analysis: str
    rationale: str
    evidence_citation: str


class InsightSynthesisOutput(BaseModel):
    key_findings: list[str]
    risks_and_bias_signals: list[str]
    recommended_next_analyses: list[RankedAnalysis]
    required_metadata_or_questions: list[str]


# ── JudgeAgent ────────────────────────────────────────────────────────────────

class FailedClaim(BaseModel):
    claim_text: str = Field(description="Exact quote of the problematic claim.")
    reason: str = Field(description="Why this claim fails the rubric.")
    missing_evidence: str | None = Field(
        default=None,
        description="Data key or context that would resolve the issue.",
    )
    severity: Literal["low", "medium", "high"]


class RubricScores(BaseModel):
    evidence_support: int = Field(ge=0, le=100)
    clinical_overreach: int = Field(ge=0, le=100)
    uncertainty_handling: int = Field(ge=0, le=100)
    internal_consistency: int = Field(ge=0, le=100)


class JudgeOutput(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    rubric: RubricScores
    failed_claims: list[FailedClaim] = Field(default_factory=list)
    rewrite_instructions: list[str] = Field(default_factory=list)
    judge_reasoning: str = ""


# ── Metadata (user-supplied answers to resolve unknowns) ─────────────────────

class LabUnit(BaseModel):
    """Unit description for a single numeric / lab column."""
    column: str = Field(description="Column name exactly as it appears in the dataset.")
    unit: str = Field(description="Physical unit, e.g. 'mg/dL', 'seconds', 'years'.")
    normal_range: str | None = Field(
        default=None,
        description="Reference range, e.g. '0.3–1.2 mg/dL'.  Omit if unknown.",
    )


class MetadataInput(BaseModel):
    """
    Structured answers the user provides to resolve unknowns raised by
    UnknownsAgent.  All fields are optional — supply only what you know.
    """
    status_mapping: dict[str, str] | None = Field(
        default=None,
        description=(
            "Maps status/event codes to clinical meanings, e.g. "
            "{'0': 'censored', '1': 'transplant', '2': 'death'}."
        ),
    )
    primary_endpoint: str | None = Field(
        default=None,
        description="Confirmed primary endpoint description.",
    )
    time_unit: str | None = Field(
        default=None,
        description="Unit for time-to-event columns: 'days', 'months', or 'years'.",
    )
    lab_units: list[LabUnit] | None = Field(
        default=None,
        description="Units for numeric / lab columns.",
    )
    study_design: str | None = Field(
        default=None,
        description="Confirmed study design, e.g. 'double-blind RCT'.",
    )
    treatment_arms: dict[str, str] | None = Field(
        default=None,
        description=(
            "Maps arm codes to labels, e.g. "
            "{'1': 'D-penicillamine', '2': 'placebo'}."
        ),
    )
    additional_answers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Free-form Q/A pairs for unknowns not covered above.  "
            "Keys should mirror the question text from UnknownsAgent."
        ),
    )


# ── Final Report ──────────────────────────────────────────────────────────────

class AgentRunRecord(BaseModel):
    step_number: int
    agent: str
    goal: str
    output: dict[str, Any]


class FinalReportSchema(BaseModel):
    provider: str
    model: str
    plan: PlanSchema
    agent_runs: list[AgentRunRecord]
    unknowns: UnknownsOutput
    final_insights: InsightSynthesisOutput
    judge: JudgeOutput | None = None
    # Closed-loop metadata fields (all None when no metadata supplied)
    metadata_used: MetadataInput | None = None
    final_insights_before: InsightSynthesisOutput | None = None
    final_insights_after: InsightSynthesisOutput | None = None
    judge_before: JudgeOutput | None = None
    judge_after: JudgeOutput | None = None
