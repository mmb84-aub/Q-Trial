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


# ── Task 5 — Robustness Guardrails ───────────────────────────────────────────

GuardrailCheckType = Literal[
    "low_cardinality_numeric",
    "range_violation",
    "unit_plausibility",
    "repeated_measures",
]


class GuardrailFlag(BaseModel):
    """
    A single robustness flag produced by the deterministic guardrail checks.
    Attached to evidence as ``guardrails[i]`` citations.
    """
    check_type: GuardrailCheckType
    column: str | None = None
    severity: Literal["high", "medium", "low"]
    detail: str
    suggested_action: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class RepeatedMeasuresSchema(BaseModel):
    """Schema details when a repeated-measures / longitudinal design is inferred."""
    id_column: str
    n_subjects: int
    total_rows: int
    max_repeats_per_subject: int
    n_subjects_with_repeats: int
    likely_longitudinal: bool
    detail: str


class GuardrailReport(BaseModel):
    """
    Aggregated results from all four Task 5 robustness guardrail checks.
    Stored in FinalReportSchema and injected into evidence so all agents
    can cite ``guardrails[i]`` and ``guardrails.repeated_measures``.
    """
    flags: list[GuardrailFlag] = Field(default_factory=list)
    repeated_measures: RepeatedMeasuresSchema | None = None
    summary: str = ""
    counts_by_type: dict[str, int] = Field(default_factory=dict)


# ── Task 6 — Literature RAG (models live in tools.literature.rag to avoid circular import) ──
from qtrial_backend.tools.literature.rag import LiteratureArticle, LiteratureRAGReport  # re-export


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
        "clinical_context", "treatment", "comorbidities", "follow_up",
        "confounding", "mechanism", "population", "outcome_ascertainment", "other"
    ]
    impact: Literal["high", "medium", "low"]
    rationale: str


class ExplicitAssumption(BaseModel):
    assumption: str
    basis: str
    risk_if_wrong: Literal["high", "medium", "low"] = "medium"


class RequiredDocument(BaseModel):
    document: str
    reason: str
    priority: Literal["essential", "recommended", "optional"]


class UnknownsOutput(BaseModel):
    ranked_unknowns: list[RankedUnknown]
    explicit_assumptions: list[ExplicitAssumption]
    required_documents: list[RequiredDocument] = Field(default_factory=list)
    summary: str = ""
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


# ── Second-pass review (uploaded prior report) ──────────────────────────────

ReviewSeverity = Literal["critical", "high", "medium", "low"]
ReviewOutcome = Literal["accept", "revise", "reject", "needs_more_context"]


class PriorReportSection(BaseModel):
    """Normalized section extracted from an uploaded prior report."""

    section_id: str = Field(
        ..., description="Stable local id, e.g. 'sec_1'."
    )
    title: str = ""
    content: str = ""
    source_span: str | None = Field(
        default=None,
        description="Optional source hint such as line range or page marker.",
    )


class PriorReportClaim(BaseModel):
    """Atomic structured claim extracted from a prior report section."""

    claim_id: str = Field(..., description="Stable local id, e.g. 'claim_1'.")
    section_id: str = Field(..., description="Which section this claim came from, e.g. 'sec_2'.")
    section_title: str = "" # The title of the section for context
    claim_text: str = Field(..., description="The normalized claim statement.")
    claim_type: Literal[
        "key_finding",
        "risk_signal",
        "bias_signal",
        "grounded_finding",
        "recommendation",
        "assumption",
        "methodology",
        "limitation",
        "unknown",
    ] = "unknown"
    source_excerpt: str = Field(
        default="",
        description="The original excerpt from the report this claim was extracted from.",
    )
    parser_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the extraction and typing (0.0-1.0).",
    )


class PriorReportNormalized(BaseModel):
    """Canonical representation of an uploaded prior report."""

    source_name: str = ""
    source_format: Literal["json", "markdown", "text", "unknown"] = "unknown"
    raw_text: str = ""
    sections: list[PriorReportSection] = Field(default_factory=list)
    extracted_claims: list[str] = Field(
        default_factory=list,
        description="Optional initial claim strings detected in normalization (legacy/flat format).",
    )
    extracted_atomic_claims: list[PriorReportClaim] = Field(
        default_factory=list,
        description="Structured atomic claims extracted via deterministic parsing.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecondPassReviewIssue(BaseModel):
    """One issue found while reviewing a prior report against current evidence."""

    issue_id: str = Field(..., description="Stable id, e.g. 'issue_1'.")
    severity: ReviewSeverity
    category: Literal[
        "unsupported_claim",
        "overreach",
        "missing_uncertainty",
        "internal_inconsistency",
        "citation_gap",
        "other",
    ]
    finding: str
    prior_report_citation: str = Field(
        ..., description="Section reference in normalized prior report, e.g. sec_2."
    )
    expected_evidence_citation: str | None = Field(
        default=None,
        description="Resolvable evidence key expected to support the claim.",
    )
    recommendation: str = ""
    
    # ── Optional fields for future bounded LLM refinement (Step 4B.2+) ──
    llm_refinement_comment: str | None = Field(
        default=None,
        description=(
            "Optional LLM-generated comment contextualizing this deterministic issue. "
            "Added by future bounded refinement layer; does not change verdict or recommendation."
        ),
    )


class SecondPassReviewOutput(BaseModel):
    """
    Structured output for the second-pass review stage.
    
    Deterministic review verdict fields (outcome, summary, issues, accepted/revised/dropped_claims)
    are the source of truth and remain unchanged by any future LLM refinement layer.
    
    Optional refinement fields (delta_summary, refinement_notes, follow_up_questions, llm_refinement_applied)
    are additive and populated only if bounded LLM refinement is applied (future Step 4B.2+).
    """

    # ── Deterministic verdict fields (source of truth; never modified by LLM)
    outcome: ReviewOutcome = "needs_more_context"
    summary: str = ""
    issues: list[SecondPassReviewIssue] = Field(default_factory=list)
    accepted_claims: list[str] = Field(default_factory=list)
    revised_claims: list[str] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)
    
    # ── Optional fields for future bounded LLM refinement (Step 4B.2+) ──
    delta_summary: str | None = Field(
        default=None,
        description=(
            "Optional LLM-generated focused summary of changes from prior report to current analysis. "
            "Example: 'Confirmed 3 key findings; found 2 disputed claims; unprecedented 1 novel signal.' "
            "Only populated if bounded LLM refinement is applied."
        ),
    )
    refinement_notes: str | None = Field(
        default=None,
        description=(
            "Optional LLM-generated contextual notes on the deterministic verdict. "
            "May include clinical framing, data-backed rebuttals, or uncertainty hedging. "
            "Does not change verdict labels; enriches explanation only."
        ),
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Optional LLM-generated clarifying questions for clinician follow-up. "
            "Examples: 'Was [treatment] adjusted during follow-up?', 'Can you confirm [outcome] coding?'. "
            "Only populated if bounded LLM refinement is applied."
        ),
    )
    llm_refinement_applied: bool = Field(
        default=False,
        description=(
            "Flag indicating whether bounded LLM refinement has been applied to this review. "
            "True only if delta_summary, refinement_notes, or follow_up_questions are populated."
        ),
    )


# ── Upstream tool-call record (from AgentLoop / static pipeline) ──────────────

class ToolCallRecord(BaseModel):
    """
    A single tool call made by the upstream statistical AgentLoop.
    Mirrors the structure of AgentLoop.tool_log entries but typed and
    extended with a stable citation_alias set by the orchestrator.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = Field(
        default=None,
        description="Raw tool result (dict/list/scalar). May be truncated.",
    )
    error: str | None = Field(
        default=None,
        description="Error message when the tool call failed.",
    )
    citation_alias: str = Field(
        default="",
        description=(
            "Stable indexed alias assigned by the orchestrator, "
            "e.g. 'tool_log[0]'.  Agents and Judge cite via this alias."
        ),
    )


# ── Task 4A — Reasoning State schemas ────────────────────────────────────────

class EvidenceSupportEntry(BaseModel):
    """One piece of evidence that supports or contradicts a hypothesis."""

    citation_key: str = Field(
        ...,
        description=(
            "Resolvable path: 'evidence.*', 'preview.*', or "
            "'tool_log[i]' alias."
        ),
    )
    raw_value: Any = Field(
        default=None,
        description="The numeric/dict value pulled from evidence for this key.",
    )
    supports_claim: bool = Field(
        ...,
        description="True when the evidence supports the parent claim/hypothesis.",
    )
    explanation: str = Field(
        ...,
        description="One sentence linking the numeric value to the claim.",
    )


class CandidateHypothesis(BaseModel):
    """A single candidate explanatory or analytic hypothesis."""

    hypothesis_id: str = Field(
        ..., description="Stable slug, e.g. 'h1', 'h2'."
    )
    statement: str
    source_agent: AgentName = Field(
        ..., description="Agent that produced this hypothesis."
    )
    confidence: Literal["high", "medium", "low"]
    evidence_support: list[EvidenceSupportEntry] = Field(default_factory=list)
    contradictions: list[str] = Field(
        default_factory=list,
        description="Citation keys that contradict this hypothesis.",
    )
    status: Literal[
        "candidate", "supported", "contradicted", "deferred", "dropped"
    ] = "candidate"


class ClaimDraft(BaseModel):
    """A claim drafted during the reasoning loop, pending deterministic validation."""

    claim_id: str = Field(..., description="Stable slug, e.g. 'c1', 'c2'.")
    text: str
    hypothesis_ids: list[str] = Field(
        default_factory=list,
        description="Hypothesis IDs this claim builds on.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "Resolvable citation keys used inline in `text`.  "
            "Validated deterministically against valid_citation_keys."
        ),
    )
    confidence: Literal["high", "medium", "low"]
    validation_status: Literal[
        "pending", "valid", "flagged", "rejected"
    ] = "pending"
    flag_reasons: list[str] = Field(
        default_factory=list,
        description="Populated by deterministic validators, NOT by the LLM.",
    )


class InternalEvidenceSnippet(BaseModel):
    """One retrieved snippet used for internal post-reasoning verification."""

    rank: int = Field(..., ge=1)
    score: float
    snippet: str
    source_type: str = ""
    source_name: str = ""
    chunk_id: str = ""
    matched_terms: list[str] = Field(default_factory=list)


class ClaimInternalVerification(BaseModel):
    """Verification result for one claim against internal BM25 evidence."""

    claim_id: str
    claim_text: str
    validation_status: Literal["pending", "valid", "flagged", "rejected"]
    support_status: Literal["supported", "weakly_supported", "unsupported", "unclear"]
    rationale: str = ""
    top_evidence: list[InternalEvidenceSnippet] = Field(default_factory=list)


class InternalVerificationSummary(BaseModel):
    """Aggregate statistics for internal claim verification."""

    total_claims: int = 0
    supported: int = 0
    weakly_supported: int = 0
    unsupported: int = 0
    unclear: int = 0
    index_chunks: int = 0


class Contradiction(BaseModel):
    """A tracked contradiction between two hypotheses."""

    hypothesis_id_a: str
    hypothesis_id_b: str
    description: str
    severity: Literal["high", "medium", "low"]
    resolved: bool = False
    resolution_note: str | None = None


ReasoningStepType = Literal[
    "evidence_scan",
    "hypothesis_gen",
    "claim_draft",
    "validation",
    "contradiction_check",
    "refinement",
    "stop_evaluation",
]


class ReasoningStepLog(BaseModel):
    """One recorded step in the reasoning trace."""

    step_index: int = Field(..., ge=0)
    step_type: ReasoningStepType
    inputs: list[str] = Field(
        default_factory=list,
        description="Keys, IDs, or labels consumed by this step.",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Keys, IDs, or labels produced by this step.",
    )
    notes: str = ""


class StopCondition(BaseModel):
    """Whether the reasoning loop is complete and why."""

    met: bool
    reason: str = ""
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="Issues that prevent stopping when met=False.",
    )


class ConfidenceSummary(BaseModel):
    """Aggregate confidence assessment computed deterministically."""

    overall: Literal["high", "medium", "low", "inconclusive"]
    num_supported_claims: int = Field(..., ge=0)
    num_flagged_claims: int = Field(..., ge=0)
    num_rejected_claims: int = Field(..., ge=0)
    limiting_factors: list[str] = Field(
        default_factory=list,
        description="Reasons why overall confidence is capped.",
    )


# ── Task 4C — Dynamic Hypothesis Generation schemas ───────────────────────

class FalsificationCheck(BaseModel):
    """A testable check generated by the LLM for one hypothesis."""

    hypothesis_id: str = Field(
        ..., description="Matches a hypothesis_id from HypothesisGenerationOutput.",
    )
    test_description: str
    expected_if_true: str
    expected_if_false: str
    citation_key: str = Field(
        ..., description="Resolvable citation key used for this check.",
    )
    verdict: Literal["supports", "contradicts", "inconclusive"] = "inconclusive"


class HiddenQuestion(BaseModel):
    """A high-impact question the pipeline didn't originally ask."""

    question: str
    rationale: str
    impact: Literal["high", "medium", "low"]
    category: str
    suggested_data_source: str = ""


class LLMHypothesis(BaseModel):
    """Raw hypothesis as returned by the LLM (pre-validation)."""

    hypothesis_id: str = Field(..., description="Stable slug, e.g. 'h1'.")
    statement: str
    confidence: Literal["high", "medium", "low"]
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Must be from valid_citation_keys only.",
    )
    rationale: str = ""


# ── Task 4B — Tool Dispatch schemas ──────────────────────────────────────────

ToolDispatchType = Literal[
    "baseline_balance",
    "survival_analysis",
    "missing_by_group",
    "group_statistics",
    "distribution_check",
]


class ToolDispatchRequest(BaseModel):
    """
    A structured request for a deterministic stats tool, produced by the
    LLM hypothesis engine.  The dispatcher maps this to an actual registered
    tool call.
    """
    tool_type: ToolDispatchType = Field(
        ...,
        description=(
            "The type of analysis to run. Allowed values: "
            "baseline_balance | survival_analysis | missing_by_group | "
            "group_statistics | distribution_check"
        ),
    )
    hypothesis_id: str = Field(
        ..., description="Hypothesis ID this investigation supports."
    )
    columns: list[str] = Field(
        ..., description="Column names from the dataset that this analysis requires."
    )
    group_column: str | None = Field(
        default=None,
        description="Column to use as the grouping variable (e.g. treatment arm).",
    )
    rationale: str = Field(
        ..., description="One sentence: why this tool call is needed to test or refute the hypothesis."
    )
    priority: Literal["high", "medium", "low"] = "medium"


class ToolDispatchResult(BaseModel):
    """Result of one hypothesis-driven tool dispatch call."""
    request: ToolDispatchRequest
    tool_called: str = Field(..., description="Registered tool name that was executed.")
    args_used: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    citation_alias: str = Field(
        default="",
        description="Stable citation alias, e.g. 'dispatched[0]'.",
    )


class HypothesisGenerationOutput(BaseModel):
    """Schema-validated LLM output for dynamic hypothesis generation."""

    hypotheses: list[LLMHypothesis]
    falsification_checks: list[FalsificationCheck] = Field(default_factory=list)
    hidden_questions: list[HiddenQuestion] = Field(default_factory=list)
    tool_dispatch_requests: list[ToolDispatchRequest] = Field(
        default_factory=list,
        description=(
            "Structured requests for deterministic stats tools to run in order "
            "to test or falsify the generated hypotheses."
        ),
    )


class ReasoningState(BaseModel):
    """
    Full structured reasoning state for one pipeline run.
    Persisted under FinalReportSchema.reasoning_state.
    All lists default to empty so early-stage runs can be serialised
    without requiring downstream state to be populated.
    """

    run_id: str = Field(
        default="",
        description="Echoes provider+timestamp for traceability.",
    )
    hypotheses: list[CandidateHypothesis] = Field(default_factory=list)
    claims: list[ClaimDraft] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    hidden_questions: list[HiddenQuestion] = Field(
        default_factory=list,
        description="High-impact questions surfaced by dynamic hypothesis generation (Task 4C).",
    )
    falsification_checks: list[FalsificationCheck] = Field(
        default_factory=list,
        description="Testable checks for generated hypotheses (Task 4C).",
    )
    step_log: list[ReasoningStepLog] = Field(default_factory=list)
    stop_condition: StopCondition = Field(
        default_factory=lambda: StopCondition(met=False, reason="not started"),
    )
    confidence_summary: ConfidenceSummary | None = None
    valid_citation_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Built at initialisation time from preview + evidence + tool_log. "
            "Used by deterministic claim validators."
        ),
    )
    dispatched_tool_results: list[ToolDispatchResult] = Field(
        default_factory=list,
        description=(
            "Results from hypothesis-driven tool dispatch calls (Task 4B). "
            "Each entry maps one ToolDispatchRequest to its empirical output."
        ),
    )
    internal_verification: list[ClaimInternalVerification] = Field(
        default_factory=list,
        description=(
            "Post-reasoning internal evidence verification results over the "
            "runtime BM25 corpus built from statistical outputs."
        ),
    )
    internal_verification_summary: InternalVerificationSummary | None = Field(
        default=None,
        description="Aggregate counts for internal BM25 claim verification.",
    )


# ── ReasoningAction (used by Task 4B executor) ────────────────────────────────

ReasoningActionType = Literal[
    "scan_evidence",
    "generate_hypothesis",
    "draft_claim",
    "validate_claims",
    "check_contradictions",
    "refine_claims",
    "evaluate_stop",
    "finalize",
]


class ReasoningAction(BaseModel):
    """
    A single action produced by the reasoning executor (Task 4B).
    Defined here so it can be persisted in outputs and referenced by
    forward-looking integration points added in Task 4A.
    """

    action_type: ReasoningActionType
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_output_keys: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: dict[str, Any] | None = None
    error_message: str | None = None


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
    # Upstream statistical context — None when not wired from AgentLoop
    prior_analysis_report: str | None = Field(
        default=None,
        description=(
            "Markdown report from the upstream statistical AgentLoop. "
            "Propagated as PRIOR_ANALYSIS_REPORT to all reasoning agents."
        ),
    )
    tool_log: list[ToolCallRecord] | None = Field(
        default=None,
        description=(
            "Typed tool-call log from the upstream AgentLoop, with stable "
            "citation aliases (tool_log[i]) set by the orchestrator."
        ),
    )
    # Task 4A — structured reasoning state (None until 4B executor is wired)
    reasoning_state: ReasoningState | None = Field(
        default=None,
        description=(
            "Structured reasoning state produced by the Task 4A/4B reasoning "
            "engine.  None in runs that predate Task 4B wiring."
        ),
    )
    # Task 5 — robustness guardrails (None in runs that predate Task 5)
    guardrail_report: GuardrailReport | None = Field(
        default=None,
        description=(
            "Robustness guardrail flags: low-cardinality numerics, range "
            "violations, unit plausibility, and repeated-measures schema inference."
        ),
    )
    # Task 6 — literature RAG (None when no hypotheses or retrieval disabled)
    literature_report: LiteratureRAGReport | None = Field(
        default=None,
        description=(
            "Hypothesis-driven literature retrieval results from PubMed / "
            "Semantic Scholar. Articles appear as lit[i] citations in insights."
        ),
    )
    # New pipeline fields — all optional so existing callers are unaffected
    study_context: str | None = Field(
        default=None,
        description="Plain-language study description provided by the clinician.",
    )
    grounded_findings: GroundedFindingsSchema | None = Field(
        default=None,
        description="Literature-grounded findings with synthesis outputs.",
    )
    reproducibility_log: ReproducibilityLog | None = Field(
        default=None,
        description="Full audit log for this analysis run.",
    )
    synthesis_quality_score: SynthesisQualityScore | None = Field(
        default=None,
        description="Self-assessment score from the synthesis LLM call.",
    )
    second_pass_review: SecondPassReviewOutput | None = Field(
        default=None,
        description=(
            "Optional second-pass review output comparing prior/external report against V1 synthesis. "
            "None when no prior report is uploaded. Includes deterministic verdicts + optional LLM refinement."
        ),
    )
    treatment_columns_excluded: list[str] = Field(
        default_factory=list,
        description="Column names excluded as treatment group assignments.",
    )


# ── New schemas: Clinical Search Terms, Evidence Strength, Grounded Findings ──

class ClinicalSearchTerm(BaseModel):
    """A statistical finding translated into a clinically phrased query string."""
    source_finding: str
    term: str
    study_context_used: str
    translation_failed: bool = False
    failure_note: str | None = None


class EvidenceStrengthScore(BaseModel):
    """Composite evidence quality score derived from recency, study type, and sample size."""
    score: int = Field(..., ge=0, le=100)
    label: Literal["Strong", "Moderate", "Weak", "Insufficient"]
    plain_language: str  # e.g. "Strong — based on a 2022 Cochrane meta-analysis of 10,000 patients"
    year: int | None = None
    study_type: str  # "meta-analysis" | "RCT" | "cohort" | "case study" | "unknown"
    sample_size: int | None = None


class GroundedFinding(BaseModel):
    """A single statistical finding with literature grounding and disclosure metadata."""
    finding_text: str
    grounding_status: Literal["Supported", "Contradicted", "Novel"]
    citations: list[LiteratureArticle] = Field(default_factory=list)
    evidence_strength: EvidenceStrengthScore | None = None
    novel_statement: str | None = None
    literature_skipped: bool = False
    literature_skip_note: str | None = None
    test_selection_rationale: str | None = None
    missingness_disclosure: str | None = None
    confidence_warning: str | list[str] | None = None


# ── New schemas: Research Questions, Synthesis, Missingness ──────────────────

class ResearchQuestion(BaseModel):
    question: str
    source_finding: str


class ControlVariable(BaseModel):
    variable: str
    reason: str


class SynthesisOutput(BaseModel):
    future_trial_hypothesis: str
    endpoint_improvement_recommendations: list[str] = Field(default_factory=list)
    recommended_sample_size: str
    variables_to_control: list[ControlVariable] = Field(default_factory=list)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)
    narrative_summary: str = Field(
        default="",
        description="Plain-language prose overview produced by the synthesis LLM call.",
    )


class ExcludedColumn(BaseModel):
    column: str
    missingness_rate: float
    reason: str = "exceeds 50% missingness threshold"


class HighMissingnessColumn(BaseModel):
    column: str
    missingness_rate: float  # 0.20–0.50


class MissingnessDisclosure(BaseModel):
    column: str
    missingness_rate: float
    rows_dropped: int
    action: Literal["excluded", "high_missingness_section", "listwise_deletion"]


class GroundedFindingsSchema(BaseModel):
    findings: list[GroundedFinding] = Field(default_factory=list)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)
    synthesis: SynthesisOutput | None = None
    excluded_columns: list[ExcludedColumn] = Field(default_factory=list)
    high_missingness_columns: list[HighMissingnessColumn] = Field(default_factory=list)


# ── New schemas: Synthesis Quality, Reproducibility Log ──────────────────────

class SynthesisQualityScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    rerun_triggered: bool = False


class LLMCallRecord(BaseModel):
    call_id: str
    stage: str  # e.g. "statistical_agent", "cst_translation", "synthesis"
    model: str
    temperature: float
    seed: int | None = None
    prompt_hash: str   # SHA-256 of the prompt
    response_hash: str  # SHA-256 of the response


class LiteratureQueryRecord(BaseModel):
    source: Literal["pubmed", "cochrane", "clinicaltrials", "semantic_scholar"]
    query_string: str
    results_count: int
    cached: bool
    error: str | None = None


class ReproducibilityLog(BaseModel):
    run_id: str
    timestamp: str  # ISO 8601
    study_context: str
    seed: int
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    literature_queries: list[LiteratureQueryRecord] = Field(default_factory=list)
    clinical_search_terms: list[ClinicalSearchTerm] = Field(default_factory=list)
    tool_call_log: list[ToolCallRecord] = Field(default_factory=list)
    synthesis_quality_score: SynthesisQualityScore | None = None
