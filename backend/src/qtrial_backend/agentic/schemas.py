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
