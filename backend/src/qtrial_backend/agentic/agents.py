from __future__ import annotations

import json
import textwrap
from typing import Any

from qtrial_backend.agentic.schemas import (
    DataQualityOutput,
    ClinicalSemanticsOutput,
    UnknownsOutput,
    InsightSynthesisOutput,
    PlanStep,
    ToolCallRecord,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName


# ── shared helpers ────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        ).strip()
    return text


def _call_llm_for_json(
    provider: ProviderName,
    system: str,
    user: str,
    schema_class: Any,
    *,
    retry: bool = True,
) -> Any:
    client = get_client(provider)
    req = LLMRequest(system_prompt=system, user_prompt=user, payload={})
    resp = client.generate(req)
    raw = _strip_fences(resp.text)

    try:
        data = json.loads(raw)
        return schema_class.model_validate(data)
    except Exception as exc:
        if not retry:
            raise ValueError(
                f"{schema_class.__name__} returned invalid JSON after retry.\n"
                f"Raw:\n{raw}\nError: {exc}"
            ) from exc

        fix_req = LLMRequest(
            system_prompt=system,
            user_prompt=(
                "Your previous response was not valid JSON matching the required schema.\n"
                f"Error: {exc}\n\n"
                f"Previous response:\n{raw}\n\n"
                "Fix it and return ONLY valid JSON. No markdown, no explanation."
            ),
            payload={},
        )
        fix_resp = client.generate(fix_req)
        fixed_raw = _strip_fences(fix_resp.text)
        data = json.loads(fixed_raw)
        return schema_class.model_validate(data)


# ── Upstream context block helpers ────────────────────────────────────────

def _build_prior_analysis_block(
    prior_analysis_report: str | None,
    tool_log: list[ToolCallRecord] | None,
) -> str:
    """
    Build the two new upstream-context blocks injected above DATASET_EVIDENCE
    in every reasoning agent prompt.

    Returns an empty string when both inputs are None (backward-compatible).
    """
    if not prior_analysis_report and not tool_log:
        return ""

    parts: list[str] = []

    if prior_analysis_report:
        parts.append(
            "PRIOR_ANALYSIS_REPORT "
            "(Markdown — produced by upstream statistical agent):\n"
            + prior_analysis_report.strip()
            + "\n"
        )

    if tool_log:
        compact = [
            {
                "alias": rec.citation_alias,
                "tool": rec.tool_name,
                "result": rec.result,
                **(  {"error": rec.error} if rec.error else {}  ),
            }
            for rec in tool_log
        ]
        parts.append(
            "TOOL_CALL_LOG "
            "(structured tool outputs — cite via alias, e.g. tool_log[0]):\n"
            + json.dumps(compact, indent=2, ensure_ascii=False, default=str)
            + "\n"
        )

    return "\n".join(parts)


def _build_tool_log_citation_hints(
    tool_log: list[ToolCallRecord] | None,
) -> str:
    """
    Return a short list of valid tool_log citation aliases so that
    InsightSynthesisAgent and Judge know which alias strings are resolvable.
    """
    if not tool_log:
        return "      (no upstream tool log available)"
    return "\n".join(
        f"      {rec.citation_alias}: {rec.tool_name}"
        for rec in tool_log
    )


# ── DataQualityAgent ──────────────────────────────────────────────────────────

_DQ_SYSTEM = textwrap.dedent("""\
    You are DataQualityAgent, a specialist in clinical trial data quality assessment.
    You receive a dataset preview and automated evidence metrics.
    Respond with ONLY valid JSON matching the exact schema. No markdown, no commentary.
""")

_DQ_USER = textwrap.dedent("""\
    Analyse the data quality issues in this clinical trial dataset.

    Required JSON schema:
    {{
      "issues": [
        {{
          "column": "<column name or null if dataset-level>",
          "issue_type": "<high_missingness|duplicate_ids|constant_column|outlier|imbalance>",
          "severity": "<high|medium|low>",
          "evidence_pointer": "<resolvable path — follow EVIDENCE POINTER RULES>",
          "detail": "<precise description citing numeric values from evidence>",
          "recommended_action": "<concrete next step>"
        }}
      ],
      "overall_quality_score": "<poor|fair|good|excellent>",
      "summary": "<two-sentence summary>"
    }}

    EVIDENCE POINTER RULES (mandatory — violations will cause rejection):
      outliers:       "evidence.outlier_flags.<column>"      e.g. "evidence.outlier_flags.bili"
      missingness:    "evidence.missingness_pct.<column>"    e.g. "evidence.missingness_pct.albumin"
      id duplicates:  "evidence.id_duplicates.<column>"      e.g. "evidence.id_duplicates.id"
      constant cols:  "evidence.constant_columns"
      distributions:  "evidence.categorical_distributions.<column>"
      tool log refs:  "tool_log[i]"                          e.g. "tool_log[0]"
      Do NOT invent paths. Do NOT use keys absent from the evidence JSON or tool log.

    Step goal: {goal}

    {prior_analysis_block}
    DATASET_PREVIEW (JSON):
    {preview}

    DATASET_EVIDENCE (JSON):
    {evidence}
""")


def run_data_quality_agent(
    preview: dict,
    evidence: dict,
    step: PlanStep,
    provider: ProviderName,
    *,
    prior_analysis_report: str | None = None,
    tool_log: list[ToolCallRecord] | None = None,
) -> DataQualityOutput:
    prior_block = _build_prior_analysis_block(prior_analysis_report, tool_log)
    user = _DQ_USER.format(
        goal=step.goal,
        prior_analysis_block=prior_block,
        preview=json.dumps(preview, indent=2, ensure_ascii=False),
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False),
    )
    return _call_llm_for_json(provider, _DQ_SYSTEM, user, DataQualityOutput)


# ── ClinicalSemanticsAgent ────────────────────────────────────────────────────

_CS_SYSTEM = textwrap.dedent("""\
    You are ClinicalSemanticsAgent, a specialist in clinical trial data semantics
    and regulatory standards (ICH E9, CDISC concepts).
    Respond with ONLY valid JSON matching the exact schema. No markdown, no commentary.
""")

_CS_USER = textwrap.dedent("""\
    Infer the clinical role of each column and identify open questions.

    Allowed column roles (use exactly):
      id | time | outcome | covariate | site | arm | demographic | lab_value | unknown

    TONE RULE: When outcome semantics are not confirmed (e.g. status coding is unknown),
    do NOT use the phrase "prognostic factor". Instead use
    "association signal pending status definition" in rationale and trial_design_signals.

    Required JSON schema:
    {{
      "column_roles": [
        {{
          "column": "<column name>",
          "inferred_role": "<role>",
          "confidence": "<high|medium|low>",
          "rationale": "<one sentence — see TONE RULE>"
        }}
      ],
      "clarifying_questions": ["<question>", ...],
      "trial_design_signals": ["<signal — see TONE RULE>", ...]
    }}

    Step goal: {goal}

    {prior_analysis_block}
    DATASET_PREVIEW (JSON):
    {preview}

    DATASET_EVIDENCE (JSON):
    {evidence}
""")


def run_clinical_semantics_agent(
    preview: dict,
    evidence: dict,
    step: PlanStep,
    provider: ProviderName,
    *,
    prior_analysis_report: str | None = None,
    tool_log: list[ToolCallRecord] | None = None,
) -> ClinicalSemanticsOutput:
    prior_block = _build_prior_analysis_block(prior_analysis_report, tool_log)
    user = _CS_USER.format(
        goal=step.goal,
        prior_analysis_block=prior_block,
        preview=json.dumps(preview, indent=2, ensure_ascii=False),
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False),
    )
    return _call_llm_for_json(provider, _CS_SYSTEM, user, ClinicalSemanticsOutput)


# ── UnknownsAgent ─────────────────────────────────────────────────────────────

_UA_SYSTEM = textwrap.dedent("""\
    You are UnknownsAgent, a specialist in identifying gaps, ambiguities, and assumptions
    in clinical trial datasets prior to statistical analysis.

    Your role is to:
    1. Surface ranked open questions that block or complicate reliable analysis.
    2. Make explicit every assumption the pipeline is currently relying on.
    3. List documents or metadata that would resolve the top unknowns.

    Respond with ONLY valid JSON matching the exact schema. No markdown, no commentary.
""")

_UA_USER = textwrap.dedent("""\
    Given the dataset preview, evidence, and outputs from DataQualityAgent and
    ClinicalSemanticsAgent, identify all unknowns, explicit assumptions, and
    required documents.

    Allowed category values (use exactly one):
      protocol | endpoint_definition | data_provenance | statistical_plan |
      population | regulatory | other

    Allowed impact values: high | medium | low
    Allowed risk_if_wrong values: high | medium | low
    Allowed priority values: essential | recommended | optional

    Required JSON schema:
    {{
      "ranked_unknowns": [
        {{
          "rank": 1,
          "question": "<specific open question>",
          "category": "<category>",
          "impact": "<high|medium|low>",
          "rationale": "<why this unknown matters for analysis>"
        }}
      ],
      "explicit_assumptions": [
        {{
          "assumption": "<what the pipeline is currently assuming>",
          "basis": "<evidence or reasoning behind this assumption>",
          "risk_if_wrong": "<high|medium|low>"
        }}
      ],
      "required_documents": [
        {{
          "document": "<document or metadata name>",
          "reason": "<what it resolves>",
          "priority": "<essential|recommended|optional>"
        }}
      ],
      "summary": "<two-sentence summary of the unknowns landscape>"
    }}

    Step goal: {goal}

    {prior_analysis_block}
    DATASET_PREVIEW (JSON):
    {preview}

    DATASET_EVIDENCE (JSON):
    {evidence}

    DATA_QUALITY_OUTPUT (JSON):
    {dq_output}

    CLINICAL_SEMANTICS_OUTPUT (JSON):
    {cs_output}
""")


def run_unknowns_agent(
    preview: dict,
    evidence: dict,
    dq_output: dict | None,
    cs_output: dict | None,
    step: PlanStep,
    provider: ProviderName,
    *,
    prior_analysis_report: str | None = None,
    tool_log: list[ToolCallRecord] | None = None,
) -> UnknownsOutput:
    prior_block = _build_prior_analysis_block(prior_analysis_report, tool_log)
    user = _UA_USER.format(
        goal=step.goal,
        prior_analysis_block=prior_block,
        preview=json.dumps(preview, indent=2, ensure_ascii=False),
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False),
        dq_output=json.dumps(dq_output or {}, indent=2, ensure_ascii=False),
        cs_output=json.dumps(cs_output or {}, indent=2, ensure_ascii=False),
    )
    return _call_llm_for_json(provider, _UA_SYSTEM, user, UnknownsOutput)


# ── InsightSynthesisAgent ─────────────────────────────────────────────────────

_IS_SYSTEM = textwrap.dedent("""\
    You are InsightSynthesisAgent, the final stage of a clinical trial data review pipeline.

    CITATION RULES (strictly enforced):
    - evidence_citation fields MUST use ONLY strings from the RESOLVABLE_CITATIONS block
      OR valid aliases from the TOOL_LOG_CITATIONS block.
    - For correlations: copy the indexed form verbatim, e.g.
        "top_correlations[0]: (albumin, status)=0.895"
      NEVER write "top_correlations.albumin_status" or "top_correlations.status_time"
      because top_correlations is a LIST — dotted column paths on it are invalid.
    - For tool log: use the alias verbatim, e.g. "tool_log[0]", "tool_log[2]"
    - Multiple citations: separate with " | "
    - Do NOT invent citation paths not present in RESOLVABLE_CITATIONS or TOOL_LOG_CITATIONS.

    TONE RULES:
    - When status column coding is unconfirmed, replace "prognostic factor" with
      "association signal pending status definition".
    - Do not assert causal relationships from correlation evidence alone.
    - When unresolved_high_impact unknowns exist, hedge conclusions with
      "pending confirmation of <unknown topic>".
    - When PRIOR_ANALYSIS_REPORT is present, ground key findings in its specific
      statistics (p-values, HRs, CIs) rather than re-stating generic patterns.
    - When TOOL_CALL_LOG is present, prefer citing the tool log alias over generic
      evidence keys when both cover the same finding.

    Respond with ONLY valid JSON matching the exact schema. No markdown, no commentary.
""")

_IS_USER = textwrap.dedent("""\
    Synthesise final insights from all prior analysis.

    Required JSON schema:
    {{
      "key_findings": ["<finding with inline citation from RESOLVABLE_CITATIONS or tool_log alias>", ...],
      "risks_and_bias_signals": ["<risk with inline citation>", ...],
      "recommended_next_analyses": [
        {{
          "rank": 1,
          "analysis": "<analysis name>",
          "rationale": "<why — follow TONE RULES>",
          "evidence_citation": "<verbatim string(s) from RESOLVABLE_CITATIONS or tool_log[i]>"
        }}
      ],
      "required_metadata_or_questions": ["<question>", ...]
    }}

    RESOLVABLE_CITATIONS (copy these strings verbatim — do not paraphrase or invent):
{citations}

    TOOL_LOG_CITATIONS (use these aliases verbatim in evidence_citation fields):
{tool_log_citations}

    {prior_analysis_block}
    DATASET_PREVIEW (JSON):
    {preview}

    DATASET_EVIDENCE (JSON):
    {evidence}

    DATA_QUALITY_OUTPUT (JSON):
    {dq_output}

    CLINICAL_SEMANTICS_OUTPUT (JSON):
    {cs_output}

    UNKNOWNS_OUTPUT (JSON):
    {unknowns_output}
""")


def _build_citation_block(citations: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for section, items in citations.items():
        for item in items:
            lines.append(f"      {item}")
    return "\n".join(lines) if lines else "      (none computed)"


def run_insight_synthesis_agent(
    preview: dict,
    evidence: dict,
    dq_output: dict | None,
    cs_output: dict | None,
    provider: ProviderName,
    *,
    citations: dict[str, list[str]] | None = None,
    unknowns_output: dict | None = None,
    prior_analysis_report: str | None = None,
    tool_log: list[ToolCallRecord] | None = None,
) -> InsightSynthesisOutput:
    prior_block = _build_prior_analysis_block(prior_analysis_report, tool_log)
    citation_block = _build_citation_block(citations or {})
    tool_log_citations = _build_tool_log_citation_hints(tool_log)
    user = _IS_USER.format(
        citations=citation_block,
        tool_log_citations=tool_log_citations,
        prior_analysis_block=prior_block,
        preview=json.dumps(preview, indent=2, ensure_ascii=False),
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False),
        dq_output=json.dumps(dq_output or {}, indent=2, ensure_ascii=False),
        cs_output=json.dumps(cs_output or {}, indent=2, ensure_ascii=False),
        unknowns_output=json.dumps(unknowns_output or {}, indent=2, ensure_ascii=False),
    )
    return _call_llm_for_json(provider, _IS_SYSTEM, user, InsightSynthesisOutput)
