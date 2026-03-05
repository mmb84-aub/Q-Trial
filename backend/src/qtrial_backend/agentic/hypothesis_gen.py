"""
Task 4C — Dynamic Hypothesis & Question Generation Layer.

Uses a single LLM call to produce:
  1. Candidate hypotheses with evidence citations
  2. Falsification checks (one per hypothesis)
  3. Hidden questions ("questions we didn't know to ask")

The LLM output is strictly schema-validated (Pydantic), then passed
through the Task 4A deterministic validators (citation validation,
endpoint lockdown, treatment-effect guard, high-confidence guard).

Results are merged into an existing ``ReasoningState`` produced by
the Task 4B deterministic reasoning engine.
"""
from __future__ import annotations

import json
import textwrap

from qtrial_backend.agentic.reasoning import (
    append_reasoning_step,
    compute_confidence_summary,
    validate_all_claims,
    validate_claim_citations,
)
from qtrial_backend.agentic.schemas import (
    CandidateHypothesis,
    ClaimDraft,
    EvidenceSupportEntry,
    FalsificationCheck,
    HiddenQuestion,
    HypothesisGenerationOutput,
    MetadataInput,
    ReasoningState,
    StopCondition,
    ToolCallRecord,
)
from qtrial_backend.core.types import ProviderName


# ── Prompt templates ──────────────────────────────────────────────────────────

_HYPO_SYSTEM = textwrap.dedent("""\
    You are HypothesisGeneratorAgent, a clinical trial reasoning specialist.
    Given a dataset preview, computed evidence, agent outputs, and optional
    context, generate candidate hypotheses, falsification checks, and hidden
    questions.

    Respond with ONLY valid JSON matching the schema below.
    No markdown fences, no commentary outside the JSON object.

    CITATION RULES (mandatory — violations cause rejection):
    - Every entry in evidence_citations and every falsification citation_key
      MUST come from the VALID_CITATION_KEYS list in the user message.
    - Do NOT invent citation keys.
    - Allowed formats: evidence.*, preview.*, tool_log[i]
""")

_HYPO_USER = textwrap.dedent("""\
    Generate candidate hypotheses, falsification checks, and hidden questions
    for this clinical trial dataset.

    Required JSON schema:
    {{
      "hypotheses": [
        {{
          "hypothesis_id": "h1",
          "statement": "<testable clinical hypothesis>",
          "confidence": "<high|medium|low>",
          "evidence_citations": ["<citation_key>", ...],
          "rationale": "<one sentence explaining why>"
        }}
      ],
      "falsification_checks": [
        {{
          "hypothesis_id": "<must match a hypothesis_id above>",
          "test_description": "<what analysis would test this>",
          "expected_if_true": "<expected observation if hypothesis holds>",
          "expected_if_false": "<expected observation if hypothesis fails>",
          "citation_key": "<evidence key supporting this check>",
          "verdict": "<supports|contradicts|inconclusive>"
        }}
      ],
      "hidden_questions": [
        {{
          "question": "<question the pipeline didn't originally ask>",
          "rationale": "<why this matters>",
          "impact": "<high|medium|low>",
          "category": "<protocol|endpoint_definition|data_provenance|statistical_plan|population|regulatory|other>",
          "suggested_data_source": "<where to look for the answer>"
        }}
      ]
    }}

    VALID_CITATION_KEYS (use ONLY these):
    {valid_keys}

    KEY FINDINGS from InsightSynthesis:
    {key_findings}

    RISKS AND BIAS SIGNALS:
    {risks}

    UNKNOWNS SUMMARY:
    {unknowns_summary}

    UNRESOLVED HIGH-IMPACT UNKNOWNS:
    {unresolved}

    {judge_block}
    {metadata_block}

    DATASET_PREVIEW (JSON):
    {preview}

    DATASET_EVIDENCE (JSON):
    {evidence}
""")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        ).strip()
    return text


# ── LLM call ──────────────────────────────────────────────────────────────────

def generate_dynamic_hypotheses(
    *,
    provider: ProviderName,
    preview: dict,
    evidence: dict,
    final_insights: dict,
    unknowns: dict,
    valid_citation_keys: list[str],
    judge_output: dict | None = None,
    metadata: MetadataInput | None = None,
    tool_log: list[ToolCallRecord] | None = None,
) -> HypothesisGenerationOutput:
    """
    Call an LLM to generate hypotheses, falsification checks, and hidden
    questions.  The response is schema-validated against
    ``HypothesisGenerationOutput``.

    Raises on LLM / network / parse failure (caller should catch).
    """
    from qtrial_backend.core.router import get_client
    from qtrial_backend.core.types import LLMRequest

    key_findings = json.dumps(
        final_insights.get("key_findings", []), indent=2, ensure_ascii=False,
    )
    risks = json.dumps(
        final_insights.get("risks_and_bias_signals", []), indent=2, ensure_ascii=False,
    )
    unknowns_summary = unknowns.get("summary", "")
    unresolved = json.dumps(unknowns.get("unresolved_high_impact", []))

    judge_block = ""
    if judge_output:
        failed = judge_output.get("failed_claims", [])
        if failed:
            judge_block = (
                "JUDGE FEEDBACK (failed claims):\n"
                + json.dumps(failed, indent=2, default=str)
            )

    metadata_block = ""
    if metadata:
        metadata_block = (
            "USER METADATA:\n"
            + json.dumps(
                metadata.model_dump(exclude_none=True),
                indent=2,
                default=str,
            )
        )

    user = _HYPO_USER.format(
        valid_keys=json.dumps(valid_citation_keys, indent=2),
        key_findings=key_findings,
        risks=risks,
        unknowns_summary=unknowns_summary,
        unresolved=unresolved,
        judge_block=judge_block,
        metadata_block=metadata_block,
        preview=json.dumps(preview, indent=2, ensure_ascii=False),
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False, default=str),
    )

    client = get_client(provider)
    req = LLMRequest(system_prompt=_HYPO_SYSTEM, user_prompt=user, payload={})
    resp = client.generate(req)
    raw = _strip_fences(resp.text)

    data = json.loads(raw)
    return HypothesisGenerationOutput.model_validate(data)


# ── Deterministic integration ─────────────────────────────────────────────────

def integrate_dynamic_hypotheses(
    *,
    state: ReasoningState,
    llm_output: HypothesisGenerationOutput,
    metadata: MetadataInput | None,
    unresolved_high_impact: list[str],
) -> ReasoningState:
    """
    Merge LLM-generated hypotheses into an existing ``ReasoningState``.

    Steps (all deterministic — no LLM):
      1. Convert ``LLMHypothesis`` → ``CandidateHypothesis`` with citation
         validation.
      2. Create ``ClaimDraft`` per hypothesis, run deterministic validators.
      3. Validate falsification-check citation keys.
      4. Merge hypotheses, claims, hidden questions, falsification checks.
      5. Recompute confidence summary over all claims.
      6. Update stop condition and step log.
    """

    # ── 1. Convert hypotheses ────────────────────────────────────────────
    new_hypotheses: list[CandidateHypothesis] = []
    for h in llm_output.hypotheses:
        evidence_entries: list[EvidenceSupportEntry] = []
        bad_cites: list[str] = []

        for cite in h.evidence_citations:
            results = validate_claim_citations([cite], state.valid_citation_keys)
            if results and results[0].passed:
                evidence_entries.append(EvidenceSupportEntry(
                    citation_key=cite,
                    supports_claim=True,
                    explanation=h.rationale,
                ))
            else:
                bad_cites.append(cite)

        new_hypotheses.append(CandidateHypothesis(
            hypothesis_id=h.hypothesis_id,
            statement=h.statement,
            source_agent="InsightSynthesisAgent",
            confidence=h.confidence,
            evidence_support=evidence_entries,
            contradictions=bad_cites,
            status="candidate" if not bad_cites else "deferred",
        ))

    # ── 2. Claims from hypotheses → validate ─────────────────────────────
    new_claims: list[ClaimDraft] = []
    for h in llm_output.hypotheses:
        new_claims.append(ClaimDraft(
            claim_id=f"hyp_{h.hypothesis_id}",
            text=h.statement,
            hypothesis_ids=[h.hypothesis_id],
            citations=h.evidence_citations,
            confidence=h.confidence,
        ))

    validated_new = validate_all_claims(
        claims=new_claims,
        valid_citation_keys=state.valid_citation_keys,
        metadata=metadata,
        unresolved_high_impact=unresolved_high_impact,
    )

    # ── 3. Validate falsification-check citations ─────────────────────────
    validated_checks: list[FalsificationCheck] = []
    for fc in llm_output.falsification_checks:
        results = validate_claim_citations(
            [fc.citation_key], state.valid_citation_keys,
        )
        if results and results[0].passed:
            validated_checks.append(fc)
        else:
            validated_checks.append(
                fc.model_copy(update={"verdict": "inconclusive"})
            )

    # ── 4. Merge into state ───────────────────────────────────────────────
    all_claims = state.claims + validated_new
    all_hypotheses = state.hypotheses + new_hypotheses

    state = state.model_copy(update={
        "hypotheses": all_hypotheses,
        "claims": all_claims,
        "hidden_questions": list(llm_output.hidden_questions),
        "falsification_checks": validated_checks,
    })

    # ── 5. Step log — hypothesis generation ───────────────────────────────
    state = append_reasoning_step(
        state,
        step_type="hypothesis_gen",
        inputs=["final_insights", "unknowns", "evidence"],
        outputs=[h.hypothesis_id for h in new_hypotheses],
        notes=(
            f"Dynamic: {len(new_hypotheses)} hypotheses, "
            f"{len(validated_checks)} falsification checks, "
            f"{len(llm_output.hidden_questions)} hidden questions."
        ),
    )

    # ── 6. Step log — validation of hypothesis claims ─────────────────────
    n_v = sum(1 for c in validated_new if c.validation_status == "valid")
    n_f = sum(1 for c in validated_new if c.validation_status == "flagged")
    n_r = sum(1 for c in validated_new if c.validation_status == "rejected")

    state = append_reasoning_step(
        state,
        step_type="validation",
        inputs=[c.claim_id for c in validated_new],
        outputs=[f"valid={n_v}", f"flagged={n_f}", f"rejected={n_r}"],
        notes=f"Hypothesis claims: {n_v} valid, {n_f} flagged, {n_r} rejected.",
    )

    # ── 7. Recompute confidence summary ───────────────────────────────────
    confidence = compute_confidence_summary(
        claims=all_claims,
        unresolved_high_impact=unresolved_high_impact,
        metadata=metadata,
    )
    state = state.model_copy(update={"confidence_summary": confidence})

    # ── 8. Update stop condition ──────────────────────────────────────────
    state = state.model_copy(update={
        "stop_condition": StopCondition(
            met=True,
            reason=(
                f"Dynamic hypothesis generation complete: "
                f"{len(all_claims)} total claims, "
                f"{len(all_hypotheses)} hypotheses, "
                f"confidence={confidence.overall}."
            ),
        ),
    })

    state = append_reasoning_step(
        state,
        step_type="stop_evaluation",
        inputs=["all_claims", "hypotheses"],
        outputs=["stop_condition.met=True"],
        notes="Reasoning complete (deterministic + dynamic).",
    )

    return state
