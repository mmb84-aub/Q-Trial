"""
Task 4A — Reasoning Core foundations.

Provides:
  - Citation-key builder (preview + evidence + tool_log → valid key set)
  - ValidationResult dataclass
  - Deterministic (non-LLM) claim validators:
      · validate_claim_citations  — every citation must be resolvable
      · check_endpoint_lockdown   — no definitive endpoint interpretation
                                    without status_mapping + time_unit
      · check_treatment_effect_guard — no treatment-effect claim without
                                       treatment_arms metadata
      · check_high_confidence_guard  — no high-confidence claim while
                                       high-impact unknowns remain
  - validate_claim          — run all validators against one ClaimDraft
  - validate_all_claims     — run across an entire list
  - compute_confidence_summary — aggregate ClaimDraft results into a
                                 ConfidenceSummary (deterministic, no LLM)

Nothing in this module calls an LLM.  All logic is pure Python.
All schemas live in agentic.schemas to avoid circular imports.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from qtrial_backend.agentic.schemas import (
    ClaimInternalVerification,
    ClaimDraft,
    ConfidenceSummary,
    InternalEvidenceSnippet,
    InternalVerificationSummary,
    MetadataInput,
    ReasoningState,
    ReasoningStepLog,
    ReasoningStepType,
    StopCondition,
    ToolCallRecord,
)
from qtrial_backend.rag import BM25Retriever, EvidenceStore, RetrievalQuery
from qtrial_backend.rag.bm25_retriever import tokenize_query
from qtrial_backend.rag.ingestion import ingest_text, ingest_tool_result


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a single deterministic validation rule."""

    passed: bool
    rule: str
    details: str
    severity: Literal["fatal", "warning"]


# ── Citation key builder ──────────────────────────────────────────────────────

def build_valid_citation_keys(
    preview: dict,
    evidence: dict,
    tool_log: list[ToolCallRecord] | None = None,
) -> list[str]:
    """
    Return the complete set of resolvable citation strings for one run.

    Naming rules
    ------------
    preview.*
        One entry per top-level key in the preview dict.

    evidence.*
        One section-level entry per top-level evidence key, plus one
        dot-path entry per sub-key (one level deep) when the value is a
        dict (e.g. ``evidence.missingness_pct.bili``).

    top_correlations[i]
        ``top_correlations`` is a *list* in evidence, so only indexed
        aliases are valid — never dotted column paths.

    tool_log[i]
        Taken from ``ToolCallRecord.citation_alias`` when non-empty.
    """
    keys: set[str] = set()

    # preview.* top-level keys
    for k in preview:
        keys.add(f"preview.{k}")

    # evidence.* — section-level + one-deep sub-keys
    for k, v in evidence.items():
        if k.startswith("__"):
            # Skip internal orchestrator markers (__{name}__)
            continue
        section = f"evidence.{k}"
        keys.add(section)
        if isinstance(v, dict):
            for sub in v:
                keys.add(f"{section}.{sub}")
        elif isinstance(v, list):
            # Lists use indexed aliases, e.g. top_correlations[0]
            for i in range(len(v)):
                keys.add(f"{k}[{i}]")

    # tool_log[i] aliases
    if tool_log:
        for rec in tool_log:
            if rec.citation_alias:
                keys.add(rec.citation_alias)

    return sorted(keys)


# ── Signal word sets (for claim-text pattern matching) ───────────────────────

_ENDPOINT_SIGNALS: frozenset[str] = frozenset(
    [
        "event", "endpoint", "outcome", "status", "death", "transplant",
        "censored", "survival", "hazard", "event-free", "time-to-event",
        "prognostic", "time to event",
    ]
)

_TREATMENT_SIGNALS: frozenset[str] = frozenset(
    [
        "treatment", "arm", "drug", "placebo", "d-penicillamine",
        "intervention", "control group", "assigned", "allocation",
        "treatment effect", "effect of treatment", "randomis", "randomiz",
        "dosed",
    ]
)

_DEFINITIVE_PHRASING: frozenset[str] = frozenset(
    [
        "proves", "confirms", "demonstrates", "establishes",
        "is a predictor", "is the primary", "is a prognostic",
        "significantly predicts", "directly causes",
    ]
)

_CAUSAL_TREATMENT_PHRASING: frozenset[str] = frozenset(
    [
        "caused by treatment", "resulted in", "led to the outcome",
        "produced a significant", "improved survival", "worsened prognosis",
        "reduced mortality", "increased event rate", "treatment effect",
        "effect of treatment",
    ]
)

# Regex for bare tool_log alias: tool_log[<integer>]
_TOOL_LOG_ALIAS_RE = re.compile(r"^tool_log\[(\d+)\]")


# ── Individual validators ─────────────────────────────────────────────────────

def validate_claim_citations(
    citations: list[str],
    valid_keys: list[str],
) -> list[ValidationResult]:
    """
    Check that every citation string in ``citations`` resolves to a known key.

    Resolution order
    ----------------
    1. Exact match against ``valid_keys``.
    2. ``tool_log[N]`` pattern — alias must appear in ``valid_keys``.
    3. Indexed-prefix form: ``"top_correlations[0]: (albumin, status)=0.895"``
       — the part before the first ``:`` must be in ``valid_keys``.
    4. Sub-path: ``"evidence.missingness_pct.bili"`` when the parent section
       ``"evidence.missingness_pct"`` is in ``valid_keys``.
    """
    valid_set = set(valid_keys)
    results: list[ValidationResult] = []

    for raw_cite in citations:
        cite = raw_cite.strip()
        if not cite:
            continue

        # 1. Exact match
        if cite in valid_set:
            results.append(ValidationResult(
                passed=True,
                rule="citation_exact_match",
                details=f"'{cite}' resolved exactly.",
                severity="warning",
            ))
            continue

        # 2. tool_log[N] alias
        m = _TOOL_LOG_ALIAS_RE.match(cite)
        if m:
            alias = m.group(0)            # e.g. "tool_log[2]"
            if alias in valid_set:
                results.append(ValidationResult(
                    passed=True,
                    rule="citation_tool_log_alias",
                    details=f"'{alias}' is a valid tool_log citation alias.",
                    severity="warning",
                ))
                continue
            # alias not in valid_set → fall through to failure

        # 3. Indexed-prefix form: "top_correlations[0]: ..."
        colon_pos = cite.find(":")
        if colon_pos > 0:
            prefix = cite[:colon_pos].strip()
            if prefix in valid_set:
                results.append(ValidationResult(
                    passed=True,
                    rule="citation_indexed_prefix",
                    details=f"Indexed prefix '{prefix}' resolved in valid keys.",
                    severity="warning",
                ))
                continue

        # 4. Sub-path: check whether any valid key is a prefix of this cite
        resolved_via_subpath = False
        for candidate in valid_set:
            if cite.startswith(candidate + ".") or cite == candidate:
                results.append(ValidationResult(
                    passed=True,
                    rule="citation_subpath",
                    details=(
                        f"'{cite}' is a sub-path of valid key '{candidate}'."
                    ),
                    severity="warning",
                ))
                resolved_via_subpath = True
                break

        if not resolved_via_subpath:
            results.append(ValidationResult(
                passed=False,
                rule="citation_unresolvable",
                details=(
                    f"Citation '{cite}' does not match any valid key.  "
                    "Valid namespaces: evidence.*, preview.*, tool_log[i]."
                ),
                severity="fatal",
            ))

    return results


def check_endpoint_lockdown(
    claim_text: str,
    metadata: MetadataInput | None,
    unresolved_high_impact: list[str],
) -> list[ValidationResult]:
    """
    Guard: a claim that touches endpoint/outcome signals must NOT make
    definitive interpretations unless ALL of the following are confirmed:

    * ``metadata.status_mapping`` is set (event codes are known)
    * ``metadata.time_unit`` is set (time scale is known)
    * No high-impact endpoint/outcome questions remain unresolved
    """
    claim_lc = claim_text.lower()
    signals_found = [s for s in _ENDPOINT_SIGNALS if s in claim_lc]
    if not signals_found:
        return []

    issues: list[str] = []

    if metadata is None or metadata.status_mapping is None:
        issues.append(
            "status_mapping not confirmed — event codes are ambiguous"
        )
    if metadata is None or metadata.time_unit is None:
        issues.append(
            "time_unit not confirmed — time scale is ambiguous"
        )

    endpoint_unknowns = [
        q for q in unresolved_high_impact
        if any(s in q.lower() for s in _ENDPOINT_SIGNALS)
    ]
    if endpoint_unknowns:
        condensed = "; ".join(endpoint_unknowns[:2])
        issues.append(f"Unresolved endpoint unknowns: {condensed}")

    if not issues:
        return [
            ValidationResult(
                passed=True,
                rule="endpoint_lockdown",
                details=(
                    "Endpoint metadata confirmed and no unresolved endpoint "
                    "unknowns; definitive interpretation is permissible."
                ),
                severity="warning",
            )
        ]

    is_definitive = any(p in claim_lc for p in _DEFINITIVE_PHRASING)
    severity: Literal["fatal", "warning"] = "fatal" if is_definitive else "warning"

    return [
        ValidationResult(
            passed=False,
            rule="endpoint_lockdown",
            details=(
                f"Claim references endpoint signals "
                f"({', '.join(signals_found[:3])}) but: {'; '.join(issues)}."
            ),
            severity=severity,
        )
    ]


def check_treatment_effect_guard(
    claim_text: str,
    metadata: MetadataInput | None,
) -> list[ValidationResult]:
    """
    Guard: a claim referencing treatment/arm signals must NOT assert
    treatment effects unless ``metadata.treatment_arms`` is confirmed.
    """
    claim_lc = claim_text.lower()
    signals_found = [s for s in _TREATMENT_SIGNALS if s in claim_lc]
    if not signals_found:
        return []

    if metadata is not None and metadata.treatment_arms is not None:
        return [
            ValidationResult(
                passed=True,
                rule="treatment_effect_guard",
                details=(
                    "treatment_arms metadata confirmed; treatment-related "
                    "claim is permissible."
                ),
                severity="warning",
            )
        ]

    is_causal = any(p in claim_lc for p in _CAUSAL_TREATMENT_PHRASING)
    severity: Literal["fatal", "warning"] = "fatal" if is_causal else "warning"

    return [
        ValidationResult(
            passed=False,
            rule="treatment_effect_guard",
            details=(
                f"Claim references treatment signals "
                f"({', '.join(signals_found[:3])}) but treatment_arms "
                "metadata is not confirmed."
            ),
            severity=severity,
        )
    ]


def check_high_confidence_guard(
    confidence: Literal["high", "medium", "low"],
    unresolved_high_impact: list[str],
) -> list[ValidationResult]:
    """
    Guard: a claim may only carry ``confidence='high'`` when there are
    zero unresolved high-impact unknowns.
    """
    if confidence != "high":
        return []

    if not unresolved_high_impact:
        return [
            ValidationResult(
                passed=True,
                rule="high_confidence_guard",
                details=(
                    "No unresolved high-impact unknowns; "
                    "high confidence is permissible."
                ),
                severity="warning",
            )
        ]

    return [
        ValidationResult(
            passed=False,
            rule="high_confidence_guard",
            details=(
                f"Claim is marked 'high' confidence but "
                f"{len(unresolved_high_impact)} high-impact unknown(s) "
                "remain unresolved."
            ),
            severity="fatal",
        )
    ]


# ── Composite validators ──────────────────────────────────────────────────────

def validate_claim(
    claim: ClaimDraft,
    valid_citation_keys: list[str],
    metadata: MetadataInput | None,
    unresolved_high_impact: list[str],
) -> ClaimDraft:
    """
    Run all deterministic validators against a single ``ClaimDraft``.

    Returns a **new** ClaimDraft (original unchanged) with updated
    ``validation_status`` and ``flag_reasons``.

    Status derivation
    -----------------
    - Any ``fatal`` failure  → ``"rejected"``
    - Only ``warning`` failures → ``"flagged"``
    - All checks pass       → ``"valid"``
    """
    all_results: list[ValidationResult] = []

    all_results.extend(
        validate_claim_citations(claim.citations, valid_citation_keys)
    )
    all_results.extend(
        check_endpoint_lockdown(claim.text, metadata, unresolved_high_impact)
    )
    all_results.extend(
        check_treatment_effect_guard(claim.text, metadata)
    )
    all_results.extend(
        check_high_confidence_guard(claim.confidence, unresolved_high_impact)
    )

    fatal_failures = [r for r in all_results if not r.passed and r.severity == "fatal"]
    warn_failures  = [r for r in all_results if not r.passed and r.severity == "warning"]

    if fatal_failures:
        new_status: Literal["pending", "valid", "flagged", "rejected"] = "rejected"
    elif warn_failures:
        new_status = "flagged"
    else:
        new_status = "valid"

    flag_reasons = [r.details for r in all_results if not r.passed]

    return claim.model_copy(
        update={
            "validation_status": new_status,
            "flag_reasons": flag_reasons,
        }
    )


def validate_all_claims(
    claims: list[ClaimDraft],
    valid_citation_keys: list[str],
    metadata: MetadataInput | None,
    unresolved_high_impact: list[str],
) -> list[ClaimDraft]:
    """Apply ``validate_claim`` to every claim in the list."""
    return [
        validate_claim(c, valid_citation_keys, metadata, unresolved_high_impact)
        for c in claims
    ]


# ── Confidence summary ────────────────────────────────────────────────────────

def compute_confidence_summary(
    claims: list[ClaimDraft],
    unresolved_high_impact: list[str],
    metadata: MetadataInput | None,
) -> ConfidenceSummary:
    """
    Compute an overall confidence level from validated claims and context.

    Deterministic — no LLM involved.

    Downgrade rules (applied in order, most severe first)
    -----------------------------------------------------
    1. No claims evaluated → ``"inconclusive"``
    2. Any unresolved high-impact unknown, or >30 % of claims rejected
       → ``"low"``
    3. >40 % of claims flagged, or fewer than 2 valid claims → ``"medium"``
    4. Otherwise, if at least half of claims are valid → ``"high"``
    5. Default fallback → ``"medium"``
    """
    n_total    = len(claims)
    n_valid    = sum(1 for c in claims if c.validation_status == "valid")
    n_flagged  = sum(1 for c in claims if c.validation_status == "flagged")
    n_rejected = sum(1 for c in claims if c.validation_status == "rejected")

    limiting: list[str] = []

    if unresolved_high_impact:
        limiting.append(
            f"{len(unresolved_high_impact)} high-impact unknown(s) unresolved."
        )
    if metadata is None or metadata.status_mapping is None:
        limiting.append("Event/status column coding not confirmed.")
    if metadata is None or metadata.time_unit is None:
        limiting.append("Time unit not confirmed.")

    if n_total == 0:
        return ConfidenceSummary(
            overall="inconclusive",
            num_supported_claims=0,
            num_flagged_claims=0,
            num_rejected_claims=0,
            limiting_factors=["No claims evaluated."],
        )

    reject_rate = n_rejected / n_total
    flag_rate   = n_flagged  / n_total

    if limiting or reject_rate > 0.30:
        overall: Literal["high", "medium", "low", "inconclusive"] = "low"
    elif flag_rate > 0.40 or n_valid < 2:
        overall = "medium"
    elif not limiting and n_valid >= max(1, n_total // 2):
        overall = "high"
    else:
        overall = "medium"

    return ConfidenceSummary(
        overall=overall,
        num_supported_claims=n_valid,
        num_flagged_claims=n_flagged,
        num_rejected_claims=n_rejected,
        limiting_factors=limiting,
    )


# ── ReasoningState factory ────────────────────────────────────────────────────

def init_reasoning_state(
    run_id: str,
    preview: dict,
    evidence: dict,
    tool_log: list[ToolCallRecord] | None = None,
) -> ReasoningState:
    """
    Create a fresh ``ReasoningState`` for a new run.

    Immediately computes ``valid_citation_keys`` so that all downstream
    validators have a consistent key set without re-deriving it.
    """
    valid_keys = build_valid_citation_keys(preview, evidence, tool_log)
    return ReasoningState(
        run_id=run_id,
        valid_citation_keys=valid_keys,
        stop_condition=StopCondition(met=False, reason="not started"),
    )


def append_reasoning_step(
    state: ReasoningState,
    step_type: ReasoningStepType,
    inputs: list[str],
    outputs: list[str],
    notes: str = "",
) -> ReasoningState:
    """
    Append a new step to the reasoning trace.

    Returns a **new** ReasoningState (original is not mutated).
    """
    new_step = ReasoningStepLog(
        step_index=len(state.step_log),
        step_type=step_type,
        inputs=inputs,
        outputs=outputs,
        notes=notes,
    )
    return state.model_copy(
        update={"step_log": state.step_log + [new_step]}
    )


# ── Task 4B — Deterministic Reasoning Engine Executor ─────────────────────────

def _extract_claims_from_insights(
    final_insights: dict,
    unresolved_high_impact: list[str],
) -> list[ClaimDraft]:
    """
    Build ``ClaimDraft`` objects from SynthesisInsights dicts.

    Sources (in order):
      1. ``key_findings`` — each string becomes a claim.
      2. ``risks_and_bias_signals`` — each becomes a low-confidence claim.
      3. ``recommended_next_analyses`` — ranked analyses with evidence
         citations (optional, only when ``evidence_citation`` is present).

    Citation heuristic: if the finding text contains a token that looks
    like an evidence path (``evidence.*``, ``preview.*``, ``tool_log[*]``)
    we extract it; otherwise citations is left empty (validation will not
    penalise a claim for *having no* citations — only for *invalid* ones).
    """
    claims: list[ClaimDraft] = []
    _idx = 0

    # Determine default confidence based on unresolved unknowns
    default_conf: Literal["high", "medium", "low"] = (
        "low" if unresolved_high_impact else "medium"
    )

    # 1. key_findings
    for finding in final_insights.get("key_findings", []):
        _idx += 1
        cites = _extract_inline_citations(finding)
        claims.append(ClaimDraft(
            claim_id=f"c{_idx}",
            text=finding,
            citations=cites,
            confidence=default_conf,
        ))

    # 2. risks_and_bias_signals (always low confidence)
    for signal in final_insights.get("risks_and_bias_signals", []):
        _idx += 1
        cites = _extract_inline_citations(signal)
        claims.append(ClaimDraft(
            claim_id=f"c{_idx}",
            text=signal,
            citations=cites,
            confidence="low",
        ))

    # 3. recommended_next_analyses (optional)
    for analysis in final_insights.get("recommended_next_analyses", []):
        if not isinstance(analysis, dict):
            continue
        cite_str = analysis.get("evidence_citation", "")
        if not cite_str:
            continue
        _idx += 1
        cites = _extract_inline_citations(cite_str)
        claims.append(ClaimDraft(
            claim_id=f"c{_idx}",
            text=analysis.get("analysis", cite_str),
            citations=cites,
            confidence=default_conf,
        ))

    return claims


_CITE_TOKEN_RE = re.compile(
    r"(evidence\.[\w.]+|preview\.[\w.]+|tool_log\[\d+\])"
)


def _extract_inline_citations(text: str) -> list[str]:
    """Pull citation-like tokens from free text."""
    return list(dict.fromkeys(_CITE_TOKEN_RE.findall(text)))


def _build_runtime_retrieval_index(
    analysis_report: str | None,
    tool_log: list[ToolCallRecord] | None,
) -> tuple[EvidenceStore, BM25Retriever]:
    """Build a temporary BM25 index from existing runtime artifacts.

    Sources indexed:
    - analysis_report markdown text (when present)
    - successful tool_log results as tool_result documents
    """
    store = EvidenceStore()
    retriever = BM25Retriever()

    if analysis_report and analysis_report.strip():
        report_doc = ingest_text(
            text=analysis_report,
            name="analysis_report",
            source_type="user_input",
            metadata={"source": "analysis_report"},
        )
        store.add_document(report_doc)

    if tool_log:
        for rec in tool_log:
            if rec.error:
                continue

            if rec.result is None:
                continue

            if isinstance(rec.result, str):
                result_text = rec.result
            else:
                try:
                    result_text = json.dumps(rec.result, default=str, ensure_ascii=False)
                except Exception:
                    result_text = str(rec.result)

            if not result_text.strip():
                continue

            try:
                doc = ingest_tool_result(
                    tool_name=rec.tool_name,
                    arguments=rec.args,
                    result_text=result_text,
                    metadata={
                        "citation_alias": rec.citation_alias,
                        "origin": "tool_log",
                    },
                )
                store.add_document(doc)
            except Exception:
                # Never block reasoning on index build noise.
                continue

    retriever.index(store.all_chunks())
    return store, retriever


def _support_status_from_hits(
    claim_text: str,
    snippets: list[InternalEvidenceSnippet],
) -> tuple[Literal["supported", "weakly_supported", "unsupported", "unclear"], str]:
    """Assign conservative deterministic support labels from retrieval hits."""
    if not snippets:
        return "unclear", "No internal evidence snippets were retrieved for this claim."

    query_terms = set(tokenize_query(claim_text))
    best = snippets[0]
    best_overlap = 0.0
    if query_terms:
        best_overlap = len(set(best.matched_terms)) / max(len(query_terms), 1)

    # Conservative thresholds to avoid overstating support.
    if best.score >= 2.0 and best_overlap >= 0.30:
        return (
            "supported",
            (
                f"High-scoring internal evidence (score={best.score:.3f}) with "
                f"substantial token overlap ({best_overlap:.2f})."
            ),
        )

    if best.score >= 1.0 and best_overlap >= 0.15:
        return (
            "weakly_supported",
            (
                f"Moderate retrieval match (score={best.score:.3f}, overlap={best_overlap:.2f}); "
                "claim aligns with evidence but support is limited."
            ),
        )

    if best.score < 0.5 and best_overlap < 0.08:
        return (
            "unsupported",
            (
                f"Best retrieved evidence is weak (score={best.score:.3f}, overlap={best_overlap:.2f})."
            ),
        )

    return "unclear", "Retrieved evidence is mixed or insufficient for a confident support decision."


def verify_claims_with_internal_evidence(
    claims: list[ClaimDraft],
    analysis_report: str | None,
    tool_log: list[ToolCallRecord] | None,
    *,
    top_k: int = 3,
) -> tuple[list[ClaimInternalVerification], InternalVerificationSummary]:
    """Run post-reasoning claim verification against a BM25 runtime corpus."""
    _, retriever = _build_runtime_retrieval_index(analysis_report, tool_log)

    verifications: list[ClaimInternalVerification] = []

    for claim in claims:
        hits = retriever.retrieve(
            RetrievalQuery(text=claim.text, top_k=top_k, min_score=0.0)
        )

        query_terms = set(tokenize_query(claim.text))
        snippets: list[InternalEvidenceSnippet] = []
        for h in hits:
            chunk_terms = set(tokenize_query(h.chunk.text))
            matched_terms = sorted(list(query_terms.intersection(chunk_terms)))[:20]
            snippet_text = h.chunk.text[:280] + ("..." if len(h.chunk.text) > 280 else "")
            snippets.append(
                InternalEvidenceSnippet(
                    rank=h.rank,
                    score=round(float(h.score), 6),
                    snippet=snippet_text,
                    source_type=str(h.provenance.source_type),
                    source_name=str(h.provenance.source_name),
                    chunk_id=str(h.provenance.chunk_id),
                    matched_terms=matched_terms,
                )
            )

        support_status, rationale = _support_status_from_hits(claim.text, snippets)

        verifications.append(
            ClaimInternalVerification(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                validation_status=claim.validation_status,
                support_status=support_status,
                rationale=rationale,
                top_evidence=snippets,
            )
        )

    summary = InternalVerificationSummary(
        total_claims=len(verifications),
        supported=sum(1 for v in verifications if v.support_status == "supported"),
        weakly_supported=sum(1 for v in verifications if v.support_status == "weakly_supported"),
        unsupported=sum(1 for v in verifications if v.support_status == "unsupported"),
        unclear=sum(1 for v in verifications if v.support_status == "unclear"),
        index_chunks=retriever.index_size,
    )

    return verifications, summary


def run_reasoning_engine(
    *,
    run_id: str,
    preview: dict,
    evidence: dict,
    final_insights: dict,
    unknowns: dict,
    metadata: MetadataInput | None = None,
    analysis_report: str | None = None,
    tool_log: list[ToolCallRecord] | None = None,
) -> ReasoningState:
    """
    Task 4B deterministic reasoning-engine executor.

    Runs entirely without LLM calls.  Consumes the outputs already
    produced by the agentic pipeline (insights, unknowns, evidence,
    preview) and applies rule-based validation logic from Task 4A.

    Steps executed (each recorded in ``step_log``):
        0. initialization — build valid citation keys
        1. claim_extraction — derive ClaimDraft objects from insights
        2. validation — run all deterministic validators
        3. confidence_summary — aggregate confidence
        4. stop_evaluation — set deterministic stop condition

    Returns
    -------
    ReasoningState
        Fully populated state ready for ``AnalysisReport.reasoning_state``.
    """

    # ── 0. Initialization ────────────────────────────────────────────────
    state = init_reasoning_state(
        run_id=run_id,
        preview=preview,
        evidence=evidence,
        tool_log=tool_log,
    )

    init_inputs = ["preview", "evidence"]
    if tool_log:
        init_inputs.append(f"tool_log({len(tool_log)} calls)")
    if analysis_report:
        init_inputs.append("analysis_report")

    state = append_reasoning_step(
        state,
        step_type="evidence_scan",
        inputs=init_inputs,
        outputs=[f"{len(state.valid_citation_keys)} valid citation keys"],
        notes="Reasoning engine initialised; citation key set built.",
    )

    # ── 1. Claim extraction ──────────────────────────────────────────────
    unresolved = unknowns.get("unresolved_high_impact", [])

    claims = _extract_claims_from_insights(final_insights, unresolved)

    state = state.model_copy(update={"claims": claims})
    state = append_reasoning_step(
        state,
        step_type="claim_draft",
        inputs=["final_insights.key_findings",
                "final_insights.risks_and_bias_signals"],
        outputs=[c.claim_id for c in claims],
        notes=f"Extracted {len(claims)} candidate claim(s) from insights.",
    )

    # ── 2. Validation ────────────────────────────────────────────────────
    validated_claims = validate_all_claims(
        claims=claims,
        valid_citation_keys=state.valid_citation_keys,
        metadata=metadata,
        unresolved_high_impact=unresolved,
    )
    state = state.model_copy(update={"claims": validated_claims})

    n_valid = sum(1 for c in validated_claims if c.validation_status == "valid")
    n_flagged = sum(1 for c in validated_claims if c.validation_status == "flagged")
    n_rejected = sum(1 for c in validated_claims if c.validation_status == "rejected")

    state = append_reasoning_step(
        state,
        step_type="validation",
        inputs=[c.claim_id for c in validated_claims],
        outputs=[
            f"valid={n_valid}",
            f"flagged={n_flagged}",
            f"rejected={n_rejected}",
        ],
        notes=(
            f"Deterministic validation complete: {n_valid} valid, "
            f"{n_flagged} flagged, {n_rejected} rejected."
        ),
    )

    # ── 2b. Internal BM25 post-reasoning verification ─────────────────
    internal_verification, iv_summary = verify_claims_with_internal_evidence(
        claims=validated_claims,
        analysis_report=analysis_report,
        tool_log=tool_log,
    )
    state = state.model_copy(
        update={
            "internal_verification": internal_verification,
            "internal_verification_summary": iv_summary,
        }
    )

    state = append_reasoning_step(
        state,
        step_type="validation",
        inputs=[c.claim_id for c in validated_claims],
        outputs=[
            f"internal_supported={iv_summary.supported}",
            f"internal_weak={iv_summary.weakly_supported}",
            f"internal_unsupported={iv_summary.unsupported}",
            f"internal_unclear={iv_summary.unclear}",
        ],
        notes=(
            "Post-reasoning internal BM25 verification complete: "
            f"{iv_summary.supported} supported, "
            f"{iv_summary.weakly_supported} weakly supported, "
            f"{iv_summary.unsupported} unsupported, "
            f"{iv_summary.unclear} unclear."
        ),
    )

    # ── 3. Confidence summary ────────────────────────────────────────────
    confidence = compute_confidence_summary(
        claims=validated_claims,
        unresolved_high_impact=unresolved,
        metadata=metadata,
    )

    # Internal verification can cap confidence conservatively.
    limiting = list(confidence.limiting_factors)
    if iv_summary.unsupported > 0:
        limiting.append(
            f"{iv_summary.unsupported} claim(s) unsupported by internal BM25 verification."
        )
    if iv_summary.total_claims > 0 and (
        (iv_summary.unsupported / iv_summary.total_claims) >= 0.4
    ):
        if confidence.overall in {"high", "medium"}:
            confidence = confidence.model_copy(update={"overall": "low"})
    elif iv_summary.unsupported > 0 and confidence.overall == "high":
        confidence = confidence.model_copy(update={"overall": "medium"})

    confidence = confidence.model_copy(update={"limiting_factors": limiting})
    state = state.model_copy(update={"confidence_summary": confidence})

    state = append_reasoning_step(
        state,
        step_type="stop_evaluation",
        inputs=["validated_claims", "unresolved_high_impact"],
        outputs=[f"overall_confidence={confidence.overall}"],
        notes=(
            f"Confidence summary: {confidence.overall}. "
            f"Limiting factors: {'; '.join(confidence.limiting_factors) or 'none'}."
        ),
    )

    # ── 4. Stop condition ────────────────────────────────────────────────
    stop_reason_parts: list[str] = [
        f"{len(validated_claims)} claim(s) validated",
        f"confidence={confidence.overall}",
    ]
    if n_rejected:
        stop_reason_parts.append(f"{n_rejected} rejected")
    if unresolved:
        stop_reason_parts.append(
            f"{len(unresolved)} high-impact unknown(s) unresolved"
        )

    state = state.model_copy(
        update={
            "stop_condition": StopCondition(
                met=True,
                reason="Deterministic reasoning complete: "
                       + "; ".join(stop_reason_parts) + ".",
            ),
        }
    )

    state = append_reasoning_step(
        state,
        step_type="stop_evaluation",
        inputs=["confidence_summary"],
        outputs=["stop_condition.met=True"],
        notes="Reasoning loop terminated (deterministic single pass).",
    )

    return state
