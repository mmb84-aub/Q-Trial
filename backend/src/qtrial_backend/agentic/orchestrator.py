"""
Q-Trial Orchestrator — matches the agreed pipeline design exactly.

Pipeline (per design doc):
  1. Clinical context input          — study_context string (from caller)
  2. Dataset upload + blinding       — sanitised df (from caller)
  3. Data Profiler + Data Quality    — pure Python (build_dataset_evidence)
  4. Statistical Analysis Agent      — LLM agent loop (run_statistical_agent_loop, called in api.py)
     + Internal Critic               — post-loop conditional checks (in runner.py)
  5. Literature Query Translation    — LLM mini-call (translate_findings_to_cst)
  6. Literature Validator            — API calls (LiteratureValidatorPipeline)
  7. Synthesis + Self-Scoring        — single LLM call (run_synthesis_call)
  8. Report Generator                — React + Python (separate)

Removed (per design doc — do NOT add back):
  - DataQualityAgent      (was LLM call → pure Python profiler)
  - ClinicalSemanticsAgent (was LLM call → statistical agent handles implicitly)
  - UnknownsAgent          (was LLM call → collapsed into synthesis)
  - InsightSynthesisAgent  (was LLM call → collapsed into synthesis)
  - JudgeAgent / Critic    (was LLM call → post-loop conditional checks)
  - call_planner           (not in design)
  - run_reasoning_engine   (not in design)
  - run_literature_rag     (hypothesis-driven RAG, not in design)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rich.console import Console

from qtrial_backend.agentic.cst_translator import translate_findings_to_cst
from qtrial_backend.agentic.finding_categories import (
    classify_claim_type,
    classify_finding_category,
    is_analytical_category,
    is_malformed_finding_fragment,
    is_methodology_instruction_text,
    is_raw_stat_artifact_finding,
    is_raw_statistical_artifact_text,
    is_user_facing_clinical_finding_eligible,
    is_user_facing_nonfinding_artifact,
    neutral_status_for_category,
)
from qtrial_backend.agentic.finding_comparison_normalizer import normalize_comparison_claims
from qtrial_backend.agentic.finding_verbalizer import verbalize_statistical_findings
from qtrial_backend.agentic.literature_validator import LiteratureValidatorPipeline
from qtrial_backend.agentic.report_comparison import build_comparison_report, normalize_qtrial_findings
from qtrial_backend.agentic.report_curation import curate_user_facing_report_sections
from qtrial_backend.agentic.reproducibility import ReproducibilityLogBuilder
from qtrial_backend.agentic.schemas import (
    ComparisonReport,
    ControlVariable,
    ExcludedColumn,
    FinalReportSchema,
    GroundedFinding,
    GroundedFindingsSchema,
    GuardrailReport,
    HighMissingnessColumn,
    ListwiseDeletionColumn,
    LiteratureRAGReport,
    MetadataInput,
    MissingnessDisclosure,
    ResearchQuestion,
    SynthesisOutput,
    SynthesisQualityScore,
    ToolCallRecord,
    UnknownsOutput,
)
from qtrial_backend.agentic.statistical_verification import build_statistical_verification_report
from qtrial_backend.agentic.validation import build_retry_prompt, validate_synthesis_output
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName
from qtrial_backend.dataset.evidence import build_dataset_evidence, format_citations
from qtrial_backend.dataset.guardrails import format_guardrail_citations, run_guardrails
from qtrial_backend.dataset.preview import build_dataset_preview
from qtrial_backend.dataset.treatment_detector import detect_treatment_columns

console = Console()

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "agentic_run.json"


# ── Tool-log coerce / compact helpers ────────────────────────────────────────

def _coerce_tool_log(
    raw_tool_log: list[dict[str, Any]] | None,
) -> list[ToolCallRecord] | None:
    if not raw_tool_log:
        return None
    records: list[ToolCallRecord] = []
    for i, entry in enumerate(raw_tool_log):
        records.append(
            ToolCallRecord(
                tool_name=entry.get("tool", entry.get("tool_name", "unknown")),
                args=entry.get("args", {}),
                result=entry.get("result"),
                error=entry.get("error"),
                citation_alias=f"tool_log[{i}]",
            )
        )
    return records or None


def _compact_tool_log_for_persistence(
    records: list[ToolCallRecord] | None,
) -> list[dict[str, Any]] | None:
    if not records:
        return None
    MAX_RESULT_CHARS = 2_000
    out: list[dict[str, Any]] = []
    for rec in records:
        result_repr: Any = rec.result
        if isinstance(result_repr, str) and len(result_repr) > MAX_RESULT_CHARS:
            result_repr = result_repr[:MAX_RESULT_CHARS] + "\u2026(truncated)"
        elif isinstance(result_repr, (dict, list)):
            result_str = json.dumps(result_repr, default=str)
            if len(result_str) > MAX_RESULT_CHARS:
                result_repr = result_str[:MAX_RESULT_CHARS] + "\u2026(truncated)"
        entry: dict[str, Any] = {
            "citation_alias": rec.citation_alias,
            "tool_name": rec.tool_name,
            "args": rec.args,
            "result": result_repr,
        }
        if rec.error:
            entry["error"] = rec.error
        out.append(entry)
    return out


# ── Step 7: Synthesis LLM call ────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = textwrap.dedent("""\
    You are a senior clinical data analyst producing the final synthesis report
    for a completed clinical trial analysis.

    You receive:
    - A study context describing the trial
    - A statistical analysis report with findings, p-values, and test rationales
    - Literature grounding results per finding (Supported / Contradicted / Novel)

    Produce a single structured JSON output with ALL of the following:
    1. forward_recommendations: refined hypothesis for the next trial, endpoint
       improvements, recommended sample size with rationale, variables to control
    2. research_questions: 3+ specific questions linked to actual findings
    3. narrative_summary: plain-language prose overview (3-5 sentences, no p-values,
       no statistical jargon, suitable for a clinician)

    RULES:
    - Do not repeat raw statistical values (p=, r=, HR=) in the narrative
    - Ground every recommendation in a specific finding from the analysis report
    - Research questions must reference actual column names or patterns observed
    - Narrative must be readable by a clinician with no statistical training

    Respond with ONLY valid JSON. No markdown, no commentary.
""")

_SYNTHESIS_USER = textwrap.dedent("""\
    STUDY CONTEXT:
    {study_context}

    STATISTICAL ANALYSIS REPORT:
    {analysis_report}

    LITERATURE GROUNDING RESULTS:
    {grounding_summary}

    Required JSON schema:
    {{
      "future_trial_hypothesis": "<refined hypothesis for the next trial>",
      "endpoint_improvement_recommendations": ["<recommendation>", ...],
      "recommended_sample_size": "<size with rationale>",
      "variables_to_control": [
        {{"variable": "<name>", "reason": "<why>"}}
      ],
      "research_questions": [
        {{"question": "<question>", "source_finding": "<which finding prompted this>"}}
      ],
      "narrative_summary": "<plain-language prose, 3-5 sentences>"
    }}
""")


def run_synthesis_call(
    analysis_report: str,
    grounded_findings: list[GroundedFinding],
    study_context: str,
    provider: ProviderName,
) -> tuple[SynthesisOutput, str]:
    """
    Single synthesis LLM call (Step 7 of the design).

    Returns (SynthesisOutput, narrative_summary).
    """
    # Build a compact grounding summary for the prompt
    grounding_lines: list[str] = []
    for i, gf in enumerate(grounded_findings):
        status = gf.grounding_status
        cites = ", ".join(a.citation_alias or a.title[:40] for a in gf.citations[:2]) if gf.citations else "no citations"
        strength = gf.evidence_strength.plain_language if gf.evidence_strength else "unknown strength"
        grounding_lines.append(
            f"[{i+1}] {status} — {gf.finding_text[:120]}\n"
            f"     Evidence: {strength} | Citations: {cites}"
        )
    grounding_summary = "\n".join(grounding_lines) if grounding_lines else "(no grounded findings)"

    client = get_client(provider)
    user = _SYNTHESIS_USER.format(
        study_context=study_context,
        analysis_report=analysis_report[:6000],  # cap to avoid token overflow
        grounding_summary=grounding_summary,
    )
    req = LLMRequest(
        system_prompt=_SYNTHESIS_SYSTEM,
        user_prompt=user,
        payload={"temperature": 0},
    )

    try:
        resp = client.generate(req)
        raw = resp.text.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
    except Exception as _synth_exc:
        # Log the real error so it is visible in the backend terminal.
        console.print(
            f"[red]⚠ Synthesis LLM call FAILED (provider={provider}): "
            f"{type(_synth_exc).__name__}: {_synth_exc}[/red]"
        )
        return (
            SynthesisOutput(
                future_trial_hypothesis="Further investigation required.",
                recommended_sample_size="To be determined.",
            ),
            "Analysis complete. See findings for details.",
        )

    narrative = str(data.get("narrative_summary", ""))

    variables_to_control = [
        ControlVariable(variable=v.get("variable", ""), reason=v.get("reason", ""))
        for v in data.get("variables_to_control", [])
        if isinstance(v, dict)
    ]
    research_questions = [
        ResearchQuestion(question=q.get("question", ""), source_finding=q.get("source_finding", ""))
        for q in data.get("research_questions", [])
        if isinstance(q, dict)
    ]

    synthesis = SynthesisOutput(
        future_trial_hypothesis=str(data.get("future_trial_hypothesis", "")),
        endpoint_improvement_recommendations=[
            str(r) for r in data.get("endpoint_improvement_recommendations", [])
        ],
        recommended_sample_size=str(data.get("recommended_sample_size", "")),
        variables_to_control=variables_to_control,
        research_questions=research_questions,
        narrative_summary=narrative,
    )

    # ── Deterministic validation + single retry ───────────────────────────
    errors = validate_synthesis_output(synthesis, grounded_findings)
    if errors:
        retry_prompt = build_retry_prompt(errors, user)
        retry_req = LLMRequest(
            system_prompt=_SYNTHESIS_SYSTEM,
            user_prompt=retry_prompt,
            payload={"temperature": 0},
        )
        try:
            retry_resp = client.generate(retry_req)
            retry_raw = retry_resp.text.strip().strip("```json").strip("```").strip()
            retry_data = json.loads(retry_raw)

            retry_variables = [
                ControlVariable(variable=v.get("variable", ""), reason=v.get("reason", ""))
                for v in retry_data.get("variables_to_control", [])
                if isinstance(v, dict)
            ]
            retry_questions = [
                ResearchQuestion(question=q.get("question", ""), source_finding=q.get("source_finding", ""))
                for q in retry_data.get("research_questions", [])
                if isinstance(q, dict)
            ]
            retry_narrative = str(retry_data.get("narrative_summary", ""))

            synthesis = SynthesisOutput(
                future_trial_hypothesis=str(retry_data.get("future_trial_hypothesis", "")),
                endpoint_improvement_recommendations=[
                    str(r) for r in retry_data.get("endpoint_improvement_recommendations", [])
                ],
                recommended_sample_size=str(retry_data.get("recommended_sample_size", "")),
                variables_to_control=retry_variables,
                research_questions=retry_questions,
                narrative_summary=retry_narrative,
            )
            narrative = retry_narrative

            retry_errors = validate_synthesis_output(synthesis, grounded_findings)
            if retry_errors:
                synthesis = synthesis.model_copy(
                    update={
                        "narrative_summary": synthesis.narrative_summary
                        + "\n\n[VALIDATION WARNINGS]: "
                        + str(retry_errors)
                    }
                )
        except Exception:
            synthesis = synthesis.model_copy(
                update={
                    "narrative_summary": synthesis.narrative_summary
                    + "\n\n[VALIDATION WARNINGS]: "
                    + str(errors)
                }
            )

    return synthesis, narrative


# ── Confidence warning annotation (internal critic output) ───────────────────

def _annotate_confidence_warnings(
    grounded_findings: GroundedFindingsSchema | None,
    evidence: dict[str, Any],
    tool_log: list[ToolCallRecord] | None,
) -> GroundedFindingsSchema | None:
    """
    Post-loop conditional checks (internal critic, Step 4 of design).
    Attaches confidence_warning to findings where:
    - Column missingness > 30%
    - Sample size below minimum for the test used
    These are deterministic threshold checks, NOT LLM calls.
    """
    if grounded_findings is None or not grounded_findings.findings or not tool_log:
        return grounded_findings

    missingness_pct = (
        evidence.get("missingness_pct")
        if isinstance(evidence.get("missingness_pct"), dict)
        else {}
    )
    tool_log_by_alias = {rec.citation_alias: rec for rec in tool_log if rec.citation_alias}

    updated_findings = []
    for finding in grounded_findings.findings:
        warnings: list[str] = []

        # Check tool_log citations embedded in finding text
        import re
        aliases = re.findall(r"tool_log\[\d+\]", finding.finding_text)
        for alias in aliases:
            record = tool_log_by_alias.get(alias)
            if record is None:
                continue
            # Missingness check
            args = record.args if isinstance(record.args, dict) else {}
            for key in ("column", "numeric_column", "time_column", "event_column"):
                col = args.get(key)
                if isinstance(col, str):
                    miss = missingness_pct.get(col)
                    if isinstance(miss, (int, float)) and float(miss) > 30.0:
                        warnings.append("Based on a column with high missingness — interpret with caution.")
                        break
            # Sample size check
            result = record.result if isinstance(record.result, dict) else {}
            if record.tool_name in {"hypothesis_test", "effect_size"}:
                ga = result.get("group_a") or {}
                gb = result.get("group_b") or {}
                try:
                    if min(int(ga.get("n", 99)), int(gb.get("n", 99))) < 2:
                        warnings.append("Sample size may be insufficient for this test.")
                except (TypeError, ValueError):
                    pass

        deduped = list(dict.fromkeys(warnings))
        if deduped:
            updated_findings.append(
                finding.model_copy(update={"confidence_warning": deduped[0] if len(deduped) == 1 else deduped})
            )
        else:
            updated_findings.append(finding)

    return grounded_findings.model_copy(update={"findings": updated_findings})


def _build_qc_finding(text: str, category: str) -> GroundedFinding:
    return GroundedFinding(
        finding_text=text,
        finding_text_raw=text,
        finding_text_plain=text,
        finding_category=category,  # type: ignore[arg-type]
        claim_type=classify_claim_type(text, finding_category=category),
        grounding_status=neutral_status_for_category(category),
        literature_skipped=True,
        literature_skip_note=(
            "This note was excluded from literature grounding because it reflects "
            "data quality, preprocessing, or pipeline context rather than a "
            "clinically meaningful analytical result."
        ),
    )


def _sanitize_clinical_analysis(clinical_analysis: dict | None) -> dict | None:
    """Final backend gate: malformed and non-clinical items cannot remain primary findings."""
    if not isinstance(clinical_analysis, dict):
        return clinical_analysis
    sanitized = dict(clinical_analysis)
    stage_3 = sanitized.get("stage_3_corrections")
    if not isinstance(stage_3, dict):
        return sanitized

    corrected = stage_3.get("corrected_findings")
    if not isinstance(corrected, list):
        return sanitized

    clean_findings: list[Any] = []
    excluded: list[Any] = list(stage_3.get("artifact_excluded_findings") or [])
    for finding in corrected:
        if is_malformed_finding_fragment(finding):
            excluded.append(_artifact_excluded_payload(finding))
            continue
        if is_user_facing_nonfinding_artifact(finding):
            excluded.append(_artifact_excluded_payload(finding))
            continue
        updated = _demote_nonclinical_finding_payload(finding)
        if (
            isinstance(updated, dict)
            and is_analytical_category(str(updated.get("finding_category") or "analytical"))
            and not is_user_facing_clinical_finding_eligible(updated)
        ):
            excluded.append(_artifact_excluded_payload(updated))
            continue
        clean_findings.append(updated)

    updated_stage_3 = dict(stage_3)
    updated_stage_3["corrected_findings"] = clean_findings
    if excluded:
        updated_stage_3["artifact_excluded_findings"] = excluded
    sanitized["stage_3_corrections"] = updated_stage_3
    return sanitized


def _sanitize_grounded_findings_schema(
    grounded_findings: GroundedFindingsSchema | None,
) -> GroundedFindingsSchema | None:
    if grounded_findings is None:
        return None
    sanitized_findings: list[GroundedFinding] = []
    seen: set[str] = set()
    for finding in grounded_findings.findings:
        if is_malformed_finding_fragment(finding):
            continue
        if is_user_facing_nonfinding_artifact(finding):
            text = _finding_display_text(finding)
            existing_category = getattr(finding, "finding_category", None)
            category = existing_category if not is_analytical_category(existing_category) else "statistical_note"
            qc_finding = finding.model_copy(
                update={
                    "finding_text": text,
                    "finding_text_plain": text,
                    "finding_text_raw": text,
                    "finding_category": category,
                    "claim_type": classify_claim_type(text, finding_category=category),
                    "grounding_status": neutral_status_for_category(category),
                    "literature_skipped": True,
                    "literature_skip_note": (
                        "Raw variable-only statistical artifact excluded from primary "
                        "analytical findings and retained only as a statistical note."
                    ),
                }
            )
            key = f"artifact:{text.lower()}"
            if key not in seen:
                seen.add(key)
                sanitized_findings.append(qc_finding)
            continue
        finding = _demote_nonclinical_grounded_finding(finding)
        if is_analytical_category(finding.finding_category) and not is_user_facing_clinical_finding_eligible(finding):
            continue
        key = _finding_display_text(finding).lower()
        if key in seen:
            continue
        seen.add(key)
        sanitized_findings.append(finding)
    return grounded_findings.model_copy(update={"findings": sanitized_findings})


def _sanitize_analytical_grounded_findings(
    grounded_findings: list[GroundedFinding],
) -> list[GroundedFinding]:
    return [
        finding for finding in grounded_findings
        if not is_user_facing_nonfinding_artifact(finding)
        and is_analytical_category(getattr(finding, "finding_category", "analytical"))
        and is_user_facing_clinical_finding_eligible(finding)
    ]


def _sanitize_final_report(report: FinalReportSchema) -> FinalReportSchema:
    """Final user-facing report sanitation boundary."""
    clinical_analysis = _sanitize_clinical_analysis(report.clinical_analysis)
    grounded_findings = _sanitize_grounded_findings_schema(report.grounded_findings)
    final_insights = report.final_insights
    if final_insights is not None:
        final_insights = final_insights.model_copy(
            update={
                "key_findings": [
                    text for text in final_insights.key_findings
                    if not is_user_facing_nonfinding_artifact(text)
                    and is_analytical_category(classify_finding_category(text))
                    and is_user_facing_clinical_finding_eligible(text)
                ],
                "risks_and_bias_signals": [
                    text for text in final_insights.risks_and_bias_signals
                    if not is_user_facing_nonfinding_artifact(text)
                    and is_analytical_category(classify_finding_category(text))
                    and is_user_facing_clinical_finding_eligible(text)
                ],
            }
        )
    sanitized = report.model_copy(
        update={
            "clinical_analysis": clinical_analysis,
            "grounded_findings": grounded_findings,
            "final_insights": final_insights,
        }
    )
    return curate_user_facing_report_sections(sanitized)


def _artifact_excluded_payload(finding: Any) -> dict[str, Any]:
    if isinstance(finding, dict):
        payload = dict(finding)
    else:
        payload = {"finding_text": _finding_display_text(finding)}
    payload["finding_category"] = "statistical_note"
    payload["claim_type"] = "statistical_note"
    payload["artifact_exclusion_reason"] = (
        "Non-finding statistical artifact excluded from primary analytical findings."
    )
    return payload


def _demote_nonclinical_finding_payload(finding: Any) -> Any:
    if not isinstance(finding, dict):
        return finding
    text = _finding_display_text(finding)
    if not text:
        return finding
    current_category = str(finding.get("finding_category") or "analytical")
    inferred = classify_finding_category(
        text,
        variable=finding.get("variable"),
        endpoint=finding.get("endpoint"),
        analysis_type=finding.get("analysis_type"),
    )
    if is_analytical_category(current_category) and not is_analytical_category(inferred):
        payload = dict(finding)
        payload["finding_category"] = inferred
        payload["claim_type"] = classify_claim_type(text, finding_category=inferred)
        return payload
    return finding


def _demote_nonclinical_grounded_finding(finding: GroundedFinding) -> GroundedFinding:
    text = _finding_display_text(finding)
    inferred = classify_finding_category(
        text,
        variable=finding.variable,
        endpoint=finding.endpoint,
    )
    if not is_analytical_category(finding.finding_category) or is_analytical_category(inferred):
        return finding
    return finding.model_copy(
        update={
            "finding_category": inferred,
            "claim_type": classify_claim_type(text, finding_category=inferred),
            "grounding_status": neutral_status_for_category(inferred),
            "literature_skipped": True,
            "literature_skip_note": (
                "This item was moved out of analytical findings because it describes "
                "data quality, preprocessing, statistical setup, or study context."
            ),
        }
    )


def _finding_display_text(finding: Any) -> str:
    if isinstance(finding, str):
        return finding.strip()
    fields = (
        "comparison_claim_text",
        "finding_text_plain",
        "finding_text",
        "source_finding_plain",
        "source_finding",
        "source_text",
        "finding_text_raw",
        "raw",
    )
    if isinstance(finding, dict):
        for field in fields:
            value = finding.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    for field in fields:
        value = getattr(finding, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_agentic_insights(
    df: pd.DataFrame,
    provider: ProviderName,
    max_rows: int = 25,
    max_cols: int = 30,
    run_judge: bool = False,          # kept for API compat, ignored
    metadata: MetadataInput | None = None,  # kept for API compat, unused
    interactive: bool = False,         # kept for API compat, ignored
    analysis_report: str | None = None,
    tool_log: list[dict[str, Any]] | None = None,
    emit: Callable | None = None,
    study_context: str = "",
    column_dict: dict[str, str] | None = None,
    missingness_disclosures: list[MissingnessDisclosure] | None = None,
    clinical_analysis: dict | None = None,
    methodology_chapter: str | None = None,
    analyst_report_text: str | None = None,
    analyst_report_name: str | None = None,
) -> FinalReportSchema:
    """
    Run the Q-Trial pipeline as specified in the design document.

    Steps:
      3. Data Profiler + Data Quality  (pure Python)
      5. Literature Query Translation  (LLM mini-call per finding)
      6. Literature Validator          (API calls: PubMed, Cochrane, CT.gov)
      7. Synthesis + Self-Scoring      (single LLM call)

    Steps 1-2 and 4 are handled upstream (api.py / runner.py).
    Step 8 (Report Generator) is handled by the frontend.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    import os, numpy as np, time as _time
    _seed = int(os.environ.get("ANALYSIS_SEED", "42"))
    np.random.seed(_seed)
    _run_id = f"{provider}_{int(_time.time())}"
    _repro = ReproducibilityLogBuilder(
        run_id=_run_id,
        study_context=study_context,
        seed=_seed,
    )

    def _emit(event_type: str, stage: str, message: str) -> None:
        if emit is not None:
            try:
                emit({"type": event_type, "stage": stage, "message": message})
            except Exception:
                pass

    typed_tool_log = _coerce_tool_log(tool_log)

    # ── Step 3: Data Profiler + Data Quality (pure Python, no LLM) ───────────
    console.print("[bold cyan]► Step 3:[/bold cyan] Data profiling…")
    preview = build_dataset_preview(df, max_rows=max_rows, max_cols=max_cols)
    evidence = build_dataset_evidence(df)
    citations = format_citations(evidence)

    _guardrail_raw = run_guardrails(df)
    guardrail_report = GuardrailReport.model_validate(_guardrail_raw)
    evidence["__guardrails__"] = _guardrail_raw
    guardrail_citations = format_guardrail_citations(_guardrail_raw)
    if guardrail_citations:
        citations["guardrails"] = guardrail_citations

    if column_dict:
        evidence["__column_dict__"] = column_dict

    _g_flags = _guardrail_raw["flags"]
    _g_hi = sum(1 for f in _g_flags if f["severity"] == "high")
    if _g_flags:
        console.print(
            f"  [bold yellow]⚠ Guardrails:[/bold yellow] "
            f"{len(_g_flags)} flag(s) ({_g_hi} high)"
        )
    else:
        console.print("  [green]✓ Guardrails: all checks passed[/green]")
    _emit("stage_complete", "dataset", "Data profiling complete")

    # ── Step 4 output: analysis_report + tool_log come from api.py ───────────
    # (run_statistical_agent_loop is called before this function)
    try:
        _client = get_client(provider)
        model_name: str = getattr(_client, "model", str(provider))
    except Exception as exc:
        console.print(f"[yellow]! Model metadata unavailable for {provider}: {exc}[/yellow]")
        model_name = str(provider)

    if not analysis_report:
        console.print(
            "  [yellow]⚠ No statistical analysis report received — "
            "synthesis will have limited grounding.[/yellow]"
        )
    _emit("stage_complete", "StatisticalAgent", "Statistical analysis complete")

    # ── Step 5: Literature Query Translation (LLM mini-call) ─────────────────
    # Extract key findings from the analysis report for CST translation.
    # The analysis_report is a Markdown string; we use it directly as the
    # source of findings for translation.
    grounded_findings_list: list[GroundedFinding] = []
    analytical_grounded_findings: list[GroundedFinding] = []
    grounded_findings_schema: GroundedFindingsSchema | None = None
    synthesis_quality: SynthesisQualityScore | None = None
    synthesis_output: SynthesisOutput | None = None
    narrative_summary: str = ""
    comparison_report: ComparisonReport | None = None
    statistical_verification_report = None

    corrected_findings = (
        (clinical_analysis or {}).get("stage_3_corrections", {})
        if isinstance(clinical_analysis, dict)
        else {}
    ).get("corrected_findings", [])
    if isinstance(corrected_findings, list) and corrected_findings:
        try:
            verbalized = verbalize_statistical_findings(corrected_findings, provider)
            normalized_for_comparison = normalize_comparison_claims(verbalized, provider)
            clinical_analysis = dict(clinical_analysis or {})
            stage_3 = dict(clinical_analysis.get("stage_3_corrections", {}))
            stage_3["corrected_findings"] = normalized_for_comparison
            clinical_analysis["stage_3_corrections"] = stage_3
            clinical_analysis = _sanitize_clinical_analysis(clinical_analysis)
        except Exception as exc:
            console.print(f"[yellow]⚠ Statistical finding verbalization skipped: {exc}[/yellow]")

    clinical_analysis = _sanitize_clinical_analysis(clinical_analysis)

    if study_context and analysis_report:
        try:
            console.print("[bold cyan]► Step 5:[/bold cyan] Literature query translation…")
            # Extract human-readable statistical findings from the analysis report.
            # The analysis_report is the agent's prose output — sentences containing
            # actual statistical signals are the correct source for CST translation
            # and for finding_text shown to clinicians.
            # The tool_log is NOT used as finding text — it contains raw JSON.
            import re as _re
            findings_for_cst: list[str | dict[str, Any]] = []
            supplemental_qc_findings: list[GroundedFinding] = []
            seen_qc_texts: set[str] = set()

            corrected_findings = (
                (clinical_analysis or {}).get("stage_3_corrections", {})
                if isinstance(clinical_analysis, dict)
                else {}
            ).get("corrected_findings", [])
            if isinstance(corrected_findings, list):
                for finding in corrected_findings:
                    if not isinstance(finding, dict):
                        continue
                    category = str(finding.get("finding_category") or "analytical")
                    plain_text = str(finding.get("finding_text_plain") or "").strip()
                    raw_text = str(finding.get("finding_text_raw") or "").strip()
                    if not plain_text and not raw_text:
                        continue
                    best_text = plain_text or raw_text
                    if (
                        is_user_facing_nonfinding_artifact(finding)
                        or is_raw_statistical_artifact_text(best_text)
                        or not is_user_facing_clinical_finding_eligible(finding)
                    ):
                        if is_analytical_category(category):
                            continue
                    if not is_analytical_category(category):
                        if best_text.lower() not in seen_qc_texts:
                            seen_qc_texts.add(best_text.lower())
                            supplemental_qc_findings.append(_build_qc_finding(best_text, category))
                        continue
                    findings_for_cst.append(
                        {
                            "finding_text_plain": best_text,
                            "finding_text_raw": raw_text or plain_text,
                            "comparison_claim_text": str(finding.get("comparison_claim_text") or "").strip() or None,
                            "finding_category": category,
                            "claim_type": str(finding.get("claim_type") or "association_claim"),
                            "variable": finding.get("variable"),
                            "endpoint": finding.get("endpoint"),
                            "direction": finding.get("direction") or "unknown",
                            "direction_label": finding.get("direction_label"),
                            "significant": finding.get("significant_after_correction", finding.get("significant")),
                            "significance": (
                                "significant"
                                if finding.get("significant_after_correction") is True
                                else "not_significant"
                                if finding.get("significant_after_correction") is False
                                else "unclear"
                            ),
                            "p_value": finding.get("adjusted_p_value", finding.get("raw_p_value")),
                            "effect_size": finding.get("effect_size"),
                            "effect_size_label": finding.get("effect_size_label"),
                            "test_type": finding.get("test_type"),
                            "confidence_warning": finding.get("confidence_warning"),
                            "metadata": finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {},
                        }
                    )
                    if len(findings_for_cst) >= 10:
                        break

            if analysis_report and len(findings_for_cst) < 10:
                _stat_pattern = _re.compile(
                    r"(p\s*[=<>]\s*0?\.\d+|p\s*[=<>]\s*\d|"
                    r"\bHR\b|\bOR\b|\bRR\b|\bAUC\b|\bCI\b|"
                    r"hazard ratio|odds ratio|risk ratio|"
                    r"correlation|r\s*=\s*[-−]?0?\.\d+|"
                    r"coefficient|regression|survival|"
                    r"statistically significant|not significant|"
                    r"median.*day|mean.*day|"
                    r"\d+\.\d+.*%|\d+%.*differ)",
                    _re.IGNORECASE,
                )
                for sentence in _re.split(r"(?<=[.!?])\s+|\n", analysis_report):
                    s = sentence.strip()
                    if not s or s.startswith("|") or s.startswith("#") or s.startswith("---") or len(s) < 30:
                        continue
                    if is_methodology_instruction_text(s):
                        continue
                    if (
                        is_user_facing_nonfinding_artifact(s)
                        or is_raw_statistical_artifact_text(s)
                        or not is_user_facing_clinical_finding_eligible(s)
                    ):
                        if (
                            (is_user_facing_nonfinding_artifact(s) or is_raw_statistical_artifact_text(s))
                            and s.lower() not in seen_qc_texts
                        ):
                            seen_qc_texts.add(s.lower())
                            supplemental_qc_findings.append(_build_qc_finding(s, "statistical_note"))
                        continue
                    if _stat_pattern.search(s):
                        category = classify_finding_category(s)
                        if is_analytical_category(category):
                            findings_for_cst.append(
                                {
                                    "finding_text_plain": s,
                                    "finding_text_raw": s,
                                    "finding_category": category,
                                    "claim_type": classify_claim_type(s, finding_category=category),
                                }
                            )
                            if len(findings_for_cst) >= 10:
                                break
                        elif s.lower() not in seen_qc_texts:
                            seen_qc_texts.add(s.lower())
                            supplemental_qc_findings.append(_build_qc_finding(s, category))

            cst_candidates_before_sanitation = len(findings_for_cst)
            cst_skipped_examples: list[str] = []
            sanitized_findings_for_cst: list[str | dict[str, Any]] = []
            for finding in findings_for_cst:
                if is_user_facing_nonfinding_artifact(finding) or not is_user_facing_clinical_finding_eligible(finding):
                    example = _finding_display_text(finding)
                    if example and len(cst_skipped_examples) < 3:
                        cst_skipped_examples.append(example[:180])
                    continue
                sanitized_findings_for_cst.append(finding)
            findings_for_cst = sanitized_findings_for_cst
            console.print(
                f"  [cyan]CST candidates:[/cyan] input={cst_candidates_before_sanitation}, "
                f"after_sanitation={len(findings_for_cst)}, skipped={cst_candidates_before_sanitation - len(findings_for_cst)}"
            )
            if cst_skipped_examples:
                console.print(f"  [dim]CST skipped examples: {cst_skipped_examples}[/dim]")

            if not findings_for_cst:
                csts = []
                console.print("  [yellow]⚠ CST translation skipped: no valid clinical findings remained.[/yellow]")
            else:
                csts = translate_findings_to_cst(findings_for_cst, study_context, provider)
            _repro.add_csts(csts)
            _emit("stage_complete", "cst_translation", f"Translated {len(csts)} findings to search terms")
            console.print(f"  [green]✓ {len(csts)} search terms generated[/green]")

            # ── Step 6: Literature Validator (API calls) ──────────────────────
            if csts:
                console.print("[bold cyan]► Step 6:[/bold cyan] Literature validation…")
                lit_pipeline = LiteratureValidatorPipeline(provider=provider)
                analytical_grounded_findings = _sanitize_analytical_grounded_findings(lit_pipeline.validate(csts))
                grounded_findings_list = analytical_grounded_findings + supplemental_qc_findings
                _repro.add_literature_queries(lit_pipeline.query_records)
            else:
                console.print("[bold cyan]► Step 6:[/bold cyan] Literature validation skipped (no CSTs).")
                analytical_grounded_findings = []
                grounded_findings_list = supplemental_qc_findings
            _emit(
                "stage_complete",
                "literature_validation",
                (
                    f"Literature: {len(analytical_grounded_findings)} analytical findings grounded; "
                    f"{len(supplemental_qc_findings)} QC notes separated"
                ),
            )
            n_supported = sum(1 for g in analytical_grounded_findings if g.grounding_status == "Supported")
            n_novel = sum(1 for g in analytical_grounded_findings if g.grounding_status == "Novel")
            console.print(
                f"  [green]✓ Literature:[/green] "
                f"{n_supported} supported, {n_novel} novel, "
                f"{len(analytical_grounded_findings) - n_supported - n_novel} contradicted"
                + (
                    f"; {len(supplemental_qc_findings)} QC note(s) kept separate"
                    if supplemental_qc_findings
                    else ""
                )
            )

            # ── Step 7: Synthesis + Self-Scoring (single LLM call) ───────────
            console.print("[bold cyan]► Step 7:[/bold cyan] Synthesis…")
            synthesis_output, narrative_summary = run_synthesis_call(
                analysis_report=analysis_report,
                grounded_findings=analytical_grounded_findings,
                study_context=study_context,
                provider=provider,
            )
            _emit("stage_complete", "synthesis", "Synthesis complete")
            console.print("  [green]✓ Synthesis complete[/green]")
            _emit("stage_complete", "synthesis_scoring", "Deterministic validation applied")

            grounded_findings_schema = GroundedFindingsSchema(
                findings=grounded_findings_list,
                research_questions=synthesis_output.research_questions,
                synthesis=synthesis_output,
                excluded_columns=[
                    ExcludedColumn(column=d.column, missingness_rate=d.missingness_rate)
                    for d in (missingness_disclosures or [])
                    if d.action == "excluded"
                ],
                high_missingness_columns=[
                    HighMissingnessColumn(
                        column=d.column,
                        missingness_rate=d.missingness_rate,
                        excluded_from_primary_analysis=d.excluded_from_primary_analysis,
                    )
                    for d in (missingness_disclosures or [])
                    if d.action == "high_missingness_section"
                ],
                listwise_deletion_columns=[
                    ListwiseDeletionColumn(
                        column=d.column,
                        missingness_rate=d.missingness_rate,
                        rows_dropped=d.rows_dropped,
                    )
                    for d in (missingness_disclosures or [])
                    if d.action == "listwise_deletion" and d.missingness_rate > 0
                ],
            )
            grounded_findings_schema = _sanitize_grounded_findings_schema(grounded_findings_schema)

        except Exception as exc:
            console.print(f"  [yellow]⚠ Steps 5-7 failed: {exc}[/yellow]")

    elif not study_context:
        console.print(
            "  [yellow]⚠ No study context — skipping literature validation and synthesis.[/yellow]"
        )

    # ── Internal critic: annotate confidence warnings (deterministic) ─────────
    grounded_findings_schema = _annotate_confidence_warnings(
        grounded_findings=grounded_findings_schema,
        evidence=evidence,
        tool_log=typed_tool_log,
    )
    grounded_findings_schema = _sanitize_grounded_findings_schema(grounded_findings_schema)

    # ── Reproducibility log ───────────────────────────────────────────────────
    repro_log = _repro.finalise(synthesis_quality_score=synthesis_quality)

    # ── Assemble FinalReportSchema ────────────────────────────────────────────
    # Build a minimal InsightSynthesisOutput from grounded findings for
    # backward-compat fields that the frontend may still read.
    from qtrial_backend.agentic.schemas import InsightSynthesisOutput, RankedAnalysis, PlanSchema, PlanStep, AgentRunRecord
    analytical_grounded_findings = _sanitize_analytical_grounded_findings(analytical_grounded_findings)
    _final_insights = InsightSynthesisOutput(
        key_findings=[gf.finding_text for gf in analytical_grounded_findings],
        risks_and_bias_signals=[
            gf.finding_text for gf in analytical_grounded_findings
            if gf.grounding_status == "Contradicted"
        ],
        recommended_next_analyses=[
            RankedAnalysis(
                rank=i + 1,
                analysis=q.question,
                rationale=q.source_finding,
                evidence_citation="",
            )
            for i, q in enumerate(
                synthesis_output.research_questions if synthesis_output else []
            )
        ],
        required_metadata_or_questions=[],
    )

    # Minimal plan stub (planner is not part of the design pipeline)
    _plan = PlanSchema(
        dataset_summary=f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols",
        steps=[
            PlanStep(
                step_number=1,
                name="Statistical Analysis",
                goal="Run statistical agent loop",
                inputs_used=["dataframe"],
                expected_output_keys=["analysis_report"],
                agent_to_call="DataQualityAgent",  # placeholder for schema compat
            )
        ],
    )

    report = FinalReportSchema(
        provider=provider,
        model=model_name,
        plan=_plan,
        agent_runs=[
            AgentRunRecord(
                step_number=1,
                agent="StatisticalAnalysisAgent",
                goal="LLM agent loop with 30+ statistical tools",
                output={"analysis_report": analysis_report or ""},
            )
        ],
        unknowns=UnknownsOutput(
            ranked_unknowns=[],
            explicit_assumptions=[],
            required_documents=[],
            summary="",
        ),
        final_insights=_final_insights,
        judge=None,
        metadata_used=None,
        final_insights_before=None,
        final_insights_after=None,
        judge_before=None,
        judge_after=None,
        prior_analysis_report=(
            (analysis_report or "") + "\n\n" + methodology_chapter
            if methodology_chapter
            else analysis_report
        ),
        tool_log=typed_tool_log,
        reasoning_state=None,
        guardrail_report=guardrail_report,
        literature_report=None,
        study_context=study_context or None,
        grounded_findings=grounded_findings_schema,
        reproducibility_log=repro_log,
        synthesis_quality_score=synthesis_quality,
        treatment_columns_excluded=detect_treatment_columns(df),
        clinical_analysis=clinical_analysis,
        statistical_verification_report=statistical_verification_report,
        comparison_report=None,
    )
    report = _sanitize_final_report(report)

    if analyst_report_text and analyst_report_name:
        try:
            report = _sanitize_final_report(report)
            report = report.model_copy(
                update={
                    "statistical_verification_report": build_statistical_verification_report(
                        df=df,
                        qtrial_findings=normalize_qtrial_findings(report),
                        analyst_report_text=analyst_report_text,
                        analyst_report_name=analyst_report_name,
                        metadata=metadata,
                        column_dict=column_dict,
                    )
                }
            )
            report = _sanitize_final_report(report)
        except Exception as exc:
            console.print(f"[yellow]⚠ Statistical verification skipped: {exc}[/yellow]")

        try:
            report = _sanitize_final_report(report)
            report = report.model_copy(
                update={
                    "comparison_report": build_comparison_report(
                        final_report=report,
                        analyst_report_name=analyst_report_name,
                        analyst_report_text=analyst_report_text,
                        provider=provider,
                    )
                }
            )
            report = _sanitize_final_report(report)
            _emit("stage_complete", "comparison", "Automated report comparison complete")
        except Exception as exc:
            console.print(f"[yellow]⚠ Comparison skipped: {exc}[/yellow]")

    report = _sanitize_final_report(report)
    report_dict = report.model_dump()
    if typed_tool_log is not None:
        report_dict["tool_log"] = _compact_tool_log_for_persistence(typed_tool_log)

    OUTPUT_FILE.write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    console.print(f"[bold green]✔ Saved →[/bold green] {OUTPUT_FILE}")

    return report
