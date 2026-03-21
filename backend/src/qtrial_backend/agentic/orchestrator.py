from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from qtrial_backend.agentic.agents import (
    run_data_quality_agent,
    run_clinical_semantics_agent,
    run_unknowns_agent,
    run_insight_synthesis_agent,
)
from qtrial_backend.agentic.judge import run_judge_agent
from qtrial_backend.agentic.planner import call_planner
from typing import Any, Callable

from rich.prompt import Confirm, Prompt

from qtrial_backend.agentic.reasoning import (
    _extract_inline_citations,
    run_reasoning_engine,
)
from qtrial_backend.agentic.schemas import (
    AgentRunRecord,
    FinalReportSchema,
    GroundedFindingsSchema,
    GuardrailReport,
    InsightSynthesisOutput,
    LiteratureRAGReport,
    MetadataInput,
    ToolCallRecord,
    UnknownsOutput,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import ProviderName
from qtrial_backend.dataset.evidence import build_dataset_evidence, format_citations
from qtrial_backend.dataset.guardrails import format_guardrail_citations, run_guardrails
from qtrial_backend.dataset.preview import build_dataset_preview
from qtrial_backend.tools.literature.rag import (
    format_literature_for_agents,
    run_literature_rag,
)
from qtrial_backend.agentic.cst_translator import translate_findings_to_cst
from qtrial_backend.agentic.literature_validator import LiteratureValidatorPipeline
from qtrial_backend.agentic.synthesis_scorer import (
    score_synthesis_quality,
    SYNTHESIS_QUALITY_THRESHOLD,
)
from qtrial_backend.agentic.reproducibility import ReproducibilityLogBuilder

console = Console()

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "agentic_run.json"


# ── Tool-log coerce / compact helpers ────────────────────────────────────────

def _coerce_tool_log(
    raw_tool_log: list[dict[str, Any]] | None,
) -> list[ToolCallRecord] | None:
    """
    Convert a raw tool_log list (as produced by AgentLoop) into typed
    ``ToolCallRecord`` objects with stable ``citation_alias`` values.

    AgentLoop.tool_log entries have keys: tool, args, result, [error].
    Returns None when input is None or empty.
    """
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
    """
    Produce a compact JSON-serialisable list for outputs/agentic_run.json.
    Result strings are truncated to 2000 chars to keep the file manageable.
    """
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


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_columns_from_tool_record(
    record: ToolCallRecord,
    available_columns: set[str],
) -> list[str]:
    candidates: list[str] = []

    args = record.args if isinstance(record.args, dict) else {}
    result = record.result if isinstance(record.result, dict) else {}

    scalar_keys = (
        "column",
        "numeric_column",
        "group_column",
        "time_column",
        "event_column",
    )
    list_keys = ("columns", "group_columns", "target_columns")

    for source in (args, result):
        for key in scalar_keys:
            value = source.get(key)
            if isinstance(value, str) and value in available_columns:
                candidates.append(value)
        for key in list_keys:
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in available_columns:
                        candidates.append(item)

    return list(dict.fromkeys(candidates))


def _sample_size_warning_for_tool_record(record: ToolCallRecord) -> str | None:
    result = record.result if isinstance(record.result, dict) else {}
    tool_name = record.tool_name

    if tool_name in {"hypothesis_test", "effect_size"}:
        # Both tools explicitly require at least 2 observations per group.
        # Source:
        # - tools/stats/hypothesis_test.py: "Need at least 2 per group."
        # - tools/stats/effect_size.py: "Need at least 2 per group."
        group_a = result.get("group_a") if isinstance(result.get("group_a"), dict) else {}
        group_b = result.get("group_b") if isinstance(result.get("group_b"), dict) else {}
        n_a = _coerce_int(group_a.get("n"))
        n_b = _coerce_int(group_b.get("n"))
        if n_a is not None and n_b is not None and min(n_a, n_b) < 2:
            return "Sample size may be insufficient for this test."

    if tool_name == "pairwise_group_test":
        # Pairwise Mann-Whitney subtests are skipped when either side has <2
        # observations, so use that repo-defined minimum for pair-level warnings.
        # Source: tools/stats/pairwise_test.py -> "skipped": "too few observations"
        pairwise_results = result.get("pairwise_results")
        if isinstance(pairwise_results, list):
            for entry in pairwise_results:
                if not isinstance(entry, dict):
                    continue
                n_a = _coerce_int(entry.get("n_a"))
                n_b = _coerce_int(entry.get("n_b"))
                if n_a is not None and n_b is not None and min(n_a, n_b) < 2:
                    return "Sample size may be insufficient for this test."

    return None


def _annotate_confidence_warnings(
    grounded_findings: GroundedFindingsSchema | None,
    evidence: dict[str, Any],
    tool_log: list[ToolCallRecord] | None,
) -> GroundedFindingsSchema | None:
    if grounded_findings is None or not grounded_findings.findings or not tool_log:
        return grounded_findings

    missingness_pct = (
        evidence.get("missingness_pct")
        if isinstance(evidence.get("missingness_pct"), dict)
        else {}
    )
    available_columns = set(missingness_pct.keys())
    tool_log_by_alias = {
        rec.citation_alias: rec for rec in tool_log if rec.citation_alias
    }

    updated_findings = []
    for finding in grounded_findings.findings:
        warnings: list[str] = []
        aliases = [
            cite for cite in _extract_inline_citations(finding.finding_text)
            if cite.startswith("tool_log[")
        ]

        for alias in aliases:
            record = tool_log_by_alias.get(alias)
            if record is None:
                continue

            columns = _extract_columns_from_tool_record(record, available_columns)
            if any(
                isinstance(missingness_pct.get(column), (int, float))
                and float(missingness_pct[column]) > 30.0
                for column in columns
            ):
                warnings.append(
                    "Based on a column with high missingness — interpret with caution."
                )

            sample_warning = _sample_size_warning_for_tool_record(record)
            if sample_warning:
                warnings.append(sample_warning)

        deduped_warnings = list(dict.fromkeys(warnings))
        if deduped_warnings:
            warning_value: str | list[str]
            if len(deduped_warnings) == 1:
                warning_value = deduped_warnings[0]
            else:
                warning_value = deduped_warnings
            updated_findings.append(
                finding.model_copy(update={"confidence_warning": warning_value})
            )
        else:
            updated_findings.append(finding)

    return grounded_findings.model_copy(update={"findings": updated_findings})


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _render_metadata_block(meta: MetadataInput) -> str:
    """Serialize metadata to a plain-text block agents can read inline."""
    parts: list[str] = [
        "=== AUTHORITATIVE METADATA (provided by user) ===",
    ]
    if meta.status_mapping:
        pairs = "; ".join(f"{k}→{v}" for k, v in meta.status_mapping.items())
        parts.append(f"Status column coding: {pairs}")
    if meta.primary_endpoint:
        parts.append(f"Primary endpoint: {meta.primary_endpoint}")
    if meta.time_unit:
        parts.append(f"Time unit: {meta.time_unit}")
    if meta.study_design:
        parts.append(f"Study design: {meta.study_design}")
    if meta.treatment_arms:
        pairs = "; ".join(f"{k}→{v}" for k, v in meta.treatment_arms.items())
        parts.append(f"Treatment arms: {pairs}")
    if meta.lab_units:
        for lu in meta.lab_units:
            rng = f" (range: {lu.normal_range})" if lu.normal_range else ""
            parts.append(f"  {lu.column}: {lu.unit}{rng}")
    if meta.additional_answers:
        for q, a in meta.additional_answers.items():
            parts.append(f"  Q: {q}")
            parts.append(f"  A: {a}")
    parts.append("=== END METADATA ===")
    return "\n".join(parts)


def _metadata_covers_semantics(meta: MetadataInput) -> bool:
    """True when the metadata touches endpoint, column roles, or study design."""
    return (
        meta.status_mapping is not None
        or meta.primary_endpoint is not None
        or meta.treatment_arms is not None
        or meta.study_design is not None
        or meta.lab_units is not None
    )


def _tag_unresolved_unknowns(
    unknowns: UnknownsOutput,
    meta: MetadataInput | None,
) -> UnknownsOutput:
    """
    Walk through high-impact unknowns and see which ones are NOT answered by
    the provided metadata.  Those unanswered questions go into
    ``unresolved_high_impact`` so synthesis can downgrade certainty.
    """
    high_qs = [u for u in unknowns.ranked_unknowns if u.impact == "high"]
    if not high_qs:
        return unknowns

    # No metadata at all → every high-impact question is unresolved
    if meta is None:
        return unknowns.model_copy(
            update={"unresolved_high_impact": [u.question for u in high_qs]},
        )

    # Map signal words to the metadata attribute they would resolve
    _SIGNALS: list[tuple[list[str], str]] = [
        (["status", "event", "outcome", "endpoint", "censored", "coding"],
         "status_mapping"),
        (["time", "duration", "follow-up", "days", "months", "years"],
         "time_unit"),
        (["lab", "unit", "bilirubin", "albumin", "cholesterol", "platelet"],
         "lab_units"),
        (["design", "rct", "randomis", "observational", "cohort"],
         "study_design"),
        (["arm", "treatment", "drug", "placebo", "allocation"],
         "treatment_arms"),
        (["primary endpoint", "primary outcome", "endpoint definition"],
         "primary_endpoint"),
    ]

    still_open: list[str] = []
    for item in high_qs:
        q_lc = item.question.lower()
        resolved = False
        for signals, attr_name in _SIGNALS:
            if any(s in q_lc for s in signals):
                if getattr(meta, attr_name, None) is not None:
                    resolved = True
                    break
        # Also check free-form additional_answers
        if not resolved and meta.additional_answers:
            for key in meta.additional_answers:
                if key.lower() in q_lc or q_lc in key.lower():
                    resolved = True
                    break
        if not resolved:
            still_open.append(item.question)

    return unknowns.model_copy(update={"unresolved_high_impact": still_open})


# ── Task 3: interactive closed-loop Q&A ───────────────────────────────────────

def _interactive_qa_loop(
    *,
    initial_unknowns: UnknownsOutput,
    initial_metadata: MetadataInput | None,
    initial_insights: InsightSynthesisOutput,
    initial_judge: Any,  # JudgeOutput | None
    preview: dict,
    evidence: dict,
    citations: list,
    plan: Any,
    collected: dict,
    provider: ProviderName,
    run_judge: bool,
    analysis_report: str | None,
    typed_tool_log: list[ToolCallRecord] | None,
    agent_runs: list[AgentRunRecord],
    max_iterations: int = 5,
) -> tuple[MetadataInput | None, InsightSynthesisOutput, Any, UnknownsOutput]:
    """
    Interactive closed-loop: present unresolved high-impact unknowns to the
    user one-by-one, collect answers, merge them into MetadataInput, re-run
    ClinicalSemanticsAgent → UnknownsAgent → InsightSynthesisAgent → Judge,
    and repeat until all critical unknowns are resolved or the user stops.

    Returns the final (metadata, insights, judge, unknowns) after the loop.
    """
    current_metadata = initial_metadata
    current_insights = initial_insights
    current_judge = initial_judge
    current_unknowns = initial_unknowns

    for iteration in range(1, max_iterations + 1):
        still_open = current_unknowns.unresolved_high_impact
        if not still_open:
            console.print("\n[green]✓ All critical unknowns resolved.[/green]")
            break

        console.print(
            f"\n[bold cyan]── Closed-loop iteration "
            f"{iteration}/{max_iterations} ──[/bold cyan]"
        )
        console.print(
            f"[yellow]{len(still_open)} high-impact unknown(s) remain. "
            "Provide answers below, or press Enter to skip a question.[/yellow]\n"
        )

        new_answers: dict[str, str] = {}
        for idx, question in enumerate(still_open, 1):
            answer = Prompt.ask(
                f"  [{idx}/{len(still_open)}] {question}\n  [dim]Your answer[/dim]",
                default="",
            )
            if answer.strip():
                new_answers[question] = answer.strip()

        if not new_answers:
            console.print("[dim]No answers provided — stopping loop.[/dim]")
            break

        # Merge new free-form answers into existing metadata
        existing_additional: dict[str, str] = (
            dict(current_metadata.additional_answers)
            if current_metadata and current_metadata.additional_answers
            else {}
        )
        existing_additional.update(new_answers)

        base = (
            current_metadata.model_dump(exclude_none=True)
            if current_metadata
            else {}
        )
        base["additional_answers"] = existing_additional
        current_metadata = MetadataInput.model_validate(base)

        # Enrich evidence with updated metadata
        meta_block = _render_metadata_block(current_metadata)
        enriched = dict(evidence)
        enriched["__user_metadata__"] = current_metadata.model_dump(exclude_none=True)
        enriched["__metadata_text__"] = meta_block
        loop_label = f"loop{iteration}"

        # Re-run ClinicalSemanticsAgent when semantics-related answers were given
        if _metadata_covers_semantics(current_metadata):
            cs_step = next(
                (s for s in plan.steps
                 if s.agent_to_call == "ClinicalSemanticsAgent"),
                plan.steps[0],
            )
            console.print(
                f"  [yellow]▸ Re-running ClinicalSemanticsAgent "
                f"({loop_label})…[/yellow]"
            )
            cs_refreshed = run_clinical_semantics_agent(
                preview, enriched, cs_step, provider,
                prior_analysis_report=analysis_report,
                tool_log=typed_tool_log,
            )
            collected["ClinicalSemanticsAgent"] = cs_refreshed.model_dump()
            agent_runs.append(
                AgentRunRecord(
                    step_number=len(agent_runs) + 1,
                    agent=f"ClinicalSemanticsAgent ({loop_label})",
                    goal=f"Re-run semantics with answers from {loop_label}",
                    output=cs_refreshed.model_dump(),
                )
            )
            console.print(
                f"    [green]✓ ClinicalSemanticsAgent ({loop_label})[/green]"
            )

        # Re-run UnknownsAgent to surface any newly-resolvable unknowns
        ua_step = next(
            (s for s in plan.steps if s.agent_to_call == "UnknownsAgent"),
            plan.steps[0],
        )
        console.print(
            f"  [yellow]▸ Re-running UnknownsAgent ({loop_label})…[/yellow]"
        )
        ua_refreshed = run_unknowns_agent(
            preview, enriched,
            collected.get("DataQualityAgent"),
            collected.get("ClinicalSemanticsAgent"),
            ua_step, provider,
            prior_analysis_report=analysis_report,
            tool_log=typed_tool_log,
        )
        collected["UnknownsAgent"] = ua_refreshed.model_dump()
        agent_runs.append(
            AgentRunRecord(
                step_number=len(agent_runs) + 1,
                agent=f"UnknownsAgent ({loop_label})",
                goal=f"Re-surface unknowns after {loop_label} answers",
                output=ua_refreshed.model_dump(),
            )
        )
        console.print(
            f"    [green]✓ UnknownsAgent ({loop_label})[/green]"
        )

        # Re-run InsightSynthesisAgent with updated context
        unknowns_enriched = dict(ua_refreshed.model_dump())
        unknowns_enriched["__metadata_text__"] = meta_block
        console.print(
            f"  [yellow]▸ Re-running InsightSynthesisAgent ({loop_label})…[/yellow]"
        )
        insights_refreshed = run_insight_synthesis_agent(
            preview, enriched,
            collected.get("DataQualityAgent"),
            collected.get("ClinicalSemanticsAgent"),
            provider,
            citations=citations,
            unknowns_output=unknowns_enriched,
            prior_analysis_report=analysis_report,
            tool_log=typed_tool_log,
        )
        collected["InsightSynthesisAgent"] = insights_refreshed.model_dump()
        agent_runs.append(
            AgentRunRecord(
                step_number=len(agent_runs) + 1,
                agent=f"InsightSynthesisAgent ({loop_label})",
                goal=f"Re-synthesise insights with metadata from {loop_label}",
                output=insights_refreshed.model_dump(),
            )
        )
        current_insights = insights_refreshed
        console.print(
            f"    [green]✓ InsightSynthesisAgent ({loop_label})[/green]"
        )

        # Re-run Judge on updated synthesis
        if run_judge:
            console.print(
                f"  [yellow]▸ Re-running JudgeAgent ({loop_label})…[/yellow]"
            )
            judge_refreshed = run_judge_agent(
                final_insights=current_insights,
                evidence=enriched,
                unknowns=ua_refreshed,
                provider=provider,
            )
            current_judge = judge_refreshed
            console.print(
                f"  [green]Judge ({loop_label}):[/green] "
                f"{judge_refreshed.overall_score}/100 | "
                f"Failed claims: {len(judge_refreshed.failed_claims)}"
            )

        # Re-tag unresolved unknowns with the updated metadata
        current_unknowns = _tag_unresolved_unknowns(ua_refreshed, current_metadata)

        remaining = current_unknowns.unresolved_high_impact
        if remaining:
            console.print(
                f"\n  [yellow]{len(remaining)} high-impact unknown(s) "
                "still open after this iteration.[/yellow]"
            )
        else:
            console.print(
                "\n  [green]✓ All high-impact unknowns resolved![/green]"
            )
            break

        # Ask user whether to continue before the next iteration
        if iteration < max_iterations:
            keep_going = Confirm.ask(
                "\n  Continue answering questions?",
                default=False,
            )
            if not keep_going:
                console.print("[dim]Loop stopped by user.[/dim]")
                break
    else:
        # max_iterations exhausted without full resolution
        console.print(
            f"[yellow]⚠ Reached max loop iterations ({max_iterations}). "
            "Remaining unknowns left as unresolved.[/yellow]"
        )

    return current_metadata, current_insights, current_judge, current_unknowns


def run_agentic_insights(
    df: pd.DataFrame,
    provider: ProviderName,
    max_rows: int = 25,
    max_cols: int = 30,
    run_judge: bool = True,
    metadata: MetadataInput | None = None,
    interactive: bool = False,
    # ── upstream statistical context (both default to None — backward compat) ──
    analysis_report: str | None = None,
    tool_log: list[dict[str, Any]] | None = None,
    emit: Callable | None = None,
    # ── new: clinical context ─────────────────────────────────────────────────
    study_context: str = "",
    # ── new: data dictionary (column name → description) ─────────────────────
    column_dict: dict[str, str] | None = None,
) -> FinalReportSchema:
    """
    Run the full agentic reasoning pipeline.

    Parameters
    ----------
    analysis_report : str | None
        Optional Markdown report produced by the upstream statistical
        AgentLoop.  Propagated as PRIOR_ANALYSIS_REPORT to all reasoning
        agents when agents.py is updated in Task 4A follow-on.
    tool_log : list[dict] | None
        Optional raw tool-call log from AgentLoop.  Each entry must have
        keys: tool (or tool_name), args, result, [error].  Coerced to
        typed ToolCallRecord objects with stable citation aliases.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Seed + reproducibility setup ─────────────────────────────────────────
    import os, numpy as np, hashlib, time as _time
    _seed = int(os.environ.get("ANALYSIS_SEED", "42"))
    np.random.seed(_seed)
    _run_id = f"{provider}_{int(_time.time())}"
    _repro = ReproducibilityLogBuilder(
        run_id=_run_id,
        study_context=study_context,
        seed=_seed,
    )

    def _emit(event_type: str, stage: str, message: str) -> None:
        """Forward a progress event to the caller without blocking on failure."""
        if emit is not None:
            try:
                emit({"type": event_type, "stage": stage, "message": message})
            except Exception:
                pass

    # Coerce raw tool_log dicts → typed ToolCallRecord list with aliases
    typed_tool_log: list[ToolCallRecord] | None = _coerce_tool_log(tool_log)

    # ── Step i: build preview + evidence ─────────────────────────────────────
    console.print("[bold cyan]► Step 1/5:[/bold cyan] Building dataset preview…")
    preview = build_dataset_preview(df, max_rows=max_rows, max_cols=max_cols)

    console.print("[bold cyan]► Step 2/5:[/bold cyan] Computing deterministic evidence…")
    evidence = build_dataset_evidence(df)
    citations = format_citations(evidence)
    # ── Step 2b: robustness guardrails (deterministic, no LLM) ───────────────
    _guardrail_raw = run_guardrails(df)
    guardrail_report = GuardrailReport.model_validate(_guardrail_raw)
    evidence["__guardrails__"] = _guardrail_raw   # agents see raw dict
    guardrail_citations = format_guardrail_citations(_guardrail_raw)
    if guardrail_citations:
        citations["guardrails"] = guardrail_citations

    _g_flags = _guardrail_raw["flags"]
    _g_hi = sum(1 for f in _g_flags if f["severity"] == "high")
    _g_rm = _guardrail_raw["repeated_measures"]
    _g_parts = []
    if _g_flags:
        _g_parts.append(
            f"{len(_g_flags)} flag(s) "
            f"([bold red]{_g_hi} high[/bold red] / "
            f"{len(_g_flags) - _g_hi} other)"
        )
    if _g_rm:
        _g_parts.append(
            "[yellow]repeated-measures schema detected[/yellow]"
            f" (id={_g_rm['id_column']}, "
            f"max_repeats={_g_rm['max_repeats_per_subject']})"
        )
    if _g_parts:
        console.print(
            "  [bold yellow]\u26a0 Guardrails:[/bold yellow] "
            + " | ".join(_g_parts)
        )
        for _gf in _g_flags:
            _icon = "[red]\u25cf[/red]" if _gf["severity"] == "high" else "[yellow]\u25cf[/yellow]"
            console.print(
                f"    {_icon} [{_gf['check_type']}] "
                f"{_gf.get('column', '')}\u2014"
                f"{_gf['detail'][:100]}"
            )
    else:
        console.print("  [green]\u2713 Guardrails: all checks passed[/green]")
    _emit("stage_complete", "dataset", "Dataset evidence + guardrails ready")
    # Attach metadata as a top-level evidence key so all agents see it
    if metadata is not None:
        evidence["__user_metadata__"] = metadata.model_dump(exclude_none=True)

    # Attach data dictionary so ClinicalSemanticsAgent can use authoritative definitions
    if column_dict:
        evidence["__column_dict__"] = column_dict

    # Attach upstream context markers so planner/evidence are aware
    if analysis_report is not None:
        evidence["__prior_analysis_available__"] = True
    if typed_tool_log is not None:
        evidence["__tool_log_n_calls__"] = len(typed_tool_log)
        evidence["__tool_log_tools__"] = [r.tool_name for r in typed_tool_log]

    # ── Step ii: planner — returns plan with synthesis guaranteed last ────────
    console.print("[bold cyan]► Step 3/5:[/bold cyan] Calling Planner (LLM)…")
    plan = call_planner(preview, evidence, provider)  # _ensure_synthesis_last inside
    _emit("stage_complete", "plan", f"Plan ready — {len(plan.steps)} steps")

    console.print(
        f"  [green]Plan:[/green] {len(plan.steps)} steps — {plan.dataset_summary}"
    )
    for s in plan.steps:
        console.print(f"    [dim]{s.step_number}. {s.name} → {s.agent_to_call}[/dim]")

    # ── Step iii: model metadata ──────────────────────────────────────────────
    _client = get_client(provider)
    model_name: str = getattr(_client, "model", str(provider))

    # ── Step iv: execute every step in plan order (synthesis is last) ─────────
    console.print("[bold cyan]► Step 4/5:[/bold cyan] Executing agent steps…")

    agent_runs: list[AgentRunRecord] = []
    collected: dict[str, dict] = {}

    for step in plan.steps:
        agent = step.agent_to_call
        console.print(
            f"  [yellow]▸ {step.step_number}/{len(plan.steps)}:[/yellow] "
            f"{step.name} [dim]({agent})[/dim]"
        )

        if agent == "DataQualityAgent":
            output_obj = run_data_quality_agent(
                preview, evidence, step, provider,
                prior_analysis_report=analysis_report,
                tool_log=typed_tool_log,
            )

        elif agent == "ClinicalSemanticsAgent":
            output_obj = run_clinical_semantics_agent(
                preview, evidence, step, provider,
                prior_analysis_report=analysis_report,
                tool_log=typed_tool_log,
            )

        elif agent == "UnknownsAgent":
            dq_out = collected.get("DataQualityAgent")
            cs_out = collected.get("ClinicalSemanticsAgent")
            output_obj = run_unknowns_agent(
                preview, evidence, dq_out, cs_out, step, provider,
                prior_analysis_report=analysis_report,
                tool_log=typed_tool_log,
            )

        elif agent == "InsightSynthesisAgent":
            # runs last in loop; collected already has DQ + CS + UA outputs
            dq_out = collected.get("DataQualityAgent")
            cs_out = collected.get("ClinicalSemanticsAgent")
            ua_out = collected.get("UnknownsAgent")
            output_obj = run_insight_synthesis_agent(
                preview, evidence, dq_out, cs_out, provider,
                citations=citations,
                unknowns_output=ua_out,
                prior_analysis_report=analysis_report,
                tool_log=typed_tool_log,
                study_context=study_context if study_context else None,
            )

        else:
            console.print(f"    [red]Unknown agent '{agent}' — skipping.[/red]")
            continue

        output_dict = output_obj.model_dump()
        collected[agent] = output_dict

        # step_number comes directly from the renumbered plan — no offset, no phantom steps
        agent_runs.append(
            AgentRunRecord(
                step_number=step.step_number,
                agent=agent,
                goal=step.goal,
                output=output_dict,
            )
        )
        console.print(f"    [green]✓ {agent} complete[/green]")
        _emit("stage_complete", agent, f"{agent} complete")

    # ── Step v: assemble report ───────────────────────────────────────────────
    console.print("[bold cyan]► Step 5/5:[/bold cyan] Assembling and saving report…")

    unknowns = UnknownsOutput.model_validate(
        collected["UnknownsAgent"]
    )
    # Tag high-impact unknowns that remain unanswered
    unknowns = _tag_unresolved_unknowns(unknowns, metadata)

    if unknowns.unresolved_high_impact:
        console.print(
            f"\n[bold yellow]⚠ {len(unknowns.unresolved_high_impact)} high-impact "
            f"unknown(s) still unresolved:[/bold yellow]"
        )
        for q in unknowns.unresolved_high_impact:
            console.print(f"  [yellow]? {q}[/yellow]")
        console.print(
            "[dim]  Tip: provide --metadata <file> to resolve these.  "
            "Synthesis certainty is reduced for unresolved items.[/dim]\n"
        )

    final_insights = InsightSynthesisOutput.model_validate(
        collected["InsightSynthesisAgent"]
    )

    # ── Step vi (optional): LLM-as-Judge ─────────────────────────────────────
    judge_output = None
    if run_judge:
        console.print("[bold cyan]► Step 6:[/bold cyan] LLM-as-Judge (pre-metadata)…")
        judge_output = run_judge_agent(
            final_insights=final_insights,
            evidence=evidence,
            unknowns=unknowns,
            provider=provider,
        )
        console.print(
            f"  [green]Judge score:[/green] {judge_output.overall_score}/100 "
            f"| Failed claims: {len(judge_output.failed_claims)}"
        )
        _emit("stage_complete", "judge", f"Judge score: {judge_output.overall_score}/100")

    # ── Step vii: closed-loop metadata re-runs ──────────────────────────────
    if metadata is not None:
        meta_block = _render_metadata_block(metadata)
        console.print(
            "\n[bold cyan]► Closed-loop:[/bold cyan] metadata provided — "
            "re-running affected agents…"
        )
        enriched = dict(evidence)
        enriched["__metadata_text__"] = meta_block

        # Re-run ClinicalSemanticsAgent when metadata clarifies column semantics
        if _metadata_covers_semantics(metadata):
            cs_step = next(
                (s for s in plan.steps
                 if s.agent_to_call == "ClinicalSemanticsAgent"),
                plan.steps[0],
            )
            console.print(
                "  [yellow]▸ Re-running ClinicalSemanticsAgent "
                "(semantics updated)…[/yellow]"
            )
            cs_refreshed = run_clinical_semantics_agent(
                preview, enriched, cs_step, provider,
                prior_analysis_report=analysis_report,
                tool_log=typed_tool_log,
            )
            collected["ClinicalSemanticsAgent"] = cs_refreshed.model_dump()
            console.print(
                "    [green]✓ ClinicalSemanticsAgent (updated)[/green]"
            )

        # Metadata re-run: only ClinicalSemanticsAgent is re-run here.
        # InsightSynthesisAgent runs once at the end with full context.
        unknowns_enriched = dict(unknowns.model_dump())
        unknowns_enriched["__metadata_text__"] = meta_block
        insights_after = None
        judge_after = None

    # ── Step viii: interactive closed-loop Q&A ────────────────────────────
    loop_metadata: MetadataInput | None = metadata
    loop_insights: InsightSynthesisOutput | None = None
    loop_judge: Any = None
    loop_unknowns: UnknownsOutput | None = None

    if interactive:
        # Start from the best state produced so far
        _start_insights = final_insights
        _start_judge = judge_output
        _start_meta = metadata  # may be None if no --metadata file was given

        console.print(
            "\n[bold cyan]► Interactive closed-loop Q&A[/bold cyan]"
        )
        loop_metadata, loop_insights, loop_judge, loop_unknowns = _interactive_qa_loop(
            initial_unknowns=unknowns,
            initial_metadata=_start_meta,
            initial_insights=_start_insights,
            initial_judge=_start_judge,
            preview=preview,
            evidence=evidence,
            citations=citations,
            plan=plan,
            collected=collected,
            provider=provider,
            run_judge=run_judge,
            analysis_report=analysis_report,
            typed_tool_log=typed_tool_log,
            agent_runs=agent_runs,
        )
        # Propagate loop results so they are used as canonical values below
        if loop_unknowns is not None:
            unknowns = loop_unknowns

    # ── Determine canonical values before final synthesis ─────────────────
    # InsightSynthesisAgent runs once at the end with full context.
    best_insights = final_insights
    best_judge = judge_output

    # ── Task 4B: deterministic reasoning engine ──────────────────────────
    console.print(
        "[bold cyan]► Reasoning Engine:[/bold cyan] "
        "Running deterministic validation…"
    )
    reasoning_state = run_reasoning_engine(
        run_id=f"{provider}_{model_name}",
        preview=preview,
        evidence=evidence,
        final_insights=best_insights.model_dump(),
        unknowns=unknowns.model_dump(),
        metadata=metadata,
        analysis_report=analysis_report,
        tool_log=typed_tool_log,
    )
    console.print(
        f"  [green]✓ Reasoning complete:[/green] "
        f"{len(reasoning_state.claims)} claims, "
        f"confidence={reasoning_state.confidence_summary.overall if reasoning_state.confidence_summary else 'n/a'}, "
        f"{len(reasoning_state.step_log)} steps logged"
    )
    _emit("stage_complete", "reasoning", f"Reasoning engine: {len(reasoning_state.claims)} claims validated")

    # ── Literature RAG: hypothesis-grounded paper retrieval ───────────────────
    literature_report: LiteratureRAGReport | None = None
    lit_block: str = ""

    if reasoning_state.hypotheses:
        try:
            console.print(
                "[bold cyan]\u25ba Literature RAG:[/bold cyan] "
                "Retrieving hypothesis-grounded papers\u2026"
            )
            hypo_dicts = [
                {"statement": h.statement, "hypothesis_id": h.hypothesis_id}
                for h in reasoning_state.hypotheses
            ]
            literature_report = run_literature_rag(
                hypotheses=hypo_dicts,
                preview=preview,
                evidence=evidence,
            )
            n_lit = len(literature_report.articles)
            if n_lit > 0:
                console.print(
                    f"  [green]\u2713 Literature:[/green] {n_lit} article(s) "
                    f"from {', '.join(literature_report.sources_used)}"
                )
                _emit("stage_complete", "literature", f"Literature: {n_lit} articles retrieved")
                for art in literature_report.articles:
                    console.print(
                        f"    [{art.citation_alias}] "
                        f"{art.title[:70]}{'...' if len(art.title) > 70 else ''} "
                        f"({art.year or 'n/d'})"
                    )
                lit_block = format_literature_for_agents(literature_report)
            else:
                console.print(
                    "  [yellow]\u26a0 Literature RAG: no results returned "
                    "(queries may be too specific or network unavailable)[/yellow]"
                )
        except Exception as exc:
            console.print(
                f"  [yellow]\u26a0 Literature RAG skipped: {exc}[/yellow]"
            )

    # ── Single final InsightSynthesisAgent call (full context) ───────────────
    # Combines: static analysis report + literature block + study_context
    parts = [p for p in [analysis_report, lit_block] if p]
    combined_context = "\n\n".join(parts) if parts else None

    console.print(
        "  [yellow]\u25b8 Running InsightSynthesisAgent "
        "(full context: analysis + literature)\u2026[/yellow]"
    )
    insights_final = run_insight_synthesis_agent(
        preview,
        evidence,
        collected.get("DataQualityAgent"),
        collected.get("ClinicalSemanticsAgent"),
        provider,
        citations=citations,
        unknowns_output=collected.get("UnknownsAgent"),
        prior_analysis_report=combined_context,
        tool_log=typed_tool_log,
        study_context=study_context if study_context else None,
    )
    collected["InsightSynthesisAgent"] = insights_final.model_dump()
    agent_runs.append(
        AgentRunRecord(
            step_number=len(agent_runs) + 1,
            agent="InsightSynthesisAgent (final)",
            goal="Synthesise all insights with full context: analysis + literature",
            output=insights_final.model_dump(),
        )
    )
    best_insights = insights_final
    console.print("    [green]\u2713 InsightSynthesisAgent (final)[/green]")
    _emit("stage_complete", "InsightSynthesisAgent", "Insight synthesis complete")
    # Use the accumulated metadata (may have been enriched by interactive loop)
    final_metadata = loop_metadata if interactive else metadata

    # ── NEW: CST translation → literature validation → synthesis scoring ──────
    grounded_findings: GroundedFindingsSchema | None = None
    synthesis_quality: Any = None

    if study_context:
        try:
            console.print(
                "[bold cyan]► CST Translation:[/bold cyan] "
                "Translating findings to clinical search terms…"
            )
            csts = translate_findings_to_cst(
                best_insights.key_findings, study_context, provider
            )
            _repro.add_csts(csts)
            _emit("stage_complete", "cst_translation", "Clinical search terms ready")

            console.print(
                "[bold cyan]► Literature Validation:[/bold cyan] "
                "Cross-referencing medical literature…"
            )
            lit_pipeline = LiteratureValidatorPipeline(provider=provider)
            grounded_list = lit_pipeline.validate(csts)
            _repro.add_literature_queries(lit_pipeline.query_records)
            grounded_findings = GroundedFindingsSchema(findings=grounded_list)
            _emit("stage_complete", "literature_validation", f"Literature: {len(grounded_list)} findings grounded")

            console.print(
                "[bold cyan]► Synthesis Scoring:[/bold cyan] "
                "Scoring synthesis quality…"
            )
            synthesis_quality = score_synthesis_quality(best_insights, study_context, provider)
            if synthesis_quality.score < SYNTHESIS_QUALITY_THRESHOLD:
                console.print(
                    f"  [yellow]⚠ Quality score {synthesis_quality.score:.2f} < "
                    f"threshold {SYNTHESIS_QUALITY_THRESHOLD:.2f} — re-running synthesis…[/yellow]"
                )
                best_insights = run_insight_synthesis_agent(
                    preview, evidence,
                    collected.get("DataQualityAgent"),
                    collected.get("ClinicalSemanticsAgent"),
                    provider,
                    citations=citations,
                    unknowns_output=collected.get("UnknownsAgent"),
                    prior_analysis_report=analysis_report,
                    tool_log=typed_tool_log,
                    study_context=study_context,
                )
                synthesis_quality.rerun_triggered = True
                synthesis_quality = score_synthesis_quality(best_insights, study_context, provider)
                synthesis_quality.rerun_triggered = True
            _emit("stage_complete", "synthesis_scoring", f"Synthesis quality: {synthesis_quality.score:.2f}")
        except Exception as exc:
            console.print(f"  [yellow]⚠ New pipeline steps skipped: {exc}[/yellow]")

    # ── Reproducibility log ───────────────────────────────────────────────────
    repro_log = _repro.finalise(synthesis_quality_score=synthesis_quality)

    grounded_findings = _annotate_confidence_warnings(
        grounded_findings=grounded_findings,
        evidence=evidence,
        tool_log=typed_tool_log,
    )

    report = FinalReportSchema(
        provider=provider,
        model=model_name,
        plan=plan,
        agent_runs=agent_runs,
        unknowns=unknowns,
        final_insights=best_insights,
        judge=best_judge,
        metadata_used=final_metadata,
        final_insights_before=None,
        final_insights_after=None,
        judge_before=None,
        judge_after=None,
        prior_analysis_report=analysis_report,
        tool_log=typed_tool_log,
        reasoning_state=reasoning_state,
        guardrail_report=guardrail_report,
        literature_report=literature_report,
        study_context=study_context or None,
        grounded_findings=grounded_findings,
        reproducibility_log=repro_log,
        synthesis_quality_score=synthesis_quality,
        treatment_columns_excluded=[],
    )

    # Persist compact tool_log (large results are truncated)
    report_dict = report.model_dump()
    if typed_tool_log is not None:
        report_dict["tool_log"] = _compact_tool_log_for_persistence(typed_tool_log)

    OUTPUT_FILE.write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    console.print(f"[bold green]✔ Saved →[/bold green] {OUTPUT_FILE}")

    return report
