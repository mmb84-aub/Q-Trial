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
from typing import Any

from qtrial_backend.agentic.reasoning import run_reasoning_engine
from qtrial_backend.agentic.hypothesis_gen import (
    generate_dynamic_hypotheses,
    integrate_dynamic_hypotheses,
)
from qtrial_backend.agentic.schemas import (
    AgentRunRecord,
    FinalReportSchema,
    InsightSynthesisOutput,
    MetadataInput,
    ToolCallRecord,
    UnknownsOutput,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import ProviderName
from qtrial_backend.dataset.evidence import build_dataset_evidence, format_citations
from qtrial_backend.dataset.preview import build_dataset_preview

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


def run_agentic_insights(
    df: pd.DataFrame,
    provider: ProviderName,
    max_rows: int = 25,
    max_cols: int = 30,
    run_judge: bool = True,
    metadata: MetadataInput | None = None,
    # ── upstream statistical context (both default to None — backward compat) ──
    analysis_report: str | None = None,
    tool_log: list[dict[str, Any]] | None = None,
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

    # Coerce raw tool_log dicts → typed ToolCallRecord list with aliases
    typed_tool_log: list[ToolCallRecord] | None = _coerce_tool_log(tool_log)

    # ── Step i: build preview + evidence ─────────────────────────────────────
    console.print("[bold cyan]► Step 1/5:[/bold cyan] Building dataset preview…")
    preview = build_dataset_preview(df, max_rows=max_rows, max_cols=max_cols)

    console.print("[bold cyan]► Step 2/5:[/bold cyan] Computing deterministic evidence…")
    evidence = build_dataset_evidence(df)
    citations = format_citations(evidence)

    # Attach metadata as a top-level evidence key so all agents see it
    if metadata is not None:
        evidence["__user_metadata__"] = metadata.model_dump(exclude_none=True)

    # Attach upstream context markers so planner/evidence are aware
    if analysis_report is not None:
        evidence["__prior_analysis_available__"] = True
    if typed_tool_log is not None:
        evidence["__tool_log_n_calls__"] = len(typed_tool_log)
        evidence["__tool_log_tools__"] = [r.tool_name for r in typed_tool_log]

    # ── Step ii: planner — returns plan with synthesis guaranteed last ────────
    console.print("[bold cyan]► Step 3/5:[/bold cyan] Calling Planner (LLM)…")
    plan = call_planner(preview, evidence, provider)  # _ensure_synthesis_last inside

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

    # ── Step vii: closed-loop metadata re-runs ──────────────────────────────
    insights_after = None
    judge_after = None

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

        # Always re-run InsightSynthesisAgent with metadata context
        console.print(
            "  [yellow]▸ Re-running InsightSynthesisAgent "
            "with metadata…[/yellow]"
        )
        unknowns_enriched = dict(unknowns.model_dump())
        unknowns_enriched["__metadata_text__"] = meta_block

        insights_after = run_insight_synthesis_agent(
            preview,
            enriched,
            collected.get("DataQualityAgent"),
            collected.get("ClinicalSemanticsAgent"),
            provider,
            citations=citations,
            unknowns_output=unknowns_enriched,
            prior_analysis_report=analysis_report,
            tool_log=typed_tool_log,
        )
        console.print(
            "    [green]✓ InsightSynthesisAgent (updated)[/green]"
        )

        # Re-run judge on updated synthesis
        if run_judge:
            console.print(
                "  [yellow]▸ Re-running JudgeAgent (post-metadata)…[/yellow]"
            )
            judge_after = run_judge_agent(
                final_insights=insights_after,
                evidence=enriched,
                unknowns=unknowns,
                provider=provider,
            )
            console.print(
                f"  [green]Judge (after):[/green] "
                f"{judge_after.overall_score}/100 | "
                f"Failed claims: {len(judge_after.failed_claims)}"
            )

    # ── Determine canonical values (latest available) ─────────────────────
    best_insights = insights_after if insights_after is not None else final_insights
    best_judge = judge_after if judge_after is not None else judge_output

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

    # ── Task 4C: dynamic hypothesis generation (LLM) ────────────────────
    try:
        console.print(
            "[bold cyan]► Hypothesis Generator:[/bold cyan] "
            "Generating dynamic hypotheses…"
        )
        hypo_output = generate_dynamic_hypotheses(
            provider=provider,
            preview=preview,
            evidence=evidence,
            final_insights=best_insights.model_dump(),
            unknowns=unknowns.model_dump(),
            valid_citation_keys=reasoning_state.valid_citation_keys,
            judge_output=best_judge.model_dump() if best_judge else None,
            metadata=metadata,
            tool_log=typed_tool_log,
        )
        reasoning_state = integrate_dynamic_hypotheses(
            state=reasoning_state,
            llm_output=hypo_output,
            metadata=metadata,
            unresolved_high_impact=unknowns.unresolved_high_impact,
        )
        console.print(
            f"  [green]✓ Hypotheses:[/green] "
            f"{len(reasoning_state.hypotheses)} hypotheses, "
            f"{len(reasoning_state.hidden_questions)} hidden questions, "
            f"{len(reasoning_state.step_log)} total steps"
        )
    except Exception as exc:
        console.print(
            f"  [yellow]⚠ Dynamic hypothesis generation skipped: {exc}[/yellow]"
        )

    report = FinalReportSchema(
        provider=provider,
        model=model_name,
        plan=plan,
        agent_runs=agent_runs,
        unknowns=unknowns,
        final_insights=best_insights,
        judge=best_judge,
        metadata_used=metadata,
        final_insights_before=final_insights if metadata is not None else None,
        final_insights_after=insights_after,
        judge_before=judge_output if metadata is not None else None,
        judge_after=judge_after,
        prior_analysis_report=analysis_report,
        tool_log=typed_tool_log,
        reasoning_state=reasoning_state,
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
