from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName
from qtrial_backend.dataset.load import load_dataset
from qtrial_backend.dataset.preview import build_dataset_preview
from qtrial_backend.prompts.insights import SYSTEM_PROMPT, USER_PROMPT

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def insights(
    file: str = typer.Option(..., "--file", "-f", help="Path to CSV/XLSX dataset"),
    provider: ProviderName = typer.Option("openai", "--provider", "-p", help="openai|gemini|claude|openrouter"),
    max_rows: int = typer.Option(25, help="Rows to include in preview"),
    max_cols: int = typer.Option(30, help="Columns to include in preview"),
    mode: str = typer.Option("agentic", "--mode", "-m", help="agentic|single"),
    judge: bool = typer.Option(
        True,
        "--judge/--no-judge",
        help="Run LLM-as-Judge validation after synthesis (agentic mode only; default: on)",
    ),
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        help=(
            "Path to a JSON file with metadata answers (status_mapping, "
            "time_unit, lab_units, study_design, treatment_arms, "
            "additional_answers).  Triggers closed-loop re-synthesis.  "
            "See data/metadata_template.json for an example."
        ),
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive/--no-interactive",
        help=(
            "Enable interactive closed-loop Q&A (agentic mode only). "
            "After the initial run the pipeline will ask you to answer "
            "each high-impact unknown in the terminal, then re-run "
            "ClinicalSemanticsAgent, UnknownsAgent, InsightSynthesisAgent, "
            "and the Judge until all critical unknowns are resolved or you "
            "choose to stop.  Compatible with --metadata (file answers are "
            "applied first)."
        ),
    ),
):
    """
    Load a dataset, run the selected pipeline mode, print insights.

    Modes:
      agentic  — multi-step planner + specialist agents (default)
      single   — original single-shot LLM call
    """
    df = load_dataset(file)

    # ── AGENTIC MODE ──────────────────────────────────────────────────────────
    if mode == "agentic":
        from qtrial_backend.agentic.orchestrator import run_pipeline
        from qtrial_backend.agentic.schemas import MetadataInput

        # Load and validate optional metadata file
        meta_obj: MetadataInput | None = None
        if metadata is not None:
            mpath = Path(metadata)
            if not mpath.exists():
                console.print(f"[red]Metadata file not found:[/red] {metadata}")
                raise typer.Exit(code=1)
            try:
                raw = json.loads(mpath.read_text(encoding="utf-8"))
                # Strip underscore-prefixed comment keys
                raw = {k: v for k, v in raw.items() if not k.startswith("_")}
                meta_obj = MetadataInput.model_validate(raw)
                console.print(f"[green]✓ Metadata loaded:[/green] {mpath.name}")
            except Exception as exc:
                console.print(f"[red]Failed to parse metadata:[/red] {exc}")
                raise typer.Exit(code=1)

        console.print(
            Panel.fit(
                f"[bold]Mode:[/bold]     agentic\n"
                f"[bold]Provider:[/bold] {provider}\n"
                f"[bold]File:[/bold]     {file}\n"
                f"[bold]Metadata:[/bold] {metadata or '(none)'}\n"
                f"[bold]Interactive:[/bold] {'yes' if interactive else 'no'}",
                title="Q-Trial Agentic Pipeline",
            )
        )

        # Generate deterministic static report first
        from qtrial_backend.report.static import build_static_report
        from qtrial_backend.agent.runner import run_statistical_agent_loop

        dataset_name = Path(file).stem
        console.print("[bold cyan]► Static Analysis:[/bold cyan] Running deterministic statistical report…")
        try:
            static_report = build_static_report(df, dataset_name)
            console.print(f"  [green]✓ Static report ready[/green] ({len(static_report)} chars)")
        except Exception as exc:
            console.print(f"  [yellow]⚠ Static report skipped: {exc}[/yellow]")
            static_report = None

        # Run LLM-driven statistical agent loop
        try:
            loop_report, tool_log = run_statistical_agent_loop(df, provider, dataset_name)
        except Exception as exc:
            console.print(f"  [yellow]⚠ Statistical agent loop skipped: {exc}[/yellow]")
            loop_report, tool_log = None, None

        # Combine both into analysis_report for the reasoning pipeline
        parts = [p for p in [static_report, loop_report] if p]
        analysis_report = "\n\n---\n\n".join(parts) if parts else None

        report = run_pipeline(
            df, provider,
            max_rows=max_rows, max_cols=max_cols,
            run_judge=judge,
            metadata=meta_obj,
            interactive=interactive,
            analysis_report=analysis_report,
            tool_log=tool_log,
        )

        fi = report.final_insights
        ua = report.unknowns

        console.print(Rule("[bold]Final Insights Report[/bold]"))

        console.print("\n[bold cyan]Key Findings[/bold cyan]")
        for item in fi.key_findings:
            console.print(f"  • {item}")

        console.print("\n[bold red]Risks & Bias Signals[/bold red]")
        for item in fi.risks_and_bias_signals:
            console.print(f"  • {item}")

        console.print("\n[bold green]Recommended Next Analyses[/bold green]")
        for rec in sorted(fi.recommended_next_analyses, key=lambda r: r.rank):
            console.print(
                f"  [{rec.rank}] [bold]{rec.analysis}[/bold]\n"
                f"       Rationale: {rec.rationale}\n"
                f"       Evidence:  {rec.evidence_citation}"
            )

        console.print("\n[bold yellow]Required Metadata / Open Questions[/bold yellow]")
        for q in fi.required_metadata_or_questions:
            console.print(f"  • {q}")

        # ── Unknowns section ──────────────────────────────────────────────────
        console.print(Rule("[bold]Unknowns & Assumptions Report[/bold]"))

        console.print("\n[bold magenta]Ranked Unknowns[/bold magenta]")
        for u in sorted(ua.ranked_unknowns, key=lambda x: x.rank):
            console.print(
                f"  [{u.rank}] [bold]{u.question}[/bold]\n"
                f"       Category: {u.category} | Impact: {u.impact}\n"
                f"       Rationale: {u.rationale}"
            )

        console.print("\n[bold blue]Explicit Assumptions[/bold blue]")
        for a in ua.explicit_assumptions:
            console.print(
                f"  • [bold]{a.assumption}[/bold]\n"
                f"    Basis: {a.basis}\n"
                f"    Risk if wrong: {a.risk_if_wrong}"
            )

        console.print("\n[bold]Required Documents[/bold]")
        for d in ua.required_documents:
            console.print(
                f"  • [{d.priority.upper()}] {d.document}\n"
                f"    Reason: {d.reason}"
            )

        console.print(f"\n[dim italic]{ua.summary}[/dim italic]")

        # ── Top Questions / resolution status ──────────────────────────────
        _print_questions_panel(ua, meta_obj)

        # ── Before/after update summary when metadata triggered re-run ─────
        if meta_obj is not None and report.final_insights_before is not None:
            _print_update_summary(report)

        console.print(
            f"\n[dim]Provider: {report.provider} | Model: {report.model} | "
            f"Steps run: {len(report.agent_runs)}[/dim]"
        )

        # ── Judge Summary ─────────────────────────────────────────────────────
        if report.judge is not None:
            _print_judge_summary(report.judge)
        return

    # ── SINGLE MODE (unchanged original behaviour) ────────────────────────────
    if mode == "single":
        payload = build_dataset_preview(df, max_rows=max_rows, max_cols=max_cols)
        req = LLMRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            payload=payload,
        )
        client = get_client(provider)
        resp = client.generate(req)
        console.print(
            Panel.fit(
                f"[bold]Provider:[/bold] {resp.provider}\n"
                f"[bold]Model:[/bold]    {resp.model}"
            )
        )
        console.print(resp.text)
        return

    console.print(f"[red]Unknown mode '{mode}'. Use 'agentic' or 'single'.[/red]")
    raise typer.Exit(code=1)


# ── Terminal print helpers ────────────────────────────────────────────────────

def _print_questions_panel(ua, meta_obj) -> None:
    """
    Show a yellow panel with the top high-impact unknowns when no metadata is
    supplied, or show a status panel when metadata partially resolves them.
    """
    high = [u for u in ua.ranked_unknowns if u.impact == "high"]
    if meta_obj is None:
        if high:
            lines: list[str] = []
            for idx, u in enumerate(high[:5], 1):
                lines.append(
                    f"  {idx}. {u.question}\n"
                    f"     [dim]({u.category})[/dim]"
                )
            console.print()
            console.print(
                Panel(
                    "\n".join(lines),
                    title=(
                        "[bold yellow]Top Questions to Answer[/bold yellow]  "
                        "[dim]— supply --metadata to resolve[/dim]"
                    ),
                    border_style="yellow",
                )
            )
    elif ua.unresolved_high_impact:
        lines = [f"  ? {q}" for q in ua.unresolved_high_impact[:5]]
        console.print()
        console.print(
            Panel(
                "\n".join(lines),
                title="[bold yellow]Still Unresolved High-Impact Unknowns[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            "\n[green]All high-impact unknowns resolved by metadata.[/green]"
        )


def _print_update_summary(report) -> None:
    """Before/after comparison panel when metadata triggered re-synthesis."""
    before = report.final_insights_before
    after = report.final_insights_after
    if before is None or after is None:
        return

    jb = report.judge_before
    ja = report.judge_after
    score_line = ""
    if jb is not None and ja is not None:
        delta = ja.overall_score - jb.overall_score
        sign = "+" if delta >= 0 else ""
        clr = "green" if delta >= 0 else "red"
        score_line = (
            f"\n  Judge score:      {jb.overall_score} \u2192 {ja.overall_score} "
            f"([{clr}]{sign}{delta}[/{clr}])"
        )

    body = (
        f"  Key findings:     {len(before.key_findings)} \u2192 "
        f"{len(after.key_findings)}\n"
        f"  Risk signals:     {len(before.risks_and_bias_signals)} \u2192 "
        f"{len(after.risks_and_bias_signals)}"
        + score_line
    )

    console.print()
    console.print(
        Panel(
            body,
            title="[bold green]Updated After Metadata[/bold green]",
            border_style="green",
        )
    )


# \u2500\u2500 Judge Summary helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _print_judge_summary(judge: "JudgeOutput") -> None:  # noqa: F821
    """Render a compact Judge Summary panel to the terminal."""
    from rich.panel import Panel as _Panel

    score = judge.overall_score
    rubric = judge.rubric
    failed = judge.failed_claims
    reasoning = judge.judge_reasoning

    rubric_lines = [
        f"  Evidence support:      {rubric.evidence_support:>3}/100",
        f"  Clinical overreach:    {rubric.clinical_overreach:>3}/100",
        f"  Uncertainty handling:  {rubric.uncertainty_handling:>3}/100",
        f"  Internal consistency:  {rubric.internal_consistency:>3}/100",
    ]

    fail_lines: list[str] = []
    high = [f for f in failed if f.severity == "high"]
    shown = (high + [f for f in failed if f not in high])[:3]
    for fc in shown:
        sev = fc.severity.upper()
        claim = fc.claim_text[:80]
        reason = fc.reason[:100]
        fail_lines.append(f"  [{sev}] \"{claim}\"")
        fail_lines.append(f"         → {reason}")

    body_parts = [
        f"[bold]Overall Score:[/bold] {score}/100",
        "",
        "[bold]Rubric:[/bold]",
        *rubric_lines,
    ]
    if reasoning:
        body_parts += ["", f"[bold]Assessment:[/bold] {reasoning[:300]}"]
    if fail_lines:
        body_parts += ["", f"[bold]Top Failures ({len(failed)} total):[/bold]"]
        body_parts += fail_lines

    console.print()
    console.print(
        _Panel(
            "\n".join(body_parts),
            title="[bold yellow]Judge Summary[/bold yellow]",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    app()
