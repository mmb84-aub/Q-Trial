from __future__ import annotations

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
    provider: ProviderName = typer.Option("openai", "--provider", "-p", help="openai|gemini|claude"),
    max_rows: int = typer.Option(25, help="Rows to include in preview"),
    max_cols: int = typer.Option(30, help="Columns to include in preview"),
    mode: str = typer.Option("agentic", "--mode", "-m", help="agentic|single"),
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
        from qtrial_backend.agentic.orchestrator import run_agentic_insights

        console.print(
            Panel.fit(
                f"[bold]Mode:[/bold]     agentic\n"
                f"[bold]Provider:[/bold] {provider}\n"
                f"[bold]File:[/bold]     {file}",
                title="Q-Trial Agentic Pipeline",
            )
        )

        report = run_agentic_insights(df, provider, max_rows=max_rows, max_cols=max_cols)

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

        console.print(
            f"\n[dim]Provider: {report.provider} | Model: {report.model} | "
            f"Steps run: {len(report.agent_runs)}[/dim]"
        )
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


if __name__ == "__main__":
    app()
