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
from qtrial_backend.agentic.planner import call_planner
from qtrial_backend.agentic.schemas import (
    AgentRunRecord,
    FinalReportSchema,
    InsightSynthesisOutput,
    UnknownsOutput,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import ProviderName
from qtrial_backend.dataset.evidence import build_dataset_evidence, format_citations
from qtrial_backend.dataset.preview import build_dataset_preview

console = Console()

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "agentic_run.json"


def run_agentic_insights(
    df: pd.DataFrame,
    provider: ProviderName,
    max_rows: int = 25,
    max_cols: int = 30,
) -> FinalReportSchema:

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Step i: build preview + evidence ─────────────────────────────────────
    console.print("[bold cyan]► Step 1/5:[/bold cyan] Building dataset preview…")
    preview = build_dataset_preview(df, max_rows=max_rows, max_cols=max_cols)

    console.print("[bold cyan]► Step 2/5:[/bold cyan] Computing deterministic evidence…")
    evidence = build_dataset_evidence(df)
    citations = format_citations(evidence)

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
            output_obj = run_data_quality_agent(preview, evidence, step, provider)

        elif agent == "ClinicalSemanticsAgent":
            output_obj = run_clinical_semantics_agent(preview, evidence, step, provider)

        elif agent == "UnknownsAgent":
            dq_out = collected.get("DataQualityAgent")
            cs_out = collected.get("ClinicalSemanticsAgent")
            output_obj = run_unknowns_agent(
                preview, evidence, dq_out, cs_out, step, provider
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

    final_insights = InsightSynthesisOutput.model_validate(
        collected["InsightSynthesisAgent"]
    )

    report = FinalReportSchema(
        provider=provider,
        model=model_name,
        plan=plan,
        agent_runs=agent_runs,
        unknowns=unknowns,
        final_insights=final_insights,
    )

    OUTPUT_FILE.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"[bold green]✔ Saved →[/bold green] {OUTPUT_FILE}")

    return report
