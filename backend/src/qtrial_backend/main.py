from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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
):
    """
    Load a dataset, build a small preview payload, send to selected LLM provider, print insights.
    """
    df = load_dataset(file)
    payload = build_dataset_preview(df, max_rows=max_rows, max_cols=max_cols)

    req = LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        payload=payload,
    )

    client = get_client(provider)
    resp = client.generate(req)

    console.print(Panel.fit(f"[bold]Provider:[/bold] {resp.provider}\n[bold]Model:[/bold] {resp.model}"))
    console.print(resp.text)


@app.command()
def analyze(
    file: str = typer.Option(
        ..., "--file", "-f", help="Path to CSV/XLSX dataset"
    ),
    provider: ProviderName = typer.Option(
        "openai", "--provider", "-p", help="openai|gemini|claude"
    ),
    max_iterations: int = typer.Option(
        25, "--max-iterations", "-i", help="Maximum agent loop iterations"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show tool call details"
    ),
):
    """
    Agentic analysis: LLM iteratively explores the dataset using statistical
    tools, searches biomedical literature, and produces a comprehensive report.
    """
    from qtrial_backend.agent.context import AgentContext
    from qtrial_backend.agent.loop import AgentLoop
    from qtrial_backend.prompts.agent_system import (
        AGENT_SYSTEM_PROMPT,
        INITIAL_USER_MESSAGE_TEMPLATE,
    )
    from qtrial_backend.tools.registry import ToolRegistry

    # Ensure tools are registered (import triggers @tool decorators)
    import qtrial_backend.tools  # noqa: F401

    # 1. Load dataset
    df = load_dataset(file)
    dataset_name = Path(file).name
    context = AgentContext(dataframe=df, dataset_name=dataset_name)

    # 2. Build initial message with schema and brief preview
    preview = build_dataset_preview(df, max_rows=5, max_cols=30)
    schema = {c: str(df[c].dtype) for c in df.columns}
    initial_msg = INITIAL_USER_MESSAGE_TEMPLATE.format(
        dataset_name=dataset_name,
        rows=context.shape[0],
        cols=context.shape[1],
        schema=json.dumps(schema, indent=2),
        preview_json=json.dumps(preview["head"][:5], indent=2, default=str),
    )

    # 3. Create client and run agent loop
    client = get_client(provider)
    all_tools = ToolRegistry.all_tools()

    console.print(
        Panel.fit(
            f"[bold]Dataset:[/bold] {dataset_name}\n"
            f"[bold]Shape:[/bold] {context.shape[0]} rows x {context.shape[1]} cols\n"
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Tools available:[/bold] {len(all_tools)}\n"
            f"[bold]Max iterations:[/bold] {max_iterations}",
            title="Agent Configuration",
        )
    )

    agent = AgentLoop(
        client=client,
        tools=all_tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    result = agent.run(initial_msg, context)

    # 4. Display results
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Provider:[/bold] {result.provider}\n"
            f"[bold]Model:[/bold] {result.model}\n"
            f"[bold]Iterations:[/bold] {result.iterations}\n"
            f"[bold]Tool calls:[/bold] {result.tool_calls_made}",
            title="Agent Summary",
        )
    )
    console.print()
    console.print(Markdown(result.text))

    if verbose and result.tool_log:
        console.print()
        log_lines = "\n".join(
            f"  {i + 1}. {e['tool']}({e['args']}) "
            f"{'ERROR' if e['is_error'] else 'OK'}"
            for i, e in enumerate(result.tool_log)
        )
        console.print(Panel(log_lines, title="Tool Call Log"))


@app.command()
def report(
    file: str = typer.Option(
        ..., "--file", "-f", help="Path to CSV/XLSX dataset"
    ),
    mode: str = typer.Option(
        "static",
        "--mode", "-m",
        help="Report mode: 'static' (deterministic, no LLM) or 'dynamic' (agentic LLM-driven)",
    ),
    provider: ProviderName = typer.Option(
        "openai", "--provider", "-p", help="LLM provider for dynamic mode: openai|gemini|claude"
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Optional path to save the report as a .md file"
    ),
    max_iterations: int = typer.Option(
        25, "--max-iterations", "-i", help="Max agent iterations (dynamic mode only)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show tool call details (dynamic mode only)"
    ),
):
    """
    Generate a report from a dataset.

    Static mode  (--mode static, default):
        Runs a fixed deterministic pipeline of statistical tools and produces a
        Markdown report. No LLM or API key required.

    Dynamic mode (--mode dynamic):
        Runs the full agentic loop — the LLM iteratively calls tools, performs
        targeted statistical tests, searches literature, and writes a narrative
        report. Requires --provider and a valid API key.
    """
    df = load_dataset(file)
    dataset_name = Path(file).name

    # ── Static report ─────────────────────────────────────────────────────────
    if mode == "static":
        from qtrial_backend.report.static import build_static_report

        console.print(
            Panel.fit(
                f"[bold]Dataset:[/bold] {dataset_name}\n"
                f"[bold]Shape:[/bold] {df.shape[0]} rows × {df.shape[1]} cols\n"
                f"[bold]Mode:[/bold] static (no LLM)",
                title="Static Report",
            )
        )

        report_md = build_static_report(df, dataset_name)
        console.print(Markdown(report_md))

        if output:
            Path(output).write_text(report_md, encoding="utf-8")
            console.print(f"\n[green]Report saved to:[/green] {output}")

    # ── Dynamic report ────────────────────────────────────────────────────────
    elif mode == "dynamic":
        from qtrial_backend.agent.context import AgentContext
        from qtrial_backend.agent.loop import AgentLoop
        from qtrial_backend.prompts.agent_system import (
            AGENT_SYSTEM_PROMPT,
            INITIAL_USER_MESSAGE_TEMPLATE,
        )
        from qtrial_backend.tools.registry import ToolRegistry
        import qtrial_backend.tools  # noqa: F401

        context = AgentContext(dataframe=df, dataset_name=dataset_name)
        preview = build_dataset_preview(df, max_rows=5, max_cols=30)
        schema = {c: str(df[c].dtype) for c in df.columns}
        initial_msg = INITIAL_USER_MESSAGE_TEMPLATE.format(
            dataset_name=dataset_name,
            rows=context.shape[0],
            cols=context.shape[1],
            schema=json.dumps(schema, indent=2),
            preview_json=json.dumps(preview["head"][:5], indent=2, default=str),
        )

        client = get_client(provider)
        all_tools = ToolRegistry.all_tools()

        console.print(
            Panel.fit(
                f"[bold]Dataset:[/bold] {dataset_name}\n"
                f"[bold]Shape:[/bold] {context.shape[0]} rows × {context.shape[1]} cols\n"
                f"[bold]Provider:[/bold] {provider}\n"
                f"[bold]Tools:[/bold] {len(all_tools)}\n"
                f"[bold]Max iterations:[/bold] {max_iterations}",
                title="Dynamic Report",
            )
        )

        agent = AgentLoop(
            client=client,
            tools=all_tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        result = agent.run(initial_msg, context)

        console.print()
        console.print(
            Panel.fit(
                f"[bold]Provider:[/bold] {result.provider}\n"
                f"[bold]Model:[/bold] {result.model}\n"
                f"[bold]Iterations:[/bold] {result.iterations}\n"
                f"[bold]Tool calls:[/bold] {result.tool_calls_made}",
                title="Agent Summary",
            )
        )
        console.print()
        console.print(Markdown(result.text))

        if output:
            Path(output).write_text(result.text, encoding="utf-8")
            console.print(f"\n[green]Report saved to:[/green] {output}")

        if verbose and result.tool_log:
            console.print()
            log_lines = "\n".join(
                f"  {i + 1}. {e['tool']}({e['args']}) "
                f"{'ERROR' if e['is_error'] else 'OK'}"
                for i, e in enumerate(result.tool_log)
            )
            console.print(Panel(log_lines, title="Tool Call Log"))

    else:
        console.print(f"[red]Unknown mode '{mode}'. Use 'static' or 'dynamic'.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
