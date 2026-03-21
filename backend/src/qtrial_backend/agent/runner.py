"""
Statistical Agent Loop runner.

Runs AgentLoop (LLM-driven iterative tool calling) on a dataset,
producing a Markdown analysis report and tool_log that feed directly
into run_agentic_insights() as `analysis_report` and `tool_log`.
"""
from __future__ import annotations

import json
from typing import Callable

import pandas as pd
from rich.console import Console

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.agent.loop import AgentLoop
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import ProviderName
from qtrial_backend.prompts.agent_system import (
    AGENT_SYSTEM_PROMPT,
    INITIAL_USER_MESSAGE_TEMPLATE,
)
from qtrial_backend.tools.registry import ToolRegistry

console = Console()


def _build_initial_message(
    df: pd.DataFrame,
    dataset_name: str,
    column_dict: dict[str, str] | None = None,
) -> str:
    """Format the standard initial user message for AgentLoop."""
    schema_lines = [f"  {col} ({dtype})" for col, dtype in df.dtypes.items()]
    schema = "\n".join(schema_lines)

    preview = json.dumps(
        json.loads(df.head(5).to_json(orient="records")),
        indent=2,
        ensure_ascii=False,
    )

    # Inject column dictionary as authoritative definitions so the agent
    # never asks about coding that is already documented (e.g. status=2).
    if column_dict:
        dict_lines = "\n".join(f"  {col}: {desc}" for col, desc in column_dict.items())
        column_descriptions = (
            "\nAUTHORITATIVE COLUMN DICTIONARY "
            "(treat these definitions as ground truth — do NOT ask about coding "
            "or meaning for any column listed here):\n"
            + dict_lines
            + "\n"
        )
    else:
        column_descriptions = ""

    return INITIAL_USER_MESSAGE_TEMPLATE.format(
        dataset_name=dataset_name,
        rows=len(df),
        cols=len(df.columns),
        schema=schema,
        column_descriptions=column_descriptions,
        preview_json=preview,
    )


def run_statistical_agent_loop(
    df: pd.DataFrame,
    provider: ProviderName,
    dataset_name: str,
    emit: Callable | None = None,
    model: str | None = None,
    column_dict: dict[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """
    Run the LLM-driven statistical AgentLoop on df.

    Returns
    -------
    analysis_report : str
        The Markdown report produced by the loop (text of AgentResponse).
    tool_log : list[dict]
        Raw tool-call log from every iteration.
    """
    # Ensure all tools are registered
    import qtrial_backend.tools  # noqa: F401

    ctx = AgentContext(dataframe=df, dataset_name=dataset_name)
    client = get_client(provider, model=model)
    tools = ToolRegistry.all_tools()

    console.print(
        f"[bold cyan]► Statistical Agent Loop:[/bold cyan] "
        f"{len(tools)} tool(s) available · provider=[bold]{provider}[/bold]"
    )

    if emit is not None:
        try:
            emit({
                "type": "progress",
                "stage": "StatisticalLoop",
                "message": f"Starting iterative statistical analysis ({len(tools)} tools)…",
            })
        except Exception:
            pass

    initial_message = _build_initial_message(df, dataset_name, column_dict)

    loop = AgentLoop(
        client=client,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        verbose=True,           # prints each tool call to terminal
    )

    response = loop.run(initial_message, ctx)

    console.print(
        f"  [bold green]✔ Agent Loop complete:[/bold green] "
        f"{response.iterations} iteration(s), "
        f"{response.tool_calls_made} tool call(s)"
    )

    if emit is not None:
        try:
            emit({
                "type": "stage_complete",
                "stage": "StatisticalLoop",
                "message": (
                    f"Statistical loop done — "
                    f"{response.iterations} iterations, "
                    f"{response.tool_calls_made} tool calls"
                ),
            })
        except Exception:
            pass

    return response.text, response.tool_log
