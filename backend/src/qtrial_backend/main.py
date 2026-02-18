from __future__ import annotations

import typer
from rich.console import Console
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


if __name__ == "__main__":
    app()
