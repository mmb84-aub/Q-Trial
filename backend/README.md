# Q-Trial Backend

Minimal Python backend for generating AI-powered insights from clinical trial datasets.

This backend:

- Loads a CSV or Excel dataset
- Builds a structured preview (schema, head rows, missingness, numeric summary)
- Sends the preview to an LLM provider
- Returns actionable clinical and data-quality insights

Supported providers:

- OpenAI (ChatGPT)
- Google Gemini
- Anthropic Claude

This is the foundational execution layer of the Q-Trial system.

---

## What It Does (Current Scope)

Pipeline:

```text
Dataset (CSV/XLSX)
        ↓
Pandas DataFrame
        ↓
Dataset Preview Builder
        ↓
LLM Provider (OpenAI | Gemini | Claude)
        ↓
Actionable Clinical + Data Insights

Important:

    The full dataset is NOT sent to the model.

    Only a structured preview is transmitted to control context size and cost.

Requirements

    Python 3.12+

    Poetry

    An API key for at least one provider

Install Poetry (Codespaces / Linux / macOS)

curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

Verify:

poetry --version

Setup

From inside backend/:

poetry config virtualenvs.in-project true
poetry install

This creates:

backend/.venv/

VS Code / Pylance (Imports not resolved)

If VS Code shows missing imports, select the interpreter:

    backend/.venv/bin/python

Then reload the window.
Environment Variables

Create a .env file inside backend/.

Example:

OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=your_anthropic_key
CLAUDE_MODEL=claude-opus-4-6

Only set the provider(s) you intend to use.
Project Layout

backend/
  data/
  src/qtrial_backend/
    config.py
    main.py
    dataset/
      load.py
      preview.py
    providers/
      openai_client.py
      gemini_client.py
      claude_client.py
    prompts/
      insights.py
    core/
      types.py
      router.py
  pyproject.toml
  README.md

Where to Put Datasets

Create a folder (recommended):

mkdir -p data

Place files in:

backend/data/

Supported formats:

    .csv

    .xlsx / .xls

Example:

backend/data/pbc.csv

Run the CLI

From inside backend/:

poetry run qtrial --file data/pbc.csv --provider openai

Other providers:

poetry run qtrial --file data/pbc.csv --provider gemini
poetry run qtrial --file data/pbc.csv --provider claude

You should see:

Provider: <provider>
Model: <model>
<insights text>

What the Model Receives

The model receives a dataset preview payload containing:

    Dataset shape

    Column schema

    First N rows (default: 25)

    Missingness percentages

    Numeric summary statistics

This reduces:

    Token cost

    Context overflow

    Leakage of full dataset contents

Current Limitations

    Output is free text (no structured JSON yet)

    No RAG grounding yet

    No evaluation/judge system yet

    CLI-only (no API server yet)

CLI vs API Notes

This backend is intentionally CLI-first for:

    Fast iteration

    Reproducibility

    Easy provider benchmarking

An API wrapper (e.g., FastAPI) can be added later without changing core logic.