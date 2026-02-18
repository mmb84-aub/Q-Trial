# Q-Trial Backend

Minimal Python backend for generating AI-powered insights from clinical trial datasets.

This backend:
* Loads a CSV or Excel dataset
* Builds a structured preview (schema, head rows, missingness, numeric summary)
* Sends the preview to an LLM provider
* Returns structured clinical/data-quality insights

### Currently supported providers:
* OpenAI (ChatGPT)
* Google Gemini
* Anthropic Claude

> This is the foundational layer of the Q-Trial system.

---

## What It Does (Current Scope)

### Pipeline
1. **Dataset** (CSV/XLSX)
2. **Pandas DataFrame**
3. **Dataset Preview Builder**
4. **LLM Provider** (OpenAI | Gemini | Claude)
5. **Actionable Clinical + Data Insights**

**Important:**
* The full dataset is **NOT** sent to the model.
* Only a structured preview is transmitted to control context size and cost.

---

## Requirements
* Python 3.12+
* Poetry
* API key for at least one provider

### Install Poetry (Codespaces / Linux / Mac)
```bash
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
export PATH="$HOME/.local/bin:$PATH"
```
Verify:

```bash
poetry --version
```
Setup Project
From inside backend/:

```bash
poetry config virtualenvs.in-project true
poetry install
```
This creates: backend/.venv/

Note: If VS Code does not detect dependencies, select the interpreter manually at:
backend/.venv/bin/python

Environment Variables
Create a .env file inside backend/. Only set the provider(s) you intend to use.

Example:

```code
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=your_anthropic_key
CLAUDE_MODEL=claude-opus-4-6
```

### Project Structure
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

### Architecture Breakdown:
dataset/ → ingestion + preview generation

providers/ → provider-specific implementations

core/ → routing + shared types

prompts/ → prompt definitions

main.py → CLI interface

Running the CLI
Place your dataset inside backend/data/ (e.g., backend/data/pbc.csv).

Run the command for your preferred provider:

# OpenAI
```python
poetry run qtrial --file data/pbc.csv --provider openai
```
# Gemini
```python
poetry run qtrial --file data/pbc.csv --provider gemini
```
# Claude
```python
poetry run qtrial --file data/pbc.csv --provider claude
```
Supported File Types: .csv, .xlsx

### Technical Details
What the Model Receives
The model receives a payload containing:

Dataset shape

Column schema

First N rows

Missingness percentages

Numeric summary statistics

This prevents oversized context, privacy leakage, and uncontrolled token costs.

### Current Limitations

No RAG grounding yet

No evaluation/judge system yet

No API server (CLI only)

#### CLI vs API
This backend is currently CLI-based. It is intentionally built this way for:

Rapid experimentation

Reproducibility

Provider benchmarking

Research workflow

An API wrapper (FastAPI) can be added later without modifying core logic.

### Next Planned Improvements
Structured JSON output mode

Save outputs to file

Provider comparison mode

RAG integration

Evaluation suite

FastAPI wrapper

### Development Notes
This focuses on a clean modular architecture, provider abstraction, and controlled prompt inputs for research extensibility.
