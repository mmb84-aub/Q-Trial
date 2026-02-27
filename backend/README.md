# Q-Trial Backend

Python backend for generating AI-powered, evidence-grounded analysis reports from clinical trial datasets.

This backend:
* Loads a CSV or Excel dataset
* Builds a structured preview (schema, head rows, missingness, numeric summary)
* Runs an **agentic analysis loop** — the LLM iteratively calls 30+ statistical and literature tools to explore the data
* Produces a comprehensive, reproducible clinical trial report with effect sizes, literature citations, and recommendations

### Supported Providers
* **OpenAI** (GPT-4o, etc.)
* **Google Gemini**
* **Anthropic Claude**
* **OpenRouter** — access any model (OpenAI, Anthropic, Meta, Mistral, etc.) through a single API key

> This is the foundational layer of the Q-Trial system.

---

## What It Does

### Two Modes

#### 1. Quick Insights (`qtrial insights`)
Single-shot LLM call that returns high-level observations about the dataset.

#### 2. Agentic Analysis (`qtrial analyze`)
A multi-iteration agent loop where the LLM:
1. Inspects the dataset schema, preview rows, and optional **data dictionary**
2. Iteratively calls statistical tools (normality tests, hypothesis tests, survival analysis, regression, etc.)
3. Searches biomedical literature (PubMed, Semantic Scholar) and builds an evidence table
4. Produces a structured report with 8 sections (see Report Format below)

### Pipeline
1. **Dataset** (CSV/XLSX)
2. **Pandas DataFrame**
3. **Dataset Preview Builder** (schema, head, missingness, numeric stats)
4. **Optional Data Dictionary** (column → plain-English description)
5. **Agent Loop** (LLM ↔ 30+ tools, up to 25 iterations)
6. **Structured Clinical Report**

**Important:**
* The full dataset is **NOT** sent to the model.
* Only a structured preview is transmitted to control context size and cost.
* Tools operate on the full dataset server-side and return summarised results.

---

## Report Format

The agent produces a report with exactly these sections:

1. **Dataset Overview** — structure, size, variable types, study design inference
2. **Data Quality Assessment** — missingness, duplicates, type issues, outliers
3. **Baseline Characteristics** — Table 1 with SMD (for RCTs), randomisation quality
4. **Key Statistical Findings** — effect sizes, CIs, corrected p-values
5. **Survival Analysis** (if applicable) — KM curves, median survival, Cox HR
6. **Feature Relations & Derived Features** — inter-feature relationships, proposed new features, underlying patterns and clusters
7. **Literature Comparison** — findings vs. published benchmarks with registered citations
8. **Recommendations** — unresolved questions, sensitivity analyses, next steps

---

## Available Tools

The agent has access to these tool categories:

| Category | Tools |
|---|---|
| **Data Quality** | `duplicate_checks`, `type_coercion_suggestions`, `missing_data_patterns` |
| **Exploration** | `column_detailed_stats`, `value_counts`, `sample_rows`, `outlier_detection` |
| **Descriptive** | `group_by_summary`, `correlation_matrix`, `cross_tabulation`, `distribution_info`, `plot_spec` |
| **Clinical Trial** | `baseline_balance`, `stat_test_selector` |
| **Inferential** | `normality_test`, `hypothesis_test`, `pairwise_group_test`, `effect_size`, `survival_analysis`, `regression`, `multiple_testing_correction` |
| **Literature** | `search_pubmed`, `search_semantic_scholar`, `evidence_table_builder`, `citation_manager` |

---

## Data Dictionary Support

You can provide a JSON file that maps column names to plain-English descriptions so the model does not assume anything about the input data:

```json
{
  "id": "Patient identifier",
  "time": "Days between registration and death/transplant/study end",
  "status": "0=censored, 1=transplant, 2=dead",
  "trt": "Treatment arm: 1=D-penicillamine, 2=placebo",
  "age": "Age in years at registration",
  "sex": "Patient sex: m=male, f=female"
}
```

Pass it with `--data-dictionary`:
```bash
poetry run qtrial analyze --file data/pbc.csv --data-dictionary data/pbc_dict.json
```

The descriptions are injected verbatim into the prompt as authoritative definitions.

---

## Requirements
* Python 3.12+
* API key for at least one provider

### Option A: Install with Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

Verify:
```bash
poetry --version
```

From inside `backend/`:
```bash
poetry config virtualenvs.in-project true
poetry install
```

This creates: `backend/.venv/`

> If VS Code does not detect dependencies, select the interpreter manually at `backend/.venv/bin/python`

### Option B: Install with pip (no Poetry)

From inside `backend/`:
```bash
python -m venv .venv

# Activate the virtual environment
# Linux / Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -e .
```

---

## Environment Variables

Create a `.env` file inside `backend/`. Only set the provider(s) you intend to use.

```env
# --- Direct providers ---
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=your_anthropic_key
CLAUDE_MODEL=claude-opus-4-6

# --- OpenRouter (access any model via one key) ---
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o          # any model slug from openrouter.ai/models

# --- Optional: increase rate limits ---
NCBI_API_KEY=your_ncbi_key              # PubMed E-utilities
S2_API_KEY=your_s2_key                  # Semantic Scholar

# --- Agent tuning ---
MAX_AGENT_ITERATIONS=25
MAX_TOOL_RESULT_CHARS=4000
```

---

## Project Structure

```
backend/
  data/
    pbc.csv                        # sample dataset
    pbc_dict.json                  # sample data dictionary
  src/qtrial_backend/
    config.py                      # env-based settings
    main.py                        # CLI (typer)
    agent/
      context.py                   # shared state (dataframe, caches, citations)
      loop.py                      # while-loop orchestrator (LLM ↔ tools)
    core/
      types.py                     # ProviderName, Message, ToolCall, ChatResponse, etc.
      router.py                    # provider routing
    dataset/
      load.py                      # CSV/XLSX ingestion
      preview.py                   # structured preview builder
    prompts/
      insights.py                  # single-shot insight prompt
      agent_system.py              # agent system prompt + initial message template
    providers/
      base.py                      # LLMClient ABC
      openai_client.py             # OpenAI direct
      gemini_client.py             # Google Gemini
      claude_client.py             # Anthropic Claude
      openrouter_client.py         # OpenRouter (OpenAI-compatible)
    tools/
      registry.py                  # @tool decorator + ToolRegistry
      converter.py                 # tool schema → provider format
      stats/                       # 20+ statistical tools
      literature/                  # PubMed, Semantic Scholar, evidence table, citation manager
  pyproject.toml
```

---

## Running the CLI

Place your dataset inside `backend/data/` (e.g., `backend/data/pbc.csv`).

### Quick Insights (single-shot)
```bash
poetry run qtrial insights --file data/pbc.csv --provider openai
```

### Agentic Analysis (multi-iteration)
```bash
# OpenAI
poetry run qtrial analyze --file data/pbc.csv --provider openai

# Gemini
poetry run qtrial analyze --file data/pbc.csv --provider gemini

# Claude
poetry run qtrial analyze --file data/pbc.csv --provider claude

# OpenRouter (any model)
poetry run qtrial analyze --file data/pbc.csv --provider openrouter

# With data dictionary + verbose logging
poetry run qtrial analyze --file data/pbc.csv --provider openrouter \
  --data-dictionary data/pbc_dict.json --verbose
```

Supported file types: `.csv`, `.xlsx`

---

## Technical Details

### What the Model Receives
* Dataset shape
* Column schema (names + dtypes)
* Column descriptions (if data dictionary provided)
* First 5 rows (preview)
* Missingness percentages
* Numeric summary statistics

### What the Agent Does Server-Side
* Runs all statistical computations on the **full dataset** via tools
* Caches duplicate tool calls automatically
* Enforces citation traceability (no fabricated references)
* Applies multiple testing correction when > 5 tests are run

### Current Limitations
* No API server (CLI only)
* No evaluation/judge system yet
* No RAG grounding yet

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
