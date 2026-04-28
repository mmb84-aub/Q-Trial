# Q-Trial Backend

Python backend powering the **Q-Trial clinical trial intelligence system**.

The backend is responsible for ingesting datasets, running the analysis pipeline, orchestrating LLM reasoning, executing statistical tools, retrieving literature evidence, and producing structured insight reports.

It exposes both:

- a **CLI interface** for experimentation and research workflows  
- a **FastAPI service** used by the frontend and external integrations

---

# What the Backend Does

The backend implements an **agentic analysis pipeline** that combines deterministic statistics with LLM reasoning.

At a high level it can:

- Load and profile clinical trial datasets
- Detect data quality issues and dataset risks
- Run statistical analysis and hypothesis tests
- Generate structured insights using LLM reasoning
- Validate claims using deterministic checks
- Retrieve biomedical literature to ground conclusions
- Surface unknowns and follow-up questions
- Produce structured outputs suitable for downstream reporting

---

# Core Capabilities

### Dataset Ingestion & Profiling

The backend loads CSV or Excel datasets and builds a structured dataset preview containing:

- schema and variable types
- preview rows
- missingness statistics
- numeric summaries
- dataset-level evidence

The full dataset **never leaves the backend** — only summarized previews are sent to the model.

---

### Deterministic Evidence & Guardrails

Before LLM reasoning begins, the system extracts deterministic signals such as:

- missing data patterns
- duplicate identifiers
- outlier summaries
- correlations
- cardinality issues

Additional guardrails detect dataset risks including:

- low-cardinality numeric columns
- implausible physiological values
- unit inconsistencies
- repeated measurements

These signals act as **grounding evidence for the analysis pipeline**.

---

### Agentic Analysis Pipeline

The backend orchestrates a multi-step agentic pipeline where specialized agents analyze the dataset:

1. Data Quality Analysis  
2. Clinical Semantics Interpretation  
3. Unknown Detection  
4. Insight Synthesis  

The pipeline may then optionally run a **Judge agent** that evaluates the reliability and grounding of the generated insights.

---

### Statistical Tool Execution

The LLM can call a large set of statistical tools that operate directly on the dataset.

These tools cover:

- data quality checks
- descriptive statistics
- hypothesis testing
- effect size computation
- regression analysis
- survival analysis
- multiple testing correction

All statistical computation runs **server-side on the full dataset**.

---

### Hypothesis-Driven Analysis

The system can dynamically generate hypotheses and follow-up checks based on earlier findings.

These hypotheses may trigger:

- additional statistical tests
- targeted dataset exploration
- literature retrieval

This allows the analysis to move beyond static summaries and perform **iterative discovery**.

---

### Literature Retrieval

The backend integrates biomedical search APIs to compare dataset findings with published research.

Supported sources:

- PubMed  
- Semantic Scholar  

Retrieved papers can be summarized and cited in the final report.

---

### Evidence Retrieval (RAG)

The backend includes a lightweight evidence retrieval subsystem that can index:

- dataset evidence
- tool outputs
- retrieved literature

Relevant evidence can then be retrieved during reasoning to support or validate claims.

---

# Supported LLM Providers

The backend supports multiple providers through a unified routing layer:

- OpenAI  
- Google Gemini  
- Anthropic Claude  
- OpenRouter  

Provider-specific tool schemas and message formats are handled automatically.

---

# Interfaces

### CLI

The CLI is useful for experimentation, benchmarking, and reproducible analysis runs.

Example:

```bash
poetry run qtrial analyze --file data/pbc.csv --provider openai
```

---

### API (FastAPI)

The backend also exposes a FastAPI server used by the frontend.

Available endpoints include:

- dataset analysis
- streaming run progress
- health checks

---

# Project Structure

```text
backend/
  src/qtrial_backend/

    main.py                CLI entrypoint
    api.py                 FastAPI server

    agent/                 agent loop and runtime context
    agentic/               planner, specialist agents, orchestrator, judge
    dataset/               ingestion, preview building, evidence extraction, guardrails
    rag/                   evidence indexing and retrieval
    report/                deterministic report generation

    core/                  shared types and provider router
    providers/             OpenAI, Gemini, Claude, OpenRouter integrations

    tools/
      stats/               statistical analysis tools
      literature/          PubMed and Semantic Scholar integrations
      registry.py          tool registration system
```

---

# Environment Variables

Create a `.env` file inside `backend/`.

Example:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-opus-4

OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o
```

Optional keys:

```env
NCBI_API_KEY=...
S2_API_KEY=...
```

---

# Installation

### With Poetry

```bash
poetry install
```

---

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

# Running the Backend

Example CLI run:

```bash
poetry run qtrial analyze \
  --file data/pbc.csv \
  --provider openai
```

Supported file formats:

- `.csv`
- `.xlsx`

---

# Development Notes

The backend is designed around a **modular architecture**:

- provider-agnostic LLM routing  
- deterministic statistical computation  
- tool-driven exploration  
- evidence-grounded reasoning  

This structure allows the system to evolve while keeping the core analysis pipeline transparent and reproducible.

---

# Current Status

The backend is functional but still evolving.

Active areas of development include:

- improving reasoning validation
- expanding statistical tool coverage
- improving literature grounding
- strengthening evaluation and benchmarking workflows