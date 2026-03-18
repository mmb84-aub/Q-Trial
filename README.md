# Q-Trial

Q-Trial is an AI-powered clinical trial dataset analyzer that combines multi-agent LLM reasoning with 30+ statistical and literature tools to surface evidence-grounded insights, risks, and optimization opportunities from clinical trial data.

---

## ✅ Implemented Features

### Core System Capabilities

- **Multi-mode analysis pipeline** — choose between a fast single-shot LLM call (`--mode single`) or the full multi-agent agentic pipeline (`--mode agentic`, default)
- **Privacy-preserving LLM context** — the full dataset is never sent to any LLM; only a structured preview (shape, schema, missingness, numeric summary, first 5 rows) is transmitted; all computation runs server-side
- **Iterative agent loop** — LLM drives up to 25 tool-calling iterations, with automatic caching of duplicate calls and exponential backoff on errors
- **Deterministic static report** — a fully reproducible pre-computation report is generated before any LLM call and injected as context
- **Structured JSON output** — all pipeline results are validated Pydantic models, enabling downstream programmatic use

### Agent / Reasoning Architecture

- **Planner agent** — decomposes the analysis task into a step-by-step execution plan (`agentic/planner.py`)
- **DataQualityAgent** — identifies missingness patterns, type errors, duplicates, and outliers (`agentic/agents.py`)
- **ClinicalSemanticsAgent** — maps column names to clinical meaning and infers study design from data patterns (`agentic/agents.py`)
- **UnknownsAgent** — ranks critical gaps and ambiguities by impact; produces actionable open questions (`agentic/agents.py`)
- **InsightSynthesisAgent** — generates key findings, risk signals, and ranked recommendations with evidence citations (`agentic/agents.py`)
- **LLM-as-Judge evaluation** — a separate Judge agent scores the final report on four rubric dimensions (Evidence Support, Clinical Overreach, Uncertainty Handling, Internal Consistency), each 0–100; produces failed-claim annotations and rewrite instructions (`agentic/judge.py`)
- **Dynamic hypothesis generation** — the system generates and dispatches statistical hypotheses based on dataset structure (`agentic/hypothesis_gen.py`, `agentic/hypothesis_tool_dispatch.py`)
- **Reasoning engine** — structured chain-of-thought scaffolding for each specialist agent (`agentic/reasoning.py`)
- **Interactive closed-loop refinement** — `--interactive` flag enables a terminal Q&A loop where the user answers high-impact unknowns, after which Semantics, Unknowns, Synthesis, and Judge agents all re-run until critical gaps are resolved
- **Metadata-driven re-synthesis** — supply a JSON metadata file (`--metadata`) to pre-answer known questions (status mappings, treatment arms, lab units, study design); triggers an automatic before/after comparison

### Data Ingestion & Preprocessing

- **CSV and XLSX ingestion** — loads any tabular dataset via `pandas` (`dataset/load.py`)
- **Structured dataset preview builder** — constructs a compact, token-efficient context payload (shape, schema, head rows, missingness %, numeric summary) (`dataset/preview.py`)
- **Evidence collection** — compiles a full statistical evidence dictionary from the dataset for agent consumption (`dataset/evidence.py`)
- **Guardrails & robustness checks** — deterministic data quality flags before any LLM call (`dataset/guardrails.py`)
- **Optional data dictionary** — a JSON file mapping column names to plain-English descriptions is injected verbatim into the prompt as authoritative definitions

### Statistical Analysis Tools (30+ tools)

| Category | Tools |
|---|---|
| **Data Quality** | `duplicate_checks`, `type_coercion_suggestions`, `missing_data_patterns` |
| **Exploration** | `column_detailed_stats`, `value_counts`, `sample_rows`, `outlier_detection` |
| **Descriptive** | `group_by_summary`, `correlation_matrix`, `cross_tabulation`, `distribution_info`, `plot_spec` |
| **Clinical Trial** | `baseline_balance` (Table 1 with SMD), `stat_test_selector` |
| **Inferential** | `normality_test`, `hypothesis_test`, `pairwise_group_test`, `effect_size`, `survival_analysis`, `regression`, `multiple_testing_correction` |
| **Literature** | `search_pubmed`, `search_semantic_scholar`, `evidence_table_builder`, `citation_manager` |

- **Tool registry with decorator pattern** — `@tool(name, description, params_model, category)` for zero-boilerplate tool registration (`tools/registry.py`)
- **Automatic tool schema conversion** — converts tool definitions to each provider's native function-calling format (`tools/converter.py`)
- **Result caching** — identical tool calls return cached results, preventing redundant computation
- **Multiple testing correction** — applied automatically when more than 5 statistical tests are run

### External Integrations (LLM Providers & APIs)

- **OpenAI** — GPT-4o and other models via direct API (`providers/openai_client.py`)
- **Google Gemini** — supports comma-separated API keys for quota rotation across multiple keys (`providers/gemini_client.py`)
- **Anthropic Claude** — Claude Opus and other Claude models (`providers/claude_client.py`)
- **OpenRouter** — single API key gives access to 100+ models from OpenAI, Anthropic, Meta, Mistral, and more (`providers/openrouter_client.py`)
- **PubMed (NCBI E-utilities)** — searches biomedical literature; rate limit increases from 3/s to 10/s with an optional `NCBI_API_KEY` (`tools/literature/pubmed.py`)
- **Semantic Scholar** — searches the S2 academic graph; optional `S2_API_KEY` for higher rate limits (`tools/literature/semantic_scholar.py`)

### Evidence Retrieval / RAG

- **RAG ingestion pipeline** — chunks documents and indexes them for retrieval (`rag/ingestion.py`, `rag/chunking.py`)
- **BM25 retriever** — lexical keyword-based retrieval over ingested documents (`rag/bm25_retriever.py`)
- **Hybrid retriever** — combines BM25 with dense embedding retrieval (`rag/retriever.py`)
- **Vector store** — in-memory document store for the retrieval pipeline (`rag/store.py`)
- **Evidence table builder** — aggregates retrieved literature into a structured evidence table with citations (`tools/literature/evidence_table.py`)
- **Citation manager** — tracks all citations through the pipeline, preventing hallucinated references (`tools/literature/citation_manager.py`)
- **RAG tool** — agent-callable tool that retrieves relevant evidence passages on demand (`tools/rag/retrieve_evidence.py`)

### Reporting / Output Generation

- **Static deterministic report** — pre-LLM report generated from direct pandas/scipy computation; always reproducible (`report/static.py`)
- **Structured final report** — Pydantic-validated `FinalReportSchema` with sections for key findings, risks, ranked recommendations, unknowns, assumptions, required documents, and judge evaluation
- **8-section clinical report structure**: Dataset Overview, Data Quality Assessment, Baseline Characteristics, Key Statistical Findings, Survival Analysis, Feature Relations & Derived Features, Literature Comparison, Recommendations
- **Before/after metadata comparison** — when metadata resolves unknowns, the system prints a side-by-side diff of finding counts and judge score delta

### API Server

- **FastAPI REST server** — `api.py` wraps the full pipeline behind an HTTP interface
- **`POST /api/run`** — accepts a multipart file upload and provider selection; returns a JSON analysis report
- **`GET /api/health`** — liveness check endpoint
- **CORS configured** for Streamlit frontend on `localhost:8501`

### Web Dashboard (Streamlit Frontend)

- **File upload** — drag-and-drop CSV/XLSX upload (`frontend/app.py`)
- **Provider selection** — dropdown to choose OpenAI, Gemini, Claude, or OpenRouter
- **Real-time progress tracking** — visual pipeline stage tracker rendered via a custom JavaScript component (`frontend/components/pipeline.js`)
- **Static report panel** — displays the deterministic pre-LLM analysis
- **Report visualization** — tabbed view of key findings, risks, recommendations, unknowns, and judge scores
- **Literature evidence panel** — shows cited PubMed/S2 references with summaries
- **Interactive unknowns panel** — displays ranked open questions and resolution status

### CLI

- **`qtrial insights`** — main analysis command; supports `--mode agentic` (default) and `--mode single`
- **`--provider`** — selects the LLM provider (`openai`, `gemini`, `claude`, `openrouter`)
- **`--judge / --no-judge`** — enables or disables the Judge agent (default: on)
- **`--interactive`** — enables the interactive closed-loop Q&A terminal session
- **`--metadata`** — path to a JSON file with pre-supplied metadata answers
- **`--max-rows` / `--max-cols`** — controls preview size sent to the LLM
- **`--verbose`** — detailed logging of each agent step and tool call
- **Rich terminal output** — colored panels, rules, and formatted tables using the `rich` library

### Configuration System

- **Environment-variable configuration** — all provider keys, model names, and agent tuning parameters loaded from `.env` via `python-dotenv` (`config.py`)
- **Configurable agent iteration limit** — `MAX_AGENT_ITERATIONS` (default: 25)
- **Configurable tool result truncation** — `MAX_TOOL_RESULT_CHARS` (default: 4000 chars per result)
- **Multi-key Gemini support** — comma-separated `GEMINI_API_KEY` values enable automatic quota rotation

---

## 🏗️ Architecture

```
Dataset (CSV/XLSX)
    │
    ▼
load_dataset()  ──────────────────────────────►  pd.DataFrame
    │
    ├──► build_dataset_preview()  ──► {shape, schema, head, missingness, numeric_summary}
    │
    ├──► build_static_report()    ──► deterministic pre-LLM statistical report
    │
    └──► run_statistical_agent_loop()  ──► LLM ↔ 30+ tools (up to 25 iterations)
              │
              ▼
    run_agentic_insights()
              │
              ├──► PlannerAgent          ──► step-by-step execution plan
              ├──► DataQualityAgent      ──► issues: missingness, types, outliers
              ├──► ClinicalSemanticsAgent──► column meanings, study design inference
              ├──► UnknownsAgent         ──► ranked critical gaps & open questions
              └──► InsightSynthesisAgent ──► key findings, risks, recommendations
                        │
                        ▼  (if --judge)
                   JudgeAgent           ──► 4-rubric score (0–100 each)
                        │
                        ▼  (if --interactive or --metadata)
                   re-synthesis loop    ──► closed-loop refinement
                        │
                        ▼
              FinalReportSchema (Pydantic)
```

**Key architectural properties:**
- **Provider-agnostic** — all four LLM clients implement the same `LLMClient` ABC (`chat()`, `generate()`)
- **Stateless LLM calls** — the model never accumulates state; each agent call is fully self-contained
- **Tool-call loop** — the `AgentLoop` in `agent/loop.py` drives the LLM–tool conversation with deduplication and error recovery
- **Pydantic schemas throughout** — all inter-agent data (`schemas.py`) and tool parameters are validated at runtime

---

## 🚀 Quickstart

### Prerequisites

- Python 3.12+
- API key for at least one supported LLM provider

### Install (Poetry)

```bash
cd backend
poetry install
cp .env.example .env   # then add your API key(s)
```

### Install (pip)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

### Run CLI Analysis

```bash
# Agentic analysis (default) with Judge evaluation
poetry run qtrial insights --file data/pbc.csv --provider openai

# With a data dictionary and interactive Q&A
poetry run qtrial insights --file data/pbc.csv --provider openai \
  --metadata data/metadata_template.json --interactive

# Single-shot quick insights
poetry run qtrial insights --file data/pbc.csv --provider openai --mode single

# Using OpenRouter (access 100+ models with one key)
poetry run qtrial insights --file data/pbc.csv --provider openrouter
```

### Run with Web Dashboard

```bash
# Terminal 1 — start the backend API
cd backend
poetry run uvicorn qtrial_backend.api:app --port 8000

# Terminal 2 — start the Streamlit frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚙️ Configuration

Create `backend/.env` (see `backend/.env.example`):

```env
# Choose one or more providers
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=key1,key2,key3   # comma-separated for quota rotation
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=your_key
CLAUDE_MODEL=claude-opus-4-6

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o

# Optional: higher rate limits for literature search
NCBI_API_KEY=your_ncbi_key
S2_API_KEY=your_s2_key

# Agent tuning
MAX_AGENT_ITERATIONS=25
MAX_TOOL_RESULT_CHARS=4000
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Language** | Python 3.12+ |
| **Data** | pandas, openpyxl, scipy |
| **LLM providers** | OpenAI, Google Gemini, Anthropic Claude, OpenRouter |
| **CLI** | Typer, Rich |
| **API server** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Validation** | Pydantic v2 |
| **Literature** | PubMed (NCBI E-utilities), Semantic Scholar |

---

## 📌 Roadmap

- [ ] Persistent output storage (save reports to disk / database)
- [ ] Evaluation benchmark suite
- [ ] Provider comparison / side-by-side mode
- [ ] Paper-ready evaluation outputs
- [ ] Full RAG integration with dense embeddings

---

## ⚠️ Note

This is an active research project. The core analysis pipeline, multi-agent reasoning, statistical toolset, Judge evaluation, and web dashboard are all **fully implemented**. The roadmap items above represent planned extensions.
