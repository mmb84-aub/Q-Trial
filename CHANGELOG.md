# CHANGELOG

All notable changes to Q-Trial are documented here.

---

## [Current] — Documentation & Code Synchronisation (Task 1)

### Removed

#### `agentic/agents.py` — DataQualityAgent, ClinicalSemanticsAgent, UnknownsAgent, InsightSynthesisAgent, JudgeAgent/Critic

**What it was:** Five specialised LLM agents, each making a separate LLM API call in sequence:
- `DataQualityAgent` — prompted the LLM to assess data quality from raw dataset stats.
- `ClinicalSemanticsAgent` — prompted the LLM to interpret column semantics and units.
- `UnknownsAgent` — prompted the LLM to enumerate unknowns and assumptions.
- `InsightSynthesisAgent` — prompted the LLM to synthesise all prior agents' outputs.
- `JudgeAgent / Critic` — prompted the LLM to evaluate the synthesis output.

**Why removed:**

| Agent | Replacement | Reason |
|-------|-------------|--------|
| DataQualityAgent | `build_dataset_evidence()` in `dataset/evidence.py` | Pure Python is deterministic, faster, and produces the same missingness/duplicate/correlation signals without wasting an LLM call. |
| ClinicalSemanticsAgent | Statistical agent loop (Stage 4) | The iterative tool-calling loop implicitly handles semantic interpretation as the LLM calls tools and reasons about outputs. A dedicated prior LLM call added latency with no measurable benefit. |
| UnknownsAgent | Synthesis stage (Stage 7) | Unknowns surfaced by the statistical loop are passed directly to the single synthesis call. A separate "unknowns" LLM call was redundant. |
| InsightSynthesisAgent | `run_synthesis_call()` in `orchestrator.py` | Collapsed to one structured LLM call with a single JSON schema (`FinalReportSchema`). Multiple chained synthesis calls caused schema drift and were harder to validate. |
| JudgeAgent / Critic | `_annotate_confidence_warnings()` in `agent/runner.py` | Post-loop confidence checks (missingness > 30%, n below minimum) are fully deterministic. An LLM judge call added stochasticity to what should be a hard rule. |

---

#### `agentic/planner.py` — `call_planner()`

**What it was:** An LLM call that generated a sequential plan of agent calls (PlanSchema with PlanStep objects) before the main pipeline ran.

**Why removed:** The pipeline was refactored to a fixed 8-stage sequence. A dynamic planner that could choose which agents to run introduced non-determinism and made the pipeline harder to debug. The fixed sequence is sufficient and more transparent.

---

#### `agentic/reasoning.py` — `run_reasoning_engine()`

**What it was:** A multi-step LLM reasoning engine (Task 4A) that maintained a `ReasoningState` with candidate hypotheses, claim drafts, contradiction checks, and iterative validation passes.

**Why removed:** Not in the agreed pipeline design. The statistical agent loop (Stage 4) already performs iterative LLM reasoning via tool calls. A separate reasoning engine duplicated this responsibility. The data models defined here (`ReasoningState`, `CandidateHypothesis`, etc.) are no longer exported.

---

#### `agentic/hypothesis_gen.py` — `generate_dynamic_hypotheses()`

**What it was:** Task 4C — a single LLM call to generate candidate hypotheses with falsification checks and "hidden questions", integrated into `ReasoningState`.

**Why removed:** Depended on `reasoning.py` (removed above) and is not part of the active pipeline design. Removed with it.

---

#### `agentic/hypothesis_tool_dispatch.py` — `dispatch_hypothesis_tools()`

**What it was:** Task 4B — routes LLM-generated `ToolDispatchRequest` objects to registered stats tools and collects results for injection into the synthesis prompt.

**Why removed:** Depended on `reasoning.py` and `hypothesis_gen.py` (both removed). Not part of the active pipeline design.

---

#### `tools/literature/rag.py` — `run_literature_rag()` function only

**What it was:** A hypothesis-driven literature retrieval function that ran PubMed and Semantic Scholar queries derived from `ReasoningState` hypotheses.

**Why removed (function only):** The function depends on the removed reasoning engine's hypothesis output format and is not called anywhere in the active pipeline. The file itself was retained because it defines `LiteratureArticle` and `LiteratureRAGReport` data models used by the active Stage 6 literature validator.

---

### Added

#### Module docstrings (all key `.py` files)

Every module in the active pipeline now has a module-level docstring specifying:
- **Input** — what the module consumes
- **Output** — what it produces
- **Does** — what it computes / why it exists

Files updated: `main.py`, `api.py`, `agent/loop.py`, `agent/context.py`, all `dataset/` modules, all `agentic/` modules, all `providers/`, `core/router.py`, `core/types.py`, `tools/registry.py`, `tools/converter.py`, all 17 `tools/stats/` modules lacking docstrings.

---

### Changed

#### `README.md` — complete rewrite

The root README now accurately describes the 8-stage pipeline, includes an ASCII architecture diagram showing all stages with their input/output and LLM-vs-pure-code classification, and contains quick-start instructions, API reference, and provider table.

The old README described removed components (hypothesis-driven analysis, judge-based evaluation) as active features.

#### `agentic/__init__.py` — stripped to active exports only

Removed re-exports for all deleted modules. The package now exports only:
`run_agentic_insights`, `FinalReportSchema`, `MetadataInput`, `ToolCallRecord`, `UnknownsOutput`.

---

## [2026-04-17] — Task 2: Clinical Statistical Rigor (feat/task2-audit)

### Added
- `tools/stats/clinical_stats.py` — three-stage clinical trial analysis: integrity (Little's MCAR, baseline balance), analysis (MMRM, effect sizes, odds ratio), and multiple-testing correction stage.
- `tools/stats/mice_imputation.py` — MICE imputation using Rubin's Rules for MAR columns.
- `tools/stats/power_analysis.py` — achieved power and required-n calculations.
- `tools/stats/mmrm.py` — Mixed-Model Repeated Measures implementation.

### Fixed
- Little's MCAR test classification label corrected from `"MAR or MNAR"` to `"Not MCAR"`.
- MMRM result now surfaces a top-level `p_value` field for Stage 3 extraction.
- `corrected_findings` now includes `n_required_80pct_power` and `odds_ratio`.
- MICE tool description updated to match actual trigger string `"Not MCAR"`.
- Static report Stage 3 table updated to show OR and required sample size columns.
