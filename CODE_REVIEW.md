# Q-Trial Repository Code Review

**Date:** April 24, 2026  
**Reviewed by:** GitHub Copilot  
**Repository:** https://github.com/mmb84-aub/Q-Trial  

---

## Executive Summary

Q-Trial is a **clinical trial dataset analysis system** that combines deterministic statistical profiling with LLM-driven reasoning. The recent integration of **QUBO-based feature selection** adds quantum-inspired optimization to reduce feature redundancy while maintaining predictive relevance.

### Overall Assessment: ✅ **Well-Structured with Technical Debt**

**Strengths:**
- Clean 8-stage pipeline architecture with clear separation of concerns
- Comprehensive quantum feature selection using QUBO formulation
- Multi-provider LLM support (OpenAI, Claude, Gemini, Bedrock)
- Statistical rigor with multiple hypothesis tests and effect sizes
- Extensive dataset quality checks (guardrails, missing data, outliers)

**Issues Found:**
- 🔴 **CRITICAL: UTF-16 encoding issue** in `feature_selector.py` (null bytes)
- ⚠️ Statistical test coverage could be broader
- ⚠️ No comprehensive end-to-end integration tests
- ⚠️ Limited error handling in some edge cases

---

## 1. Project Architecture

### Overall Structure
```
Q-Trial/
├── backend/              # Python FastAPI service
│   ├── src/qtrial_backend/
│   │   ├── agentic/      # LLM orchestration & reasoning
│   │   ├── agent/        # Statistical agent loop
│   │   ├── dataset/      # Data loading & profiling
│   │   ├── quantum/      # ✨ QUBO feature selection
│   │   ├── providers/    # LLM client implementations
│   │   ├── report/       # Report generation
│   │   ├── tools/        # Statistical tools (30+)
│   │   ├── rag/          # Literature retrieval
│   │   └── core/         # Type definitions, routing
│   ├── tests/            # Unit tests
│   └── [test_*.py files] # Integration tests
├── frontend/             # React/TypeScript UI
└── outputs/              # Sample analysis outputs
```

### 8-Stage Pipeline
1. **Clinical Context Input** [pure code] - Study metadata
2. **Dataset Upload** [pure code] - Load & detect treatment
3. **Data Profiling** [pure code] - Structure, quality, guardrails
4. **Statistical Analysis** [LLM agent loop] - Iterative tool calling
5. **Literature Query** [LLM mini-call] - Translate to PubMed/Scholar
6. **Literature Validation** [deterministic] - Fact-check claims
7. **Synthesis** [LLM call] - Structured report generation
8. **Judge Validation** [optional LLM] - Confidence assessment

---

## 2. QUBO Feature Selection Implementation

### What Was Added

A **quantum-inspired feature selection module** that uses QUBO (Quadratic Unconstrained Binary Optimization) to select a subset of features that:
- ✅ Maximizes relevance to outcome
- ✅ Minimizes redundancy between selected features
- ✅ Maintains interpretability (4–20 features)

### Algorithm Pipeline (7 Steps)

| Step | Function | Purpose | Input | Output |
|------|----------|---------|-------|--------|
| 1 | `compute_relevance_scores()` | Statistical relevance to outcome | DataFrame | dict: {col: score} |
| 2 | `compute_redundancy_matrix()` | Pairwise feature correlations | DataFrame | M×M matrix [0,1] |
| 3 | `construct_qubo_matrix()` | QUBO optimization objective | Scores, matrix | QUBO matrix |
| 4 | `solve_qubo()` | Simulated annealing solver | QUBO matrix | Binary solution |
| 5 | `apply_hard_constraints()` | Post-solver filtering | Solution | Valid feature subset |
| 6 | `mean_pairwise_correlation()` | Redundancy measurement | Feature subset | Float [0,1] |
| 7 | `run_qubo_feature_selection()` | Main orchestration | DataFrame, params | Selected columns dict |

### QUBO Formulation

**Objective Function:**
$$\text{minimize} \; -\sum_{i=1}^{M} r_i x_i + \lambda \sum_{i<j} c_{ij} x_i x_j + \mu \sum_{i=1}^{M} \frac{x_i}{M}$$

**Where:**
- $x_i \in \{0,1\}$ = binary feature selection variable
- $r_i \in [0,1]$ = relevance score (correlation with outcome)
- $c_{ij} \in [0,1]$ = redundancy (pairwise correlation magnitude)
- $\lambda$ = adaptive penalty weight for redundancy (0.8–1.2)
- $\mu = 0.05$ = parsimony weight (encourages reasonable feature count)

### Statistical Relevance Scoring

Feature-outcome correlation varies by data types:

| Feature | Outcome | Method | Implementation |
|---------|---------|--------|-----------------|
| Numeric | Numeric | Pearson correlation | `abs(np.corrcoef())` |
| Categorical | Numeric | Eta-squared (ANOVA) | `f_oneway()` variance ratio |
| Numeric | Categorical | Point-biserial | `scipy.stats.pointbiserialr()` |
| Categorical | Categorical | Cramér's V | Chi-square based, normalized |

**Normalization:** Uses 95th percentile (not max) to avoid artificial 100% scores

### Solver Configuration

**Implementation:** D-Wave `neal.SimulatedAnnealingSampler()`
- Simulated annealing (classical, no quantum hardware needed)
- Default: 1,000 reads × 1,000 sweeps
- Production: 2,000 × 2,000 sweeps
- Temperature range: β ∈ [0.01, 5.0]
- **Multi-attempt strategy:** 3 independent runs, selects best

### Hard Constraints

Applied post-solver to ensure valid selection:

```python
min_features = max(4, ceil(sqrt(M)))    # Diversity floor
max_features = min(20, M - 1)           # Interpretability cap
always_include = [outcome_column]       # Clinical relevance
never_include = [explicitly_excluded]   # User preferences
```

### Fallback Mechanism

If QUBO achieves <15% redundancy reduction (below target), falls back to **greedy diversity selection**:
- Iteratively adds features with lowest max-correlation to selected set
- Achieves robust feature reduction without stochasticity

### Output Dictionary

```python
{
    "selected_columns": [...],           # Final feature list
    "excluded_columns": [...],           # Unselected features
    "relevance_scores": {col: score},   # Correlation to outcome
    "redundancy_before": 0.XXX,         # Mean pairwise correlation
    "redundancy_after": 0.XXX,          # After selection
    "redundancy_reduction": 0.XXX,      # Fractional reduction
    "redundancy_reduction_pct": XX.X,   # Percentage reduction
    "n_candidates": M,                  # Total features analyzed
    "n_selected": N,                    # Features selected
    "solver": "simulated_annealing",
    "selection_method": "qubo" | "greedy_diversity",
    "lambda_penalty": 0.5,
    "num_reads": 2000,
    "num_sweeps": 2000,
}
```

---

## 3. Integration Points

### 1. **API Endpoint** (`backend/src/qtrial_backend/api.py`, lines 180–206)

```python
from qtrial_backend.quantum import run_qubo_feature_selection

# Build dataset profile for QUBO
profile = build_dataset_evidence(df)
outcome_column = meta.outcome_column if meta else None

# Run feature selection
quantum_evidence = await asyncio.to_thread(
    run_qubo_feature_selection, df, profile, outcome_column, 0.5
)

# Select subset of dataframe
selected_df = df[quantum_evidence["selected_columns"]]

# Pass to downstream stages
build_static_report(selected_df, ..., quantum_evidence=quantum_evidence)
run_statistical_agent_loop(selected_df, ..., quantum_evidence=quantum_evidence)
```

✅ **Good:** Runs in thread to avoid blocking FastAPI event loop  
✅ **Good:** Passes evidence to both static and agentic pipelines  
⚠️ **Issue:** No timeout enforcement; long-running QUBO could block

### 2. **Agent Runner** (`backend/src/qtrial_backend/agent/runner.py`, lines 59–73)

```python
if quantum_evidence is not None:
    n_selected = quantum_evidence.get("n_selected", 0)
    n_candidates = quantum_evidence.get("n_candidates", 0)
    reduction_pct = quantum_evidence.get("redundancy_reduction_pct", 0.0)
    
    # Inject into LLM system message
    system_msg += (
        f"N={n_selected} variables selected from {n_candidates} using "
        f"QUBO-based feature selection (redundancy reduced by {reduction_pct:.1f}%)"
    )
```

✅ **Good:** Provides context to LLM about feature reduction  
✅ **Good:** Transparent about selection method

### 3. **Static Report** (`build_static_report()`)

Passes `quantum_evidence` through to report generation for documentation purposes.

---

## 4. Dependencies

### New Dependency Added

```
dwave-neal>=0.6.0  # D-Wave Simulated Annealing Solver
```

Full runtime dependencies now include:
```
pandas>=3.0.1,<4.0.0
scipy>=1.17.1
numpy
scikit-learn>=1.4.0
dwave-neal>=0.6.0      # ← QUBO solver (NEW)
lifelines>=0.29.0      # Survival analysis
statsmodels>=0.14.0    # Statistical models
```

✅ **Good:** Minimal new dependencies  
✅ **Good:** Using well-established D-Wave library  

---

## 5. Issues & Critical Findings

### 🔴 CRITICAL: UTF-16 Encoding Issue

**File:** `backend/src/qtrial_backend/quantum/feature_selector.py`

**Problem:** File is saved with UTF-16 encoding, causing null bytes when Python tries to parse:
```
SyntaxError: source code string cannot contain null bytes
```

**Impact:** The code cannot be imported or executed by the Python interpreter.

**Root Cause:** Likely saved in VS Code or editor with wrong encoding.

**Solution:**
```powershell
# Re-save file with UTF-8 encoding
# In VS Code: Click "UTF-16" in status bar → select "UTF-8"
# OR use Python to fix:
```

**ACTION REQUIRED:** Convert file encoding to UTF-8 immediately.

---

### ⚠️ MEDIUM: Test Coverage Gaps

**Missing tests for:**
- ❌ QUBO solver with edge cases (single feature, high correlations)
- ❌ Fallback to greedy diversity
- ❌ Feature selection with missing data
- ❌ Categorical-only datasets
- ❌ Mixed data types (all correlation methods)

**Existing tests:**
- ✅ `debug_qubo.py` - Manual debugging script
- ✅ `test_report_json.py` - Integration with static report
- ✅ `test_greedy.py` - Baseline comparison

**Recommendation:** Add formal pytest suite for quantum module:
```python
# tests/test_qubo_feature_selection.py
def test_qubo_single_feature():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
    result = run_qubo_feature_selection(df, None, 'y')
    assert len(result['selected_columns']) >= 1

def test_qubo_high_correlation():
    # Features x and x2 are highly correlated
    # Should select only one
    ...
```

---

### ⚠️ MEDIUM: Error Handling

**Issue 1:** `api.py` silently falls back to full dataset if QUBO fails
```python
except Exception as exc:
    console.print(f"[yellow]⚠ Feature selection warning: {exc}[/yellow]")
    quantum_evidence = None
    selected_df = df  # ← Uses all features
```

**Better approach:**
- Distinguish between transient (retry) vs permanent errors
- Log full stack trace for debugging
- Optionally raise for critical failures

**Issue 2:** No timeout on simulated annealing solver

If QUBO gets stuck, no max_time_ms or similar timeout exists, could hang entire API request.

**Recommendation:**
```python
# feature_selector.py
sampler = neal.SimulatedAnnealingSampler(
    # Add timeout (if supported in future versions)
    # or wrap with concurrent.futures.ThreadPoolExecutor
)
```

---

### ⚠️ LOW: Parameter Tuning

**Current parameters:**
- `lambda_penalty = 0.5` (hardcoded in API)
- `num_reads = 2000, num_sweeps = 2000`
- `parsimony_weight = 0.05`
- `min_features = max(4, ceil(sqrt(M)))`

**Questions:**
- Were these tuned on validation data?
- How sensitive is selection to lambda_penalty?
- Is adaptive lambda (0.8–1.2x multiplier) necessary?

**Recommendation:**
- Document parameter selection rationale in code comments
- Consider making key params configurable via `MetadataInput`

---

## 6. Code Quality Assessment

### Strengths ✅

| Aspect | Status | Notes |
|--------|--------|-------|
| **Documentation** | ⭐⭐⭐⭐ | Module docstrings, algorithm description, literature references (PMID 37173974, Skolik et al. 2021) |
| **Type Hints** | ⭐⭐⭐ | Most functions typed, some edge cases missing |
| **Error Messages** | ⭐⭐⭐ | Clear logging, informative feedback |
| **Code Organization** | ⭐⭐⭐⭐ | Modular pipeline (7 well-defined steps) |
| **Statistical Rigor** | ⭐⭐⭐⭐ | Pearson, Cramér's V, eta-squared, point-biserial |
| **Integration** | ⭐⭐⭐ | Clean API, passes evidence downstream |

### Weaknesses ⚠️

| Aspect | Status | Notes |
|--------|--------|-------|
| **Test Coverage** | ⭐⭐ | Only manual tests, no pytest suite |
| **Edge Case Handling** | ⭐⭐ | Assumes numeric/categorical columns only |
| **Performance Optimization** | ⭐⭐ | No caching, no feature importance ranking |
| **Hyperparameter Tuning** | ⭐⭐ | Defaults not validated on real datasets |
| **Encoding Issues** | 🔴 | UTF-16 file encoding blocker |

---

## 7. Literature & Methodology

### References Cited in Code

1. **PMID 37173974** (2023): "Leukocytes Classification using Quantum-Inspired Evolutionary Algorithm"
   - Application domain: Cell classification
   - Relevance: QUBO for feature selection precedent

2. **Romero et al. (2022)**: "Machine Learning: Science and Technology 3, 015017"
   - Topic: Quantum-inspired ML
   - Relevance: Formulation inspiration

3. **Skolik et al. (2021)**: "Quantum Machine Intelligence 3, 27"
   - Topic: QUBO optimization
   - Relevance: 15% redundancy reduction target (used in fallback logic)

### Soundness Assessment

✅ **Algorithm is theoretically sound:**
- QUBO formulation correctly balances relevance + redundancy
- Multi-attempt strategy (3 runs) appropriate for stochastic solver
- Fallback to greedy when QUBO underperforms (Skolik target)
- Constraints ensure interpretability

⚠️ **Empirical validation needed:**
- No benchmarking against classical methods (e.g., Lasso, Elastic Net)
- No comparison of QUBO vs greedy on real clinical data
- No impact assessment: Does feature selection improve downstream LLM reasoning?

**Recommendation:** Add benchmark suite comparing QUBO to:
```python
# Baseline methods
- SelectKBest (sklearn)
- Lasso (elastic net)
- Correlation-based filter
- Random forest feature importance
```

---

## 8. Frontend Integration

### React Components Affected

The feature selection should be surfaced in the UI:

✅ Present in data flow:
- `UploadForm.tsx` - File upload triggers backend pipeline
- `StepTracker.tsx` - Shows progress through 8 stages
- `ProgressStream.tsx` - Real-time status updates

⚠️ Missing:
- UI panel showing selected vs excluded features
- Visualization of redundancy reduction
- Option to adjust lambda_penalty or re-run with different params

**Suggestion:** Add "Feature Selection Details" card:
```tsx
<Card title="Quantum Feature Selection">
  <p>Selected {n_selected} from {n_candidates} features</p>
  <p>Redundancy reduced: {reduction_pct}%</p>
  <Table>
    <Column>Feature</Column>
    <Column>Relevance Score</Column>
    <Column>Correlation</Column>
  </Table>
</Card>
```

---

## 9. Changelog Assessment

### Recent Changes (From CHANGELOG.md)

**✅ Positive Refactoring:**
- Removed 5 redundant agents (DataQuality, ClinicalSemantics, Unknowns, InsightSynthesis, Judge)
- Replaced with deterministic code where appropriate
- Reduced LLM calls from ~5 to ~2 (faster, cheaper)
- Fixed pipeline to 8 deterministic stages (more testable)

**⚠️ Breaking Changes:**
- `agentic/__init__.py` stripped down; removed imports for deleted agents
- API may have changed for existing users

---

## 10. Recommendations & Next Steps

### Priority 1 (Critical) 🔴

1. **Fix UTF-16 encoding issue in feature_selector.py**
   ```powershell
   # Convert to UTF-8
   (Get-Content "feature_selector.py" -Encoding UTF16) | Set-Content -Encoding UTF8
   ```
   - **Blocker:** Code cannot run until fixed
   - **Effort:** 5 minutes
   - **Impact:** Unblocks all testing

2. **Add comprehensive pytest suite**
   ```python
   # tests/test_qubo.py
   - test_basic_selection
   - test_high_correlation
   - test_missing_data
   - test_categorical_only
   - test_fallback_greedy
   - test_output_schema
   ```
   - **Effort:** 4–6 hours
   - **Impact:** Confidence in code quality

### Priority 2 (Important) ⚠️

3. **Add timeout to QUBO solver**
   - Prevent infinite hangs in production
   - Set `max_time_ms = 30000` or similar

4. **Benchmark QUBO vs classical baselines**
   - Compare with Lasso, SelectKBest, random forest importance
   - Measure impact on downstream analysis quality

5. **Document parameter selection**
   - Why lambda_penalty = 0.5?
   - Why min_features = sqrt(M)?
   - Add sensitivity analysis

### Priority 3 (Nice-to-Have) 💡

6. **UI enhancement:** Feature selection details panel
7. **Performance:** Cache correlation matrix for repeated calls
8. **Flexibility:** Expose lambda_penalty via MetadataInput
9. **Comparison:** Add "classical baseline" option for A/B testing

---

## 11. Summary Table

| Category | Assessment | Status |
|----------|-----------|--------|
| **Architecture** | Well-designed 8-stage pipeline | ✅ Excellent |
| **QUBO Implementation** | Theoretically sound, properly formulated | ✅ Good |
| **Dependencies** | Minimal, well-chosen (D-Wave) | ✅ Good |
| **Code Quality** | Well-documented, modular | ✅ Good |
| **Test Coverage** | Manual tests only, needs pytest | ⚠️ Medium |
| **Error Handling** | Basic, could be more robust | ⚠️ Medium |
| **Encoding Issues** | **UTF-16 blocker** | 🔴 Critical |
| **Production Readiness** | 70% ready after encoding fix | ⚠️ Medium |

---

## Final Verdict

**Q-Trial is a sophisticated, well-architected clinical trial analysis system with a thoughtfully integrated QUBO feature selection module.** The quantum-inspired approach is novel and theoretically sound, and the multi-stage pipeline demonstrates good software engineering practices.

**However, the UTF-16 encoding issue is a blocker that must be resolved immediately.** After that, the codebase would benefit from a comprehensive test suite and parameter validation on real clinical datasets.

### Ready for:
- ✅ Code review & documentation
- ✅ Integration testing (after encoding fix)
- ⚠️ Production deployment (after tests pass)
- 🔴 **NOT ready for immediate deployment** (encoding blocker)

---

**Recommendations:**
1. **Immediate:** Fix encoding, run tests
2. **Short-term:** Add pytest suite, benchmark against baselines
3. **Medium-term:** Collect user feedback on feature selection quality
4. **Long-term:** Explore actual quantum hardware (D-Wave Advantage) for very high-dimensional datasets

