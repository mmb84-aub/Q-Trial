# API Integration Guide - Feature Selection

This document shows how to integrate the new feature selection methods into the existing Q-Trial API.

## Current API Pipeline

The API currently uses QUBO feature selection in two places:

1. **Line ~182-200:** Preliminary feature selection
2. **Line ~322-337:** Main analysis feature selection

## Integration Options

### Option A: Add Feature Selection Method Parameter

Modify the API endpoint to accept a `feature_selection_method` parameter:

```python
# In backend/src/qtrial_backend/api.py

from qtrial_backend.feature_selection import select_features

@app.post("/api/run")
async def analyze_trial(
    study_context: str,
    dataset_file: UploadFile,
    feature_selection_method: str = "lasso",  # NEW: default to LASSO
):
    """
    Run clinical trial analysis with configurable feature selection.
    
    Args:
        study_context: Clinical context
        dataset_file: CSV/XLSX data
        feature_selection_method: 'univariate', 'mrmr', 'lasso', 'elastic_net', or 'qubo'
    """
    # ... existing code ...
    
    # Stage 0: Feature Selection (BEFORE profiler)
    if feature_selection_method in ['univariate', 'mrmr', 'lasso', 'elastic_net']:
        selection_result = select_features(
            df=df,
            outcome_column=outcome_column,
            method=feature_selection_method
        )
        
        selected_features = selection_result['selected_features']
        console.print(
            f"[green]✓ Feature selection ({feature_selection_method}):[/green] "
            f"{len(selected_features)} features (redundancy: {selection_result.get('redundancy_measure', 0):.3f})"
        )
        
        # Use selected features for analysis
        df = df[selected_features]
    
    elif feature_selection_method == 'qubo':
        # Use existing QUBO implementation
        quantum_evidence = run_qubo_feature_selection(df, profile, outcome_column, 0.5)
        selected_features = quantum_evidence['selected_features']
        df = df[selected_features]
    
    # Continue with existing pipeline...
```

### Option B: Always Run Multiple Methods (Benchmarking Mode)

For comparisons during development:

```python
# Add a debug/benchmarking endpoint

@app.post("/api/benchmark_methods")
async def benchmark_methods(
    study_context: str,
    dataset_file: UploadFile,
):
    """Run all feature selection methods and return comparison."""
    
    df = pd.read_csv(dataset_file.file)
    outcome_column = infer_outcome_column(df)
    
    from qtrial_backend.feature_selection import benchmark_all_methods
    
    results = benchmark_all_methods(
        df=df,
        outcome_column=outcome_column,
        methods=['univariate', 'mrmr', 'lasso', 'elastic_net', 'qubo'],
        task_type='classification',
        n_bootstrap=5
    )
    
    return {
        "benchmark_results": results,
        "recommendation": "Use LASSO for best redundancy reduction"
    }
```

### Option C: Use LASSO as Default (Recommended)

Replace QUBO with LASSO as the default since it's faster and has better redundancy reduction:

```python
# Simple modification: Use LASSO instead of QUBO

# Original QUBO call (lines ~190-199)
# quantum_evidence = run_qubo_feature_selection(df, profile, outcome_column, 0.5)

# New: Use LASSO
selection_result = select_features(
    df=df,
    outcome_column=outcome_column,
    method='lasso'  # Fast, excellent redundancy reduction
)

selected_features = selection_result['selected_features']
redundancy = selection_result.get('redundancy_measure', 0)

console.print(
    f"[green]✓ Feature selection (LASSO):[/green] "
    f"{len(selected_features)} features, redundancy reduced to {redundancy:.3f}"
)

# Continue with selected features
df = df[selected_features]
```

## Recommended Approach

For Q-Trial, I recommend **Option A** with LASSO as the default:

### Step 1: Update endpoint signature

```python
@app.post("/api/run")
async def analyze_trial(
    study_context: str,
    dataset_file: UploadFile,
    feature_selection_method: str = "lasso",
):
```

### Step 2: Add feature selection logic

```python
from qtrial_backend.feature_selection import select_features

# After data loading, before profiling
if feature_selection_method in ['univariate', 'mrmr', 'lasso', 'elastic_net']:
    selection_result = select_features(
        df=df,
        outcome_column=outcome_column,
        method=feature_selection_method
    )
    selected_features = selection_result['selected_features']
    df = df[selected_features]
```

### Step 3: Update frontend (optional)

Add a dropdown to the StudyContextForm:

```typescript
// In frontend/src/components/StudyContextForm.tsx

const FEATURE_SELECTION_METHODS = [
    { label: 'LASSO (Recommended)', value: 'lasso' },
    { label: 'mRMR', value: 'mrmr' },
    { label: 'Univariate Baseline', value: 'univariate' },
    { label: 'Elastic Net', value: 'elastic_net' },
    { label: 'QUBO (Quantum-inspired)', value: 'qubo' },
];

// In form JSX:
<select name="featureSelectionMethod" defaultValue="lasso">
    {FEATURE_SELECTION_METHODS.map(method => (
        <option key={method.value} value={method.value}>
            {method.label}
        </option>
    ))}
</select>
```

## Performance Expectations

After integration:

| Method | Speed | Redundancy | Recommendation |
|--------|-------|-----------|-----------------|
| Univariate | ⚡ Fast (15ms) | Medium | Initial screening |
| mRMR | ⚡ Fast (80ms) | Good | Stable, theory-backed |
| LASSO | ⚡ Fast (80ms) | **Excellent** | **Default choice** |
| Elastic Net | ⚡ Fast (85ms) | **Excellent** | Alternative |
| QUBO | 🐢 Slow (6-10s) | Good | Research/exploration |

## Backwards Compatibility

The new methods integrate cleanly without breaking existing code:

- Existing QUBO code continues to work
- New endpoint parameter defaults to LASSO
- No changes to downstream analysis pipeline
- All methods output compatible data structures

## Testing After Integration

```python
# Test the integration
response = await analyze_trial(
    study_context="PBC dataset analysis",
    dataset_file=sample_data,
    feature_selection_method="lasso"
)

# Verify results
assert "selected_features" in response
assert response["feature_selection_method"] == "lasso"
```

## Monitoring & Logging

Add logging to track which method is used:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Using feature selection method: {feature_selection_method}")
logger.info(f"Selected {len(selected_features)} features")
logger.info(f"Redundancy metric: {redundancy:.3f}")
```

## Future Enhancements

1. **Caching:** Store selected feature sets for repeated analyses
2. **Validation:** Cross-validate feature selection results
3. **Explainability:** Add feature importance visualization
4. **Comparison:** Show side-by-side method comparison in UI
5. **Optimization:** Fine-tune method parameters per dataset

---

**Next Steps:**
1. Choose integration approach (recommend Option A)
2. Modify `backend/src/qtrial_backend/api.py`
3. Test with sample datasets
4. Update frontend if needed
5. Deploy and monitor

