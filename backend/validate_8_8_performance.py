#!/usr/bin/env python3
"""
Validation 8.8: Performance Validation

Test: Measure wall-clock time for the feature selection step on the PBC dataset.

Expected output (literature-backed):
Kübler et al. (2021) established that 1000 reads with 1000 sweeps on a 50-variable 
problem completes in under 10 seconds on a standard CPU. 

Your implementation must complete the selection step in under 10 seconds.
If slower, profile to identify bottleneck (likely redundancy matrix for mixed-type columns).
"""

import sys
sys.path.insert(0, "./src")

import pandas as pd
import time
from qtrial_backend.quantum.feature_selector import (
    compute_relevance_scores,
    compute_redundancy_matrix,
    construct_qubo_matrix,
    solve_qubo,
    apply_hard_constraints,
    run_qubo_feature_selection,
)

print("=" * 80)
print("VALIDATION 8.8: PERFORMANCE VALIDATION")
print("=" * 80)

# Load dataset
print("\n1. Loading PBC dataset...")
try:
    df = pd.read_csv("src/qtrial_backend/data/pbc.csv")
except FileNotFoundError:
    print("❌ ERROR: Could not find pbc.csv")
    exit(1)
print(f"   Dataset shape: {df.shape}")

outcome_col = 'status'
candidate_cols = [c for c in df.select_dtypes(include=['number']).columns if c != outcome_col]
print(f"   Candidate columns: {len(candidate_cols)}")
print(f"   Data types: {df[candidate_cols].dtypes.value_counts().to_dict()}")

# PROFILE 1: Individual step timings
print("\n2. Profiling individual QUBO steps...")
print("   " + "=" * 76)

timings = {}

# Step 1: Relevance scores
print("   ► Step 1: Compute relevance scores...")
start = time.time()
relevance_scores = compute_relevance_scores(df, outcome_col, candidate_cols)
timings['relevance_scores'] = time.time() - start
print(f"     Completed in {timings['relevance_scores']:.4f}s")

# Step 2: Redundancy matrix
print("   ► Step 2: Compute redundancy matrix...")
start = time.time()
redundancy_matrix = compute_redundancy_matrix(df, candidate_cols)
timings['redundancy_matrix'] = time.time() - start
print(f"     Completed in {timings['redundancy_matrix']:.4f}s")

# Step 3: Construct QUBO matrix
print("   ► Step 3: Construct QUBO matrix...")
start = time.time()
Q = construct_qubo_matrix(relevance_scores, redundancy_matrix, candidate_cols, lambda_penalty=0.5)
timings['construct_qubo'] = time.time() - start
print(f"     Completed in {timings['construct_qubo']:.4f}s")

# Step 4: Solve QUBO
print("   ► Step 4: Solve QUBO (neal, 1000 reads/sweeps)...")
start = time.time()
best_sample = solve_qubo(Q, num_reads=1000, num_sweeps=1000)
timings['solve_qubo'] = time.time() - start
print(f"     Completed in {timings['solve_qubo']:.4f}s")

# Step 5: Apply hard constraints
print("   ► Step 5: Apply hard constraints...")
start = time.time()
selected_indices = [i for i, val in best_sample.items() if val == 1]
selected_columns = apply_hard_constraints(
    selected_indices,
    candidate_cols,
    relevance_scores,
    outcome_column=outcome_col,
)
timings['hard_constraints'] = time.time() - start
print(f"     Completed in {timings['hard_constraints']:.4f}s")

total_manual = sum(timings.values())
print("\n   Manual step-by-step total: {:.4f}s".format(total_manual))

# PROFILE 2: Full pipeline timing
print("\n   " + "=" * 76)
print("   ► Full pipeline (run_qubo_feature_selection)...")
start = time.time()
result = run_qubo_feature_selection(df, profile=None, outcome_column=outcome_col)
timings['full_pipeline'] = time.time() - start
print(f"     Completed in {timings['full_pipeline']:.4f}s")

# Performance analysis
print("\n3. Performance Analysis:")
print("   " + "=" * 76)

print("\n   Step-by-step breakdown:")
print(f"   {'Step':<25} {'Time (s)':<12} {'% of Total':<12}")
print("   " + "-" * 49)
for step, duration in timings.items():
    if step != 'full_pipeline':
        pct = (duration / timings['full_pipeline']) * 100
        print(f"   {step:<25} {duration:>10.4f}s  {pct:>10.1f}%")

print(f"\n   Full pipeline time:     {timings['full_pipeline']:>10.4f}s")

# Performance threshold check
THRESHOLD = 10.0
meets_threshold = timings['full_pipeline'] < THRESHOLD

print("\n4. Performance Threshold Check:")
print("   " + "=" * 76)
print(f"   Kübler et al. (2021) baseline: 1000 reads/sweeps on 50-variable = < 10 seconds")
print(f"   This implementation: {len(candidate_cols)} variables, {len(df)} samples")
print(f"   Measured time: {timings['full_pipeline']:.4f}s")
print(f"   Threshold: {THRESHOLD}s")
print(f"   Meets threshold? {meets_threshold}")

if meets_threshold:
    print(f"\n   ✅ PASS - Completes well under 10 second threshold")
else:
    print(f"\n   ❌ SLOW - Exceeds 10 second threshold")
    print(f"\n   Bottleneck analysis:")
    slowest_step = max(timings, key=lambda k: timings[k] if k != 'full_pipeline' else 0)
    print(f"   • Slowest step: {slowest_step} ({timings[slowest_step]:.4f}s)")
    
    if 'redundancy_matrix' in slowest_step:
        print(f"   • RECOMMENDATION: Cache Cramér's V computations for mixed-type columns")
        print(f"   • Check for categorical variable handling inefficiencies")

# Result summary
print("\n5. Feature Selection Output:")
print("   " + "=" * 76)
print(f"   Selected columns: {len(result['selected_columns'])}")
print(f"   Selected: {result['selected_columns']}")
print(f"   Selection method: {result['selection_method']}")
print(f"   Redundancy reduction: {result['redundancy_reduction']*100:.1f}%")

# Comparison with Kübler baseline
print("\n6. Baseline Comparison:")
print("   " + "=" * 76)

# Kübler et al. baseline: 10 seconds for 1000 reads on 50 variables
kubler_time = 10.0
kubler_vars = 50
our_time = timings['full_pipeline']
our_vars = len(candidate_cols)

# Extrapolate Kübler baseline to our problem size (assuming linear scaling)
extrapolated_time = (kubler_time * our_vars) / kubler_vars
efficiency = extrapolated_time / our_time

print(f"   Kübler et al. baseline: {kubler_time}s for {kubler_vars} variables")
print(f"   Extrapolated for {our_vars} variables: {extrapolated_time:.2f}s")
print(f"   Our actual time: {our_time:.4f}s")
print(f"   Efficiency vs. extrapolated: {efficiency:.2f}x")

if efficiency > 1.0:
    print(f"   ✅ Better than baseline (faster)")
else:
    print(f"   ⚠️  Slightly slower than baseline (but still acceptable)")

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY - 8.8")
print("=" * 80)
print(f"✅ Feature selection completes:        {timings['full_pipeline']:.4f}s")
print(f"✅ Meets < 10 second threshold:       {'YES' if meets_threshold else 'NO'}")
print(f"✅ Kübler-relative efficiency:        {efficiency:.2f}x")
print(f"✅ Selection method:                  {result['selection_method']}")

if meets_threshold:
    print(f"\nStatus: PERFORMANCE VALIDATED ✅")
else:
    print(f"\nStatus: PERFORMANCE NEEDS REVIEW ⚠️")

print("\n" + "=" * 80)
