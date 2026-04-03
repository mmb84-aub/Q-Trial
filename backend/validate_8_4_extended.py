#!/usr/bin/env python3
"""
Validation 8.4 Extended: Redundancy Reduction with Tuned Parameters

Testing with increased num_reads (2000) and higher lambda_penalty (0.7, 1.0)
to achieve the 15% minimum redundancy reduction threshold.
"""

import sys
sys.path.insert(0, "./src")

import pandas as pd
import numpy as np
from qtrial_backend.quantum.feature_selector import (
    compute_relevance_scores,
    compute_redundancy_matrix,
    construct_qubo_matrix,
    solve_qubo,
    apply_hard_constraints,
    mean_pairwise_correlation,
)

print("=" * 80)
print("VALIDATION 8.4 EXTENDED: TUNING LAMBDA_PENALTY")
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
print(f"   Candidates: {len(candidate_cols)}")

# Compute relevance and redundancy once
print("\n2. Computing relevance and redundancy matrices...")
relevance_scores = compute_relevance_scores(df, outcome_col, candidate_cols)
redundancy_matrix = compute_redundancy_matrix(df, candidate_cols)

# Measure baseline redundancy
numeric_candidates = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c].dtype)]
redundancy_before = mean_pairwise_correlation(df, numeric_candidates)
print(f"   Redundancy before: {redundancy_before:.6f}")

# Test multiple lambda values
lambda_values = [0.5, 0.7, 1.0, 1.5, 2.0]
print(f"\n3. Testing different lambda_penalty values:")
print("   " + "=" * 76)

results_by_lambda = {}

for lam in lambda_values:
    print(f"\n   Lambda = {lam}")
    
    # Construct and solve QUBO with this lambda
    Q = construct_qubo_matrix(relevance_scores, redundancy_matrix, candidate_cols, lambda_penalty=lam)
    best_sample = solve_qubo(Q, num_reads=1000, num_sweeps=1000)
    selected_indices = [i for i, val in best_sample.items() if val == 1]
    
    # Apply hard constraints
    selected_columns = apply_hard_constraints(
        selected_indices,
        candidate_cols,
        relevance_scores,
        outcome_column=outcome_col,
    )
    
    # Measure redundancy after
    numeric_selected = [c for c in selected_columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
    redundancy_after = mean_pairwise_correlation(df, numeric_selected)
    reduction = (redundancy_before - redundancy_after) / redundancy_before if redundancy_before > 0 else 0.0
    
    results_by_lambda[lam] = {
        'n_selected': len(selected_columns),
        'redundancy_after': redundancy_after,
        'reduction': reduction,
        'selected': selected_columns,
    }
    
    print(f"      Selected: {len(selected_columns)} columns")
    print(f"      Redundancy after: {redundancy_after:.6f}")
    print(f"      Reduction: {reduction*100:.2f}%")
    print(f"      {'✅ PASS' if reduction >= 0.15 else '⚠️  Below 15%'}")

# Find best lambda
print("\n4. Summary of Results:")
print("   " + "=" * 76)
print(f"   {'Lambda':<8} {'Selected':<10} {'Red.Before':<15} {'Red.After':<15} {'Reduction':<12}")
print("   " + "-" * 76)
for lam in lambda_values:
    r = results_by_lambda[lam]
    print(f"   {lam:<8.1f} {r['n_selected']:<10} {redundancy_before:<15.6f} {r['redundancy_after']:<15.6f} {r['reduction']*100:>10.2f}%")

best_lambda = max(lambda_values, key=lambda l: results_by_lambda[l]['reduction'])
best_reduction = results_by_lambda[best_lambda]['reduction']

print("\n   " + "=" * 76)
print(f"   Best lambda: {best_lambda} (achieves {best_reduction*100:.2f}% reduction)")

if best_reduction >= 0.15:
    print(f"   ✅ ACHIEVES >= 15% THRESHOLD at lambda={best_lambda}")
else:
    print(f"   ⚠️  Maximum reduction {best_reduction*100:.2f}% still below 15% threshold")
    print(f"      Consider increasing num_reads further or examining dataset characteristics")

print("\n" + "=" * 80)
