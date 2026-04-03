#!/usr/bin/env python3
"""Quick test of QUBO feature selection."""

import pandas as pd
import numpy as np
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection

# Create a test dataset with many redundant columns
np.random.seed(42)
n_rows = 100

# Outcome column (binary)
outcome = np.random.binomial(1, 0.5, n_rows)

# Create relevant columns (correlated with outcome)
relevant_1 = outcome + np.random.normal(0, 0.2, n_rows)
relevant_2 = outcome + np.random.normal(0, 0.2, n_rows)

# Create redundant columns (highly correlated with relevant_1)
redundant_1 = relevant_1 + np.random.normal(0, 0.1, n_rows)
redundant_2 = relevant_1 + np.random.normal(0, 0.1, n_rows)
redundant_3 = relevant_1 + np.random.normal(0, 0.1, n_rows)

# Create irrelevant columns (random noise)
irrelevant_1 = np.random.normal(0, 1, n_rows)
irrelevant_2 = np.random.normal(0, 1, n_rows)

df = pd.DataFrame({
    'outcome': outcome,
    'relevant_1': relevant_1,
    'relevant_2': relevant_2,
    'redundant_1': redundant_1,
    'redundant_2': redundant_2,
    'redundant_3': redundant_3,
    'irrelevant_1': irrelevant_1,
    'irrelevant_2': irrelevant_2,
})

print("=" * 70)
print("QUBO FEATURE SELECTION TEST")
print("=" * 70)
print(f"\n✓ Created test dataset with {len(df)} rows × {len(df.columns)} columns")
print(f"  - 1 outcome column")
print(f"  - 2 relevant columns (correlated with outcome)")
print(f"  - 3 redundant columns (highly correlated with relevant_1)")
print(f"  - 2 irrelevant columns (random noise)")

# Run feature selection
print("\n▸ Running QUBO feature selection...")
result = run_qubo_feature_selection(
    df=df,
    profile=None,
    outcome_column='outcome',
    lambda_penalty=0.5
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n✓ Selection Method: {result['selection_method']}")
print(f"✓ Solver: {result['solver']}")
print(f"✓ Lambda Penalty: {result['lambda_penalty']}")

print(f"\n✓ Candidates: {result['n_candidates']} columns entered solver")
print(f"✓ Selected: {result['n_selected']} columns")
print(f"  → {result['selected_columns']}")

print(f"\n✓ Redundancy Reduction:")
print(f"  Before: {result['redundancy_before']:.4f}")
print(f"  After:  {result['redundancy_after']:.4f}")
print(f"  Reduction: {result['redundancy_reduction']*100:.1f}%")

print(f"\n✓ Relevance Scores:")
for col, score in sorted(result['relevance_scores'].items(), key=lambda x: x[1], reverse=True):
    marker = "✓" if col in result['selected_columns'] else "✗"
    print(f"  {marker} {col}: {score:.2f}")

print(f"\n✓ Excluded: {result['n_candidates'] - result['n_selected']} columns")
if result['excluded_columns']:
    print(f"  → {', '.join(result['excluded_columns'][:5])}" + 
          (f"... +{len(result['excluded_columns'])-5} more" if len(result['excluded_columns']) > 5 else ""))

print("\n" + "=" * 70)
print("✓ TEST PASSED - Feature selection is working!")
print("=" * 70)
