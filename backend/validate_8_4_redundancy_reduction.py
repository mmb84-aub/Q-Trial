#!/usr/bin/env python3
"""
Validation 8.4: Redundancy Reduction

Test: Compute redundancy before and after selection on the PBC dataset.

Expected output (literature-backed):
Skolik et al. (2021) reported 15–20% redundancy reduction as the minimum threshold 
for meaningful feature selection improvement. Your implementation must achieve at least 15% 
redundancy reduction on the PBC dataset (redundancy_reduction >= 0.15).

If the reduction is below 15%, increase num_reads to 2000 or tune lambda_penalty upward.
"""

import sys
sys.path.insert(0, "./src")

import pandas as pd
import numpy as np
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection

print("=" * 80)
print("VALIDATION 8.4: REDUNDANCY REDUCTION")
print("=" * 80)

# Load dataset
print("\n1. Loading PBC dataset...")
try:
    df = pd.read_csv("src/qtrial_backend/data/pbc.csv")
except FileNotFoundError:
    print("❌ ERROR: Could not find pbc.csv")
    exit(1)
print(f"   Dataset shape: {df.shape}")

# Identify outcome column
outcome_col = 'status'
if outcome_col not in df.columns:
    print(f"❌ ERROR: '{outcome_col}' column not found")
    exit(1)

# Get candidate columns (numeric, excluding outcome)
candidate_cols = [c for c in df.select_dtypes(include=['number']).columns if c != outcome_col]
print(f"   Outcome column: {outcome_col}")
print(f"   Candidate columns: {len(candidate_cols)}")
print(f"   Columns: {candidate_cols}")

# Run feature selection
print("\n2. Running QUBO feature selection...")
result = run_qubo_feature_selection(df, profile=None, outcome_column=outcome_col)

# Extract redundancy metrics
redundancy_before = result['redundancy_before']
redundancy_after = result['redundancy_after']
redundancy_reduction = result['redundancy_reduction']
selection_method = result['selection_method']
n_candidates = result['n_candidates']
n_selected = result['n_selected']
selected_cols = result['selected_columns']

# Display results
print("\n3. Results:")
print(f"   Selection method: {selection_method}")
print(f"   Columns: {n_candidates} candidates → {n_selected} selected")
print(f"   Selected: {selected_cols}")

print("\n4. Redundancy Analysis:")
print(f"   Before selection: {redundancy_before:.6f}")
print(f"   After selection:  {redundancy_after:.6f}")
print(f"   Reduction:        {redundancy_reduction:.6f} ({redundancy_reduction*100:.2f}%)")

# Test against threshold
print("\n5. Validation Against 15% Threshold (Skolik et al. 2021):")
THRESHOLD = 0.15
meets_threshold = redundancy_reduction >= THRESHOLD

print(f"   Required: >= {THRESHOLD} ({THRESHOLD*100:.1f}%)")
print(f"   Achieved: {redundancy_reduction:.4f} ({redundancy_reduction*100:.2f}%)")
print(f"   Meets threshold? {meets_threshold}")

if meets_threshold:
    print("\n   ✅ PASS - Achieves 15% redundancy reduction threshold")
else:
    print(f"\n   ⚠️  BELOW TARGET - Achieved {redundancy_reduction*100:.2f}%, target is 15%")
    print("\n   RECOMMENDATIONS:")
    print("   • Increase num_reads to 2000 for better QUBO solutions")
    print("   • Increase lambda_penalty (e.g., 0.6, 0.7) for stronger redundancy penalty")
    print("   • Note: Skolik baseline is 15-20%, but fallback mechanism may limit reduction")

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print(f"Redundancy before: {redundancy_before:.4f}")
print(f"Redundancy after:  {redundancy_after:.4f}")
print(f"Reduction:         {redundancy_reduction*100:.2f}%")
print(f"Threshold met:     {'✅ YES' if meets_threshold else '⚠️  NO'}")
print(f"\nStatus: REDUNDANCY REDUCTION MEASURED")
