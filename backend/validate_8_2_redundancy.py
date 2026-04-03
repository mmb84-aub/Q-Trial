#!/usr/bin/env python3
"""
8.2 Unit Validation — Redundancy Matrix

Tests that the redundancy matrix correctly identifies known correlated pairs:
- chol & trig (cholesterol & triglycerides): should be > 0.5
- stage & edema: should be > 0.4
"""

import pandas as pd
import numpy as np
from qtrial_backend.quantum.feature_selector import compute_redundancy_matrix

# Load PBC dataset
try:
    df = pd.read_csv("src/qtrial_backend/data/pbc.csv")
except FileNotFoundError:
    print("❌ ERROR: Could not find pbc.csv")
    exit(1)

print("=" * 80)
print("VALIDATION 8.2 — REDUNDANCY MATRIX (Known Correlations)")
print("=" * 80)
print(f"\n✓ Loaded PBC dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Get all numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print(f"✓ Numeric columns for redundancy: {len(numeric_cols)}")
print(f"  Columns: {numeric_cols}")

# Compute redundancy matrix
print("\n▸ Computing redundancy matrix...")
redundancy = compute_redundancy_matrix(df, numeric_cols)
print(f"✓ Computed {len(numeric_cols)} × {len(numeric_cols)} redundancy matrix")

# Create a dataframe for easier inspection
redundancy_df = pd.DataFrame(
    redundancy,
    index=numeric_cols,
    columns=numeric_cols
)

# Define expected correlations
expected_correlations = {
    ('chol', 'trig'): ('Cholesterol vs Triglycerides', 0.5),
    ('stage', 'edema'): ('Stage vs Edema', 0.4),
}

# Alternative column names
column_aliases = {
    'chol': ['chol', 'cholesterol', 'choles'],
    'trig': ['trig', 'triglycerides', 'trigly'],
    'stage': ['stage', 'disease_stage', 'dis_stage'],
    'edema': ['edema', 'oedema', 'edma'],
}

# Map actual column names
actual_col_names = {}
for key, variants in column_aliases.items():
    for variant in variants:
        if variant in numeric_cols:
            actual_col_names[key] = variant
            break

print("\n" + "=" * 80)
print("EXPECTED CORRELATION CHECKS")
print("=" * 80)

all_pass = True

for (col1_key, col2_key), (description, min_threshold) in expected_correlations.items():
    col1 = actual_col_names.get(col1_key)
    col2 = actual_col_names.get(col2_key)
    
    if col1 is None or col2 is None:
        print(f"\n❌ {description}")
        print(f"   {col1_key} or {col2_key} not found in dataset")
        all_pass = False
        continue
    
    # Get redundancy value (symmetric, so check either direction)
    idx1 = numeric_cols.index(col1)
    idx2 = numeric_cols.index(col2)
    redundancy_val = max(redundancy[idx1, idx2], redundancy[idx2, idx1])
    
    status = "✅ PASS" if redundancy_val > min_threshold else "❌ FAIL"
    print(f"\n{status} {description}")
    print(f"   {col1} × {col2}: {redundancy_val:.4f} (expected > {min_threshold})")
    
    if redundancy_val <= min_threshold:
        all_pass = False

print("\n" + "=" * 80)
print("TOP 20 PAIRWISE CORRELATIONS (excluding diagonal)")
print("=" * 80)

# Extract all unique pairs with their correlations
pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        pairs.append((numeric_cols[i], numeric_cols[j], redundancy[i, j]))

# Sort by correlation descending
pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'Rank':<6} {'Column 1':<15} {'Column 2':<15} {'Redundancy':<15} {'Status':<20}")
print("-" * 80)

for rank, (col1, col2, corr) in enumerate(pairs[:20], 1):
    status = ""
    for (key1, key2), (desc, threshold) in expected_correlations.items():
        actual1 = actual_col_names.get(key1)
        actual2 = actual_col_names.get(key2)
        if (col1 == actual1 and col2 == actual2) or (col1 == actual2 and col2 == actual1):
            if corr > threshold:
                status = "✓ EXPECTED"
            else:
                status = f"⚠ Below threshold ({threshold})"
    
    print(f"{rank:<6} {col1:<15} {col2:<15} {corr:<15.4f} {status:<20}")

print("\n" + "=" * 80)
print("FULL REDUNDANCY MATRIX (heatmap view)")
print("=" * 80)
print("\nTop 10 correlations for each column:")
print("-" * 80)

for i, col in enumerate(numeric_cols):
    if i > 9:  # Show first 10 columns only for brevity
        break
    
    # Get all correlations with this column
    col_corrs = [(numeric_cols[j], redundancy[i, j]) for j in range(len(numeric_cols)) if i != j]
    col_corrs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{col}:")
    for rank, (other_col, corr_val) in enumerate(col_corrs[:5], 1):
        print(f"  {rank}. {other_col:<15} {corr_val:.4f}")

print("\n" + "=" * 80)
if all_pass:
    print("✅ VALIDATION 8.2 PASSED — Redundancy matrix is correct!")
else:
    print("⚠️  VALIDATION 8.2 NEEDS REVIEW — Some expected correlations missing")
print("=" * 80)
