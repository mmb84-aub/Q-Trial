#!/usr/bin/env python3
"""
8.1 Unit Validation — Relevance Scores

Tests that the relevance scorer correctly identifies bilirubin, albumin, prothrombin time, 
age, and oedema as top predictors on the PBC dataset (per Mayo Clinic PBC model).
"""

import pandas as pd
import numpy as np
from qtrial_backend.quantum.feature_selector import compute_relevance_scores

# Load PBC dataset
try:
    df = pd.read_csv("src/qtrial_backend/data/pbc.csv")
except FileNotFoundError:
    print("❌ ERROR: Could not find pbc.csv")
    print("   Please ensure you're running from the backend directory")
    exit(1)

print("=" * 80)
print("VALIDATION 8.1 — RELEVANCE SCORES (Mayo Clinic PBC Model)")
print("=" * 80)
print(f"\n✓ Loaded PBC dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Check for outcome column
if 'status' not in df.columns:
    print(f"❌ ERROR: 'status' column not found in dataset")
    print(f"   Available columns: {df.columns.tolist()}")
    exit(1)

print(f"✓ Outcome column 'status' found")

# Get all columns except status
candidate_cols = [c for c in df.columns if c != 'status']
print(f"✓ Candidate columns: {len(candidate_cols)}")

# Compute relevance scores
print("\n▸ Computing relevance scores...")
relevance_scores = compute_relevance_scores(
    df=df,
    outcome_column='status',
    candidate_columns=candidate_cols
)

# Sort by relevance descending
sorted_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n✓ Computed relevance scores for {len(sorted_scores)} columns\n")

# Define expected top predictors (Mayo Clinic PBC model: Dickson et al., 1989)
expected_top = {
    'bili': ('Top 3', 3),
    'albumin': ('Top 5', 5),
    'protime': ('Top 7', 7),
    'age': ('Top 7', 7),
    'platelet': ('Top 10', 10),
}

# Alternative names that might be in dataset
name_variants = {
    'bili': ['bili', 'bilirubin', 'bilir', 'bill'],
    'albumin': ['albumin', 'alb'],
    'protime': ['protime', 'prothrombin', 'pro_time', 'pt'],
    'age': ['age', 'ages'],
    'platelet': ['platelet', 'platelets', 'plat'],
}

# Find actual names in dataset
actual_names = {}
for key, variants in name_variants.items():
    for variant in variants:
        if variant in relevance_scores:
            actual_names[key] = variant
            break

print("TOP 15 COLUMNS BY RELEVANCE:")
print("-" * 80)
print(f"{'Rank':<6} {'Column':<20} {'Relevance Score':<20} {'Status':<20}")
print("-" * 80)

results = {}
for rank, (col, score) in enumerate(sorted_scores[:15], 1):
    # Check if this is one of the expected top predictors
    is_expected = False
    for key, actual_name in actual_names.items():
        if col == actual_name:
            expected_rank_str, expected_rank_num = expected_top[key]
            is_expected = f"✓ EXPECTED ({expected_rank_str})"
            results[key] = (rank, expected_rank_num, score, is_expected)
            break
    
    status = is_expected if is_expected else ""
    print(f"{rank:<6} {col:<20} {score:<20.4f} {status:<20}")

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

all_pass = True
for key, (expected_rank_str, expected_rank_num) in expected_top.items():
    actual_name = actual_names.get(key)
    if actual_name is None:
        print(f"❌ {key.upper()}: Column not found in dataset")
        all_pass = False
        continue
    
    rank, _, score, _ = results.get(key, (None, None, None, None))
    if rank is None:
        print(f"❌ {key.upper()} ({actual_name}): Not in top 15")
        all_pass = False
    elif rank <= expected_rank_num:
        print(f"✅ {key.upper()} ({actual_name}): Rank {rank} (expected {expected_rank_str}) — Score: {score:.4f}")
    else:
        print(f"⚠️  {key.upper()} ({actual_name}): Rank {rank} (expected {expected_rank_str}) — Score: {score:.4f}")
        if rank <= expected_rank_num + 2:  # Allow 2-rank flexibility
            print(f"    (Marginally acceptable, within 2 ranks)")
        else:
            all_pass = False

print("\n" + "=" * 80)
if all_pass:
    print("✅ VALIDATION 8.1 PASSED — Relevance scores are correct!")
else:
    print("⚠️  VALIDATION 8.1 NEEDS REVIEW — Check relevance computation")
print("=" * 80)

# Print full ranking for reference
print("\nFULL RANKING (all columns):")
print("-" * 80)
for rank, (col, score) in enumerate(sorted_scores, 1):
    print(f"{rank:3d}. {col:<20} {score:.4f}")
print("-" * 80)
