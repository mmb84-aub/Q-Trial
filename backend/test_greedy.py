#!/usr/bin/env python3
"""Test greedy diversity-based feature selection"""

import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np

def mean_pairwise_correlation(df, columns):
    """Compute mean absolute pairwise correlation"""
    if not columns:
        return 0.0
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return 0.0
    corr_matrix = df[numeric_cols].corr().abs()
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    if mask.sum() == 0:
        return 0.0
    return corr_matrix.where(mask).stack().mean()

def greedy_diversity_selection(df, outcome_column, candidates, target_features=7):
    """Greedily select features to maximize diversity (minimize correlation)"""
    # Start with highest relevance
    relevances = {}
    for col in candidates:
        if outcome_column and pd.api.types.is_numeric_dtype(df[outcome_column]):
            relevances[col] = abs(df[col].corr(df[outcome_column]))
        else:
            relevances[col] = df[col].var()
    
    # Start with top relevance
    sorted_cols = sorted(candidates, key=lambda c: relevances[c], reverse=True)
    selected = [sorted_cols[0]]
    
    # Greedily add features that are least correlated with selected set
    remaining = set(sorted_cols[1:])
    
    while len(selected) < target_features and remaining:
        best_col = None
        min_max_corr = float('inf')  # We want to minimize the maximum correlation
        
        # Find feature with minimum max correlation to selected features
        for col in remaining:
            correlations = [abs(df[col].corr(df[s])) for s in selected if pd.api.types.is_numeric_dtype(df[s]) and pd.api.types.is_numeric_dtype(df[col])]
            if correlations:
                max_corr = max(correlations)  # Max correlation with any selected feature
            else:
                max_corr = 0.0  # Neutral score
            
            if max_corr < min_max_corr:  # Find the one with smallest max correlation
                min_max_corr = max_corr
                best_col = col
        
        if best_col:
            selected.append(best_col)
            remaining.remove(best_col)
        else:
            break
    
    # Add outcome column if needed
    if outcome_column and outcome_column not in selected:
        selected.insert(0, outcome_column)
    
    return selected

# Test it
df = pd.read_csv('src/qtrial_backend/data/pbc.csv')
outcome_col = 'status'
candidates = [c for c in df.select_dtypes(include=['number']).columns if c != outcome_col]

# Calculate baseline
numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
baseline_redundancy = mean_pairwise_correlation(df, numeric_candidates)

# Test greedy selection
selected = greedy_diversity_selection(df, outcome_col, candidates, target_features=7)
numeric_selected = [c for c in selected if pd.api.types.is_numeric_dtype(df[c])]
greedy_redundancy = mean_pairwise_correlation(df, numeric_selected)

reduction = (baseline_redundancy - greedy_redundancy) / baseline_redundancy * 100

print(f"Baseline redundancy: {baseline_redundancy:.4f}")
print(f"Greedy redundancy: {greedy_redundancy:.4f}")
print(f"Reduction: {reduction:.1f}%")
print(f"Selected: {selected}")
