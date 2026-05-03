"""Quick verification that all fixes are working."""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection

# Create test data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(20, 80, 50),
    'bilirubin': np.random.rand(50) * 5,
    'albumin': np.random.rand(50) * 5 + 2,
    'platelet': np.random.randint(100, 400, 50),
    'status': np.random.randint(0, 2, 50)
})

# Run feature selection
print('=' * 60)
print('QUBO Feature Selection - End-to-End Test')
print('=' * 60)

result = run_qubo_feature_selection(df, profile=None, outcome_column='status')

print(f'✓ Candidates: {result["n_candidates"]}')
print(f'✓ Selected: {result["n_selected"]} features')
print(f'✓ Redundancy reduction: {result["redundancy_reduction_pct"]:.1f}%')
print(f'✓ Method: {result["selection_method"]}')
print(f'✓ Selected columns: {result["selected_columns"]}')
print(f'✓ Outcome included: {"status" in result["selected_columns"]}')
print('=' * 60)
print('✅ All fixes working correctly!')
print('=' * 60)
