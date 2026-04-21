import sys
sys.path.insert(0, 'src')
import pandas as pd
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection

df = pd.read_csv('src/qtrial_backend/data/pbc.csv')
result = run_qubo_feature_selection(df, 'status')

print('=== FULL RESULT ===')
print(f'Selection method: {result["selection_method"]}')
print(f'Selected columns: {result["selected_columns"]}')
print(f'Redundancy before: {result["redundancy_before"]}')
print(f'Redundancy after: {result["redundancy_after"]}')
print(f'Redundancy reduction: {result["redundancy_reduction"]} ({result["redundancy_reduction_pct"]}%)')
