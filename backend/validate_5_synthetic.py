"""
Synthetic test: inject deliberate errors to prove all 4 guardrail checks fire.
"""
import pandas as pd
import numpy as np
from qtrial_backend.dataset.guardrails import run_guardrails

# Build a synthetic dataset with deliberate violations
np.random.seed(42)
n = 40

df_synthetic = pd.DataFrame({
    # low_cardinality_numeric: trt as 0/1 integer
    "trt":       np.random.choice([0, 1], n),
    # range_violation: age with 3 impossible values (200+)
    "age":       np.concatenate([np.random.uniform(30, 80, n - 3), [210, 195, 250]]),
    # unit_plausibility: albumin in g/L (should be g/dL — median ~35 g/L instead of ~3.5 g/dL)
    "albumin":   np.random.uniform(28, 45, n),
    # repeated_measures: id column with duplicates
    "id":        np.repeat(np.arange(1, 11), 4),
    # normal numeric — should not flag
    "bili":      np.random.uniform(0.3, 8, n),
})

report = run_guardrails(df_synthetic)

print("=== SYNTHETIC TEST RESULTS ===")
print(f"Summary: {report['summary']}")
print(f"Flags ({len(report['flags'])}):")
for f in report["flags"]:
    print(f"  [{f['check_type']}] col={f.get('column')} severity={f['severity']}")
    print(f"    {f['detail'][:100]}")

print(f"\nRepeated measures: ", end="")
rm = report["repeated_measures"]
if rm:
    print(f"DETECTED — id_col={rm['id_column']}, "
          f"max_repeats={rm['max_repeats_per_subject']}, "
          f"longitudinal={rm['likely_longitudinal']}")
else:
    print("None")

print("\nExpected flags:")
print("  ✓ low_cardinality_numeric → trt (2 distinct integers)")
print("  ✓ range_violation         → age (3 values >130 years)")
print("  ✓ unit_plausibility       → albumin (median ~36, expected 1-6 g/dL)")
print("  ✓ repeated_measures       → id column with 4 rows per subject")
