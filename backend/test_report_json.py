#!/usr/bin/env python3
"""Test static report with quantum evidence"""

import sys
sys.path.insert(0, "src")

import pandas as pd
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection
from qtrial_backend.report.static import build_static_report

# Load dataset
df = pd.read_csv("src/qtrial_backend/data/pbc.csv")

# Run QUBO feature selection
print("Running QUBO feature selection...")
quantum_evidence = run_qubo_feature_selection(df, profile=None, outcome_column="status")
print(f"Selected {len(quantum_evidence['selected_columns'])} features")
print(f"Redundancy reduction: {quantum_evidence['redundancy_reduction_pct']:.1f}%")

# Build static report with quantum evidence
print("\nBuilding static report...")
try:
    # Suppress console output
    import io
    from contextlib import redirect_stdout
    
    with redirect_stdout(io.StringIO()):
        report = build_static_report(df, "pbc", quantum_evidence=quantum_evidence)
    
    print("✅ Report generated successfully")
    
    # Check if Variable Selection section exists
    if "Variable Selection" in report:
        print("✅ Variable Selection section FOUND in report")
        # Print a snippet
        idx = report.find("Variable Selection")
        print(f"\nSnippet:\n{report[idx:idx+300]}...")
    else:
        print("⚠️  Variable Selection section not found in report")
        print("\nReport length:", len(report))
        
except Exception as e:
    print(f"❌ Error building report: {e}")
    import traceback
    traceback.print_exc()
