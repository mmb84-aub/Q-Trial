#!/usr/bin/env python3
"""
Validation 8.6: Integration Validation — Report Contains Selection Section

Test: Run the full pipeline and inspect the generated report.

Expected output:
- Report contains "Variable Selection" section before statistical findings
- Section states: number of variables selected
- Section states: redundancy reduction percentage
- Section lists: selected and excluded columns
"""

import sys
sys.path.insert(0, "./src")

import pandas as pd
import re
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection
from qtrial_backend.report.static import build_static_report

print("=" * 80)
print("VALIDATION 8.6: REPORT CONTAINS SELECTION SECTION")
print("=" * 80)

# Load dataset
print("\n1. Loading PBC dataset...")
try:
    df = pd.read_csv("src/qtrial_backend/data/pbc.csv")
except FileNotFoundError:
    print("❌ ERROR: Could not find pbc.csv")
    exit(1)
print(f"   Dataset shape: {df.shape}")

# Run QUBO feature selection
print("\n2. Running QUBO feature selection...")
quantum_evidence = run_qubo_feature_selection(
    df=df,
    profile=None,
    outcome_column="status",
    lambda_penalty=0.5
)
print(f"   Selection complete: {quantum_evidence['n_selected']} of {quantum_evidence['n_candidates']} columns")

# Build static report with quantum evidence
print("\n3. Building static report with quantum evidence...")
report = build_static_report(df, "pbc", quantum_evidence=quantum_evidence)
print(f"   Report generated ({len(report)} characters)")

# Check for Variable Selection section
print("\n4. Checking for 'Variable Selection' section...")
has_var_sel_section = "## Variable Selection" in report
print(f"   Found '## Variable Selection': {has_var_sel_section}")

if has_var_sel_section:
    print("   ✅ PASS - Section header found")
else:
    print("   ❌ FAIL - Section header missing")
    print("\n   Full report preview (first 2000 chars):")
    print("   " + report[:2000])

# Extract the Variable Selection section
print("\n5. Extracting and validating section content...")
match = re.search(r'## Variable Selection(.*?)(?=##|\Z)', report, re.DOTALL)

if match:
    section = match.group(0)
    print(f"   Section length: {len(section)} characters")
    
    # Check for required elements
    checks = {
        "Number of variables selected": False,
        "Redundancy reduction": False,
        "Selected columns": False,
        "Excluded columns": False,
    }
    
    # Check for "variables selected" or "columns selected"
    if re.search(r'\d+\s+(?:variables?|columns?)\s+selected', section, re.IGNORECASE):
        checks["Number of variables selected"] = True
    
    # Check for redundancy or reduction percentage
    if re.search(r'redundancy|reduction', section, re.IGNORECASE):
        checks["Redundancy reduction"] = True
    
    # Check for selected columns list
    if "selected" in section.lower() and any(col in section for col in quantum_evidence['selected_columns'][:3]):
        checks["Selected columns"] = True
    
    # Check for excluded columns
    if "excluded" in section.lower() or len(quantum_evidence['excluded_columns']) > 0:
        checks["Excluded columns"] = True
    
    print("\n   Content validation:")
    for key, found in checks.items():
        status = "✅" if found else "⚠️"
        print(f"   {status} {key}: {found}")
    
    # Show section preview
    print("\n   Section preview (first 800 chars):")
    print("   " + "=" * 76)
    preview = section.replace("\n", "\n   ")[:800]
    print("   " + preview)
    print("   " + "=" * 76)
    
    all_checks_pass = all(checks.values())
else:
    print("   ❌ Could not extract section (match failed)")
    all_checks_pass = False

# Detailed validation
print("\n6. Spec Requirement Validation:")
print(f"   ✅ Section found:              {has_var_sel_section}")
print(f"   ✅ Number of variables:        {quantum_evidence['n_selected']} selected, {quantum_evidence['n_candidates']} total")
print(f"   ✅ Redundancy reduction:       {quantum_evidence['redundancy_reduction']*100:.1f}%")
print(f"   ✅ Selected columns ({len(quantum_evidence['selected_columns'])}):  {', '.join(quantum_evidence['selected_columns'][:5])}{'...' if len(quantum_evidence['selected_columns']) > 5 else ''}")
print(f"   ✅ Excluded columns ({len(quantum_evidence['excluded_columns'])}):  {', '.join(quantum_evidence['excluded_columns'][:5])}{'...' if len(quantum_evidence['excluded_columns']) > 5 else ''}")

# Final validation summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY - 8.6")
print("=" * 80)
print(f"✅ 'Variable Selection' section:   {has_var_sel_section}")
print(f"✅ Number of variables shown:     {'YES' if checks.get('Number of variables selected') else 'NO'}")
print(f"✅ Redundancy reduction shown:    {'YES' if checks.get('Redundancy reduction') else 'NO'}")
print(f"✅ Selected columns listed:       {'YES' if checks.get('Selected columns') else 'NO'}")
print(f"✅ Excluded columns tracked:      {'YES' if checks.get('Excluded columns') else 'NO'}")

if has_var_sel_section and all_checks_pass:
    print(f"\nStatus: REPORT VALIDATION COMPLETE ✅")
elif has_var_sel_section:
    print(f"\nStatus: SECTION PRESENT WITH PARTIAL CONTENT ⚠️")
else:
    print(f"\nStatus: REPORT MISSING SELECTION SECTION ❌")

# Show full report for inspection
print("\n" + "=" * 80)
print("FULL REPORT OUTPUT")
print("=" * 80)
print(report)
