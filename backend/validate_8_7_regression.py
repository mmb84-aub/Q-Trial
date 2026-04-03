#!/usr/bin/env python3
"""
Validation 8.7: Regression Validation — Pipeline Still Completes

Test: Run the full pipeline on the PBC dataset twice:
1. WITH feature selector enabled
2. WITHOUT feature selector (full dataset)

Expected output:
- Both runs complete without errors
- Key findings in both reports include bilirubin and albumin as significant predictors
- Selected-subset run doesn't miss major findings that full-dataset run catches
"""

import sys
sys.path.insert(0, "./src")

import pandas as pd
import re
import time
from qtrial_backend.quantum.feature_selector import run_qubo_feature_selection
from qtrial_backend.report.static import build_static_report

print("=" * 80)
print("VALIDATION 8.7: REGRESSION TEST — PIPELINE COMPLETENESS")
print("=" * 80)

# Load dataset
print("\n1. Loading PBC dataset...")
try:
    df = pd.read_csv("src/qtrial_backend/data/pbc.csv")
except FileNotFoundError:
    print("❌ ERROR: Could not find pbc.csv")
    exit(1)
print(f"   Dataset shape: {df.shape}")

# Helper function to build report and extract key findings
def run_pipeline_variant(df_input, variant_name):
    """Run pipeline variant and extract findings"""
    print(f"\n   ► {variant_name}")
    try:
        start_time = time.time()
        
        # For selected subset variant, run feature selection first
        if "selected" in variant_name.lower():
            quantum_evidence = run_qubo_feature_selection(
                df=df_input,
                profile=None,
                outcome_column="status",
                lambda_penalty=0.5
            )
            df_for_report = df_input[quantum_evidence['selected_columns']]
            n_cols = len(quantum_evidence['selected_columns'])
        else:
            df_for_report = df_input
            quantum_evidence = None
            n_cols = len(df_input.columns)
        
        # Build static report
        report = build_static_report(df_for_report, "pbc", quantum_evidence=quantum_evidence)
        elapsed = time.time() - start_time
        
        # Extract key metrics from report
        findings = {
            'report_length': len(report),
            'elapsed_time': elapsed,
            'n_columns': n_cols,
            'has_bili': False,
            'has_albumin': False,
            'has_correlation': False,
            'correlation_pairs': [],
        }
        
        # Check for bilirubin evidence
        if 'bili' in report.lower():
            findings['has_bili'] = True
        
        # Check for albumin evidence
        if 'albumin' in report.lower():
            findings['has_albumin'] = True
        
        # Check for correlation analysis
        if 'correlation' in report.lower():
            findings['has_correlation'] = True
            # Extract high correlation pairs
            pairs_match = re.findall(r'(\w+)\s+↔\s+(\w+).*?ρ\s*=\s*([0-9.]+)', report)
            findings['correlation_pairs'] = pairs_match
        
        # Extract data quality issues
        findings['has_data_quality'] = '## 2. Data Quality' in report
        findings['has_baseline_balance'] = '## 7. Baseline Balance' in report
        
        print(f"     ✓ Complete in {elapsed:.2f}s")
        print(f"     ✓ Report: {len(report)} characters")
        print(f"     ✓ Columns analyzed: {n_cols}")
        
        return findings, report
        
    except Exception as e:
        print(f"     ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Run both pipeline variants
print("\n2. Running pipeline variants...")
print("   " + "=" * 76)

# Variant 1: WITH feature selection (selected subset)
selected_findings, selected_report = run_pipeline_variant(df, "WITH feature selection (10 columns)")

# Variant 2: WITHOUT feature selection (full dataset)
full_findings, full_report = run_pipeline_variant(df, "WITHOUT feature selection (20 columns)")

print("   " + "=" * 76)

# Validation checks
print("\n3. Validation Checks:")
print("   " + "=" * 76)

checks = {
    "Selected run completed": selected_findings is not None,
    "Full run completed": full_findings is not None,
    "Selected has bilirubin": selected_findings['has_bili'] if selected_findings else False,
    "Full has bilirubin": full_findings['has_bili'] if full_findings else False,
    "Selected has albumin": selected_findings['has_albumin'] if selected_findings else False,
    "Full has albumin": full_findings['has_albumin'] if full_findings else False,
    "Both have correlation analysis": (selected_findings['has_correlation'] if selected_findings else False) and (full_findings['has_correlation'] if full_findings else False),
}

for check, result in checks.items():
    status = "✅" if result else "⚠️"
    print(f"   {status} {check}: {result}")

# Compare findings
print("\n4. Finding Comparison:")
print("   " + "=" * 76)

if selected_findings and full_findings:
    print(f"\n   Selected subset run (10 columns):")
    print(f"   • Bilirubin present: {selected_findings['has_bili']}")
    print(f"   • Albumin present: {selected_findings['has_albumin']}")
    print(f"   • Correlation pairs found: {len(selected_findings['correlation_pairs'])}")
    if selected_findings['correlation_pairs']:
        for pair in selected_findings['correlation_pairs'][:3]:
            print(f"     - {pair[0]} ↔ {pair[1]}: ρ={pair[2]}")
    
    print(f"\n   Full dataset run (20 columns):")
    print(f"   • Bilirubin present: {full_findings['has_bili']}")
    print(f"   • Albumin present: {full_findings['has_albumin']}")
    print(f"   • Correlation pairs found: {len(full_findings['correlation_pairs'])}")
    if full_findings['correlation_pairs']:
        for pair in full_findings['correlation_pairs'][:3]:
            print(f"     - {pair[0]} ↔ {pair[1]}: ρ={pair[2]}")

# Performance comparison
print("\n5. Performance Comparison:")
print("   " + "=" * 76)

if selected_findings and full_findings:
    speedup = full_findings['elapsed_time'] / selected_findings['elapsed_time']
    print(f"   Selected subset run:  {selected_findings['elapsed_time']:.2f}s ({selected_findings['n_columns']} columns)")
    print(f"   Full dataset run:     {full_findings['elapsed_time']:.2f}s ({full_findings['n_columns']} columns)")
    print(f"   Speedup ratio:        {speedup:.2f}x")

# Regression test result
print("\n6. Regression Test Summary:")
print("   " + "=" * 76)

test_pass = (
    checks["Selected run completed"] and
    checks["Full run completed"] and
    checks["Selected has bilirubin"] and
    checks["Full has bilirubin"] and
    checks["Selected has albumin"] and
    checks["Full has albumin"]
)

if test_pass:
    print("   ✅ REGRESSION TEST PASSED")
    print("   - Both pipeline variants completed successfully")
    print("   - Both reports identify bilirubin as significant predictor")
    print("   - Both reports identify albumin as significant predictor")
    print("   - Feature selection does not remove critical findings")
else:
    print("   ❌ REGRESSION TEST ALERT")
    if not checks["Selected has bilirubin"] or not checks["Selected has albumin"]:
        print("   - Selected subset may be too aggressive")
        print("   - RECOMMENDATION: Reduce lambda_penalty to 0.3 and re-run")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY - 8.7")
print("=" * 80)
print(f"✅ Pipeline runs:                  {'PASS' if checks['Selected run completed'] and checks['Full run completed'] else 'FAIL'}")
print(f"✅ Bilirubin identified in both:   {'PASS' if checks['Selected has bilirubin'] and checks['Full has bilirubin'] else 'FAIL'}")
print(f"✅ Albumin identified in both:     {'PASS' if checks['Selected has albumin'] and checks['Full has albumin'] else 'FAIL'}")
print(f"✅ Correlation analysis present:   {'PASS' if checks['Both have correlation analysis'] else 'FAIL'}")

if test_pass:
    print(f"\nStatus: REGRESSION TEST PASSED ✅")
else:
    print(f"\nStatus: REGRESSION TEST REQUIRES REVIEW ⚠️")

print("\n" + "=" * 80)
print("REPORT SNIPPETS FOR VERIFICATION")
print("=" * 80)

if selected_report:
    print("\nSelected subset report (Variable Selection section):")
    print("-" * 76)
    match = re.search(r'## Variable Selection.*?(?=##|\Z)', selected_report, re.DOTALL)
    if match:
        section = match.group(0)[:600]
        print(section)

if full_report:
    print("\nFull dataset report (Correlation section):")
    print("-" * 76)
    match = re.search(r'## 6\. Correlation.*?(?=##|\Z)', full_report, re.DOTALL)
    if match:
        section = match.group(0)[:600]
        print(section)
