import pandas as pd
from qtrial_backend.dataset.guardrails import run_guardrails, format_guardrail_citations

df = pd.read_csv("data/pbc_sample.csv")
report = run_guardrails(df)

print("=== SUMMARY ===")
print(report["summary"])
print()
print(f"Flags: {len(report['flags'])}")
print(f"Counts by type: {report['counts_by_type']}")
print()
print("=== FLAGS ===")
for i, f in enumerate(report["flags"]):
    print(f"[{i}] ({f['severity']}) [{f['check_type']}] col={f.get('column')}")
    print(f"    {f['detail'][:120]}")
print()
print("=== REPEATED MEASURES ===")
rm = report["repeated_measures"]
print(rm if rm else "None detected")
print()
print("=== CITATIONS ===")
for c in format_guardrail_citations(report):
    print(" ", c)
