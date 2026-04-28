"""
Diagnostic: show why range_violation and unit_plausibility returned 0 flags.
Prints actual stats for every column that matches a catalogue entry.
"""
import pandas as pd

df = pd.read_csv("data/pbc_sample.csv")

RANGE_CATALOGUE = [
    ("age",         0,      130,    "years"),
    ("bili",        0,      100,    "mg/dL"),
    ("albumin",     0.5,    8,      "g/dL"),
    ("protime",     5,      60,     "seconds"),
    ("platelet",    10,     1500,   "×10³/μL"),
    ("ast",         0,      5000,   "U/L"),
    ("alk_phos",    0,      15000,  "U/L"),
    ("copper",      0,      2000,   "μg/day"),
    ("trig",        0,      3000,   "mg/dL"),
    ("chol",        0,      1500,   "mg/dL"),
]

UNIT_PLAUSIBILITY = [
    ("age",         1,    100,   "age >100 median: units may not be years"),
    ("bili",        0,    50,    "bilirubin median >50: may be μmol/L not mg/dL"),
    ("albumin",     1,    6,     "albumin median outside 1-6: may be g/L not g/dL"),
    ("protime",     8,    40,    "prothrombin outside 8-40s"),
    ("platelet",    10,   1200,  "platelet outside 10-1200 ×10³/μL"),
    ("chol",        0.5,  600,   "cholesterol outside 0.5-600 mg/dL"),
    ("trig",        0,    3000,  "trig outside 0-3000 mg/dL"),
]

numeric_cols = df.select_dtypes(include="number").columns

print("=" * 70)
print("RANGE VIOLATION CHECK — actual vs plausible bounds")
print("=" * 70)
for col in numeric_cols:
    cl = col.lower().replace(" ", "_").replace("-", "_")
    for pattern, lo, hi, unit in RANGE_CATALOGUE:
        if pattern not in cl:
            continue
        s = df[col].dropna()
        out_of_range = s[(s < lo) | (s > hi)]
        print(f"\n  {col:15s} | expected [{lo}, {hi}] {unit}")
        print(f"             | min={s.min():.3f}  max={s.max():.3f}  "
              f"median={s.median():.3f}  n={len(s)}")
        print(f"             | violations: {len(out_of_range)} "
              f"({round(100*len(out_of_range)/len(s),1)}%)")
        if not out_of_range.empty:
            print(f"             | violating values: {sorted(out_of_range.unique().tolist())[:5]}")
        break

print()
print("=" * 70)
print("UNIT PLAUSIBILITY CHECK — median vs expected median range")
print("=" * 70)
for col in numeric_cols:
    cl = col.lower().replace(" ", "_").replace("-", "_")
    for pattern, med_lo, med_hi, hint in UNIT_PLAUSIBILITY:
        if pattern not in cl:
            continue
        s = df[col].dropna()
        if len(s) < 3:
            break
        med = s.median()
        flag = "⚠ FLAG" if not (med_lo <= med <= med_hi) else "✓ OK"
        print(f"  {col:15s} | median={med:.3f} | expected [{med_lo}, {med_hi}] | {flag}")
        print(f"             | hint: {hint}")
        break
