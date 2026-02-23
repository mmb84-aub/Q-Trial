from __future__ import annotations

SYSTEM_PROMPT = """You are a data analyst focused on clinical trial datasets.
Be practical and specific. Do not hallucinate columns that don't exist.
If you need more data, state exactly what is missing.
Return concise bullet points, grouped by category.
"""

USER_PROMPT = """Given the dataset preview payload (schema, head rows, missingness, numeric summary),
return actionable insights and issues.

Categories to include:
1) Data quality issues
2) Potential bias / imbalance signals
3) Clinical-trial relevant considerations (e.g., endpoints, missingness implications, leakage risk)
4) Next analysis steps

Keep it brief but useful.
"""
