from __future__ import annotations

import json
import textwrap
from typing import Any

from qtrial_backend.agentic.schemas import (
    FailedClaim,
    JudgeOutput,
    RubricScores,
)
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName


# ── prompt ────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = textwrap.dedent("""\
    You are JudgeAgent, a rigorous scientific peer-reviewer evaluating a clinical
    data-analysis report produced by an AI pipeline.

    Your role is to identify unsupported claims, clinical overreach, and internal
    inconsistencies — NOT to rewrite the report.

    Rubric dimensions (score 0–100 each; higher = better):

    1. evidence_support: Every factual claim must map to a key present in the
       EVIDENCE_DICT, or must be explicitly framed as an assumption.
       Deduct for any claim referencing columns, statistics, or thresholds absent
       from the evidence.

    2. clinical_overreach: Deduct when the report makes causal, prognostic, or
       prescriptive clinical statements the data cannot support.
       Hedged language ("suggests", "may indicate", "is consistent with") is fine.

    3. uncertainty_handling: When endpoint semantics, column meanings, or study
       design are unknown/ambiguous, the report must hedge.
       Deduct for definitive conclusions drawn from ambiguous data.

    4. internal_consistency: Deduct when conclusions contradict the high-impact
       unknowns listed in UNKNOWNS_OUTPUT (e.g. claiming clean data when
       missingness > 20% was flagged, or using "definitive" language when
       high-impact unknowns remain unresolved).

    Respond with ONLY valid JSON — no markdown fences, no extra text.
""")

_JUDGE_USER = textwrap.dedent("""\
    Evaluate the FINAL_INSIGHTS_REPORT below according to the rubric.

    Required JSON schema:
    {{
      "overall_score": <int 0-100>,
      "rubric": {{
        "evidence_support": <int 0-100>,
        "clinical_overreach": <int 0-100>,
        "uncertainty_handling": <int 0-100>,
        "internal_consistency": <int 0-100>
      }},
      "failed_claims": [
        {{
          "claim_text": "<exact quote from report>",
          "reason": "<why it fails the rubric>",
          "missing_evidence": "<what data/key would fix it, or null>",
          "severity": "<low|medium|high>"
        }}
      ],
      "rewrite_instructions": ["<ordered instruction>", ...],
      "judge_reasoning": "<2-4 sentence overall assessment>"
    }}

    === FINAL_INSIGHTS_REPORT ===
    {insights_json}

    === EVIDENCE_DICT (keys available to the report) ===
    {evidence_json}

    === UNKNOWNS_OUTPUT ===
    {unknowns_json}
""")


# ── agent ─────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        ).strip()
    return text


def run_judge_agent(
    final_insights: Any,   # InsightSynthesisOutput or dict
    evidence: dict[str, Any],
    unknowns: Any,          # UnknownsOutput or dict
    provider: ProviderName,
) -> JudgeOutput:
    """
    Evaluate ``final_insights`` against the rubric and return a typed
    :class:`JudgeOutput`.  Falls back to a zero-score output on parse failure
    so the pipeline never crashes due to judge errors.
    """
    # Normalise inputs to plain dicts for serialisation
    insights_dict = (
        final_insights.model_dump()
        if hasattr(final_insights, "model_dump")
        else final_insights
    )
    unknowns_dict = (
        unknowns.model_dump()
        if hasattr(unknowns, "model_dump")
        else unknowns
    )

    # Truncate evidence to avoid exceeding context limits
    evidence_str = json.dumps(evidence, indent=2, default=str, ensure_ascii=False)
    if len(evidence_str) > 8000:
        evidence_str = evidence_str[:8000] + "\n... (truncated for context)"

    user = _JUDGE_USER.format(
        insights_json=json.dumps(insights_dict, indent=2, ensure_ascii=False),
        evidence_json=evidence_str,
        unknowns_json=json.dumps(unknowns_dict, indent=2, ensure_ascii=False),
    )

    client = get_client(provider)
    req = LLMRequest(system_prompt=_JUDGE_SYSTEM, user_prompt=user, payload={})
    resp = client.generate(req)
    raw = _strip_fences(resp.text)

    try:
        data = json.loads(raw)
        return JudgeOutput.model_validate(data)
    except Exception as exc:
        # Retry once with an explicit correction prompt
        try:
            fix_req = LLMRequest(
                system_prompt=_JUDGE_SYSTEM,
                user_prompt=(
                    "Your previous response was not valid JSON matching the required schema.\n"
                    f"Error: {exc}\n\nPrevious response:\n{raw}\n\n"
                    "Fix it and return ONLY valid JSON. No markdown, no explanation."
                ),
                payload={},
            )
            fix_resp = client.generate(fix_req)
            fixed_raw = _strip_fences(fix_resp.text)
            data = json.loads(fixed_raw)
            return JudgeOutput.model_validate(data)
        except Exception as exc2:
            return JudgeOutput(
                overall_score=0,
                rubric=RubricScores(
                    evidence_support=0,
                    clinical_overreach=0,
                    uncertainty_handling=0,
                    internal_consistency=0,
                ),
                failed_claims=[
                    FailedClaim(
                        claim_text="(entire report)",
                        reason=f"JudgeAgent returned unparseable output after retry: {exc2}",
                        missing_evidence=None,
                        severity="high",
                    )
                ],
                rewrite_instructions=["Inspect judge LLM raw output; schema mismatch."],
                judge_reasoning=f"Parse error: {exc2}. Raw snippet: {raw[:300]}",
            )
