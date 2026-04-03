"""
Synthesis Quality Scorer.

After the synthesis LLM call produces its SynthesisOutput, this module runs a
self-scoring LLM call that evaluates the synthesis on four dimensions:
  1. Completeness — are all required output fields populated?
  2. Clinical relevance — are outputs grounded in the study context?
  3. Plain language — is the narrative free of statistical jargon?
  4. Evidence grounding — are recommendations linked to actual findings?

If the score is below SYNTHESIS_QUALITY_THRESHOLD (default 0.7), the caller
should re-run the synthesis call once before proceeding.
"""
from __future__ import annotations

import json
import os

from qtrial_backend.agentic.schemas import SynthesisOutput, SynthesisQualityScore
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName

SYNTHESIS_QUALITY_THRESHOLD = float(
    os.environ.get("SYNTHESIS_QUALITY_THRESHOLD", "0.7")
)

_SYSTEM_PROMPT = """\
You are a clinical research quality reviewer. Score the following synthesis \
output on a scale from 0.0 to 1.0 based on four equally weighted criteria:

1. Completeness (0–0.25): Are future_trial_hypothesis, \
endpoint_improvement_recommendations, recommended_sample_size, \
variables_to_control, and research_questions all non-empty?
2. Clinical relevance (0–0.25): Are the recommendations and research questions \
relevant to the stated study context?
3. Plain language (0–0.25): Is the narrative_summary free of raw statistical \
values, p-values, and engineering jargon?
4. Evidence grounding (0–0.25): Are recommendations and research questions \
linked to specific observed findings rather than generic advice?

Return ONLY a JSON object:
{"score": <float 0.0–1.0>, "rationale": "<one sentence per criterion>"}
No markdown, no extra keys.
"""


def score_synthesis_quality(
    output: SynthesisOutput,
    study_context: str,
    provider: ProviderName = "gemini",
) -> SynthesisQualityScore:
    """
    Score the SynthesisOutput and return a SynthesisQualityScore.

    On LLM failure, returns a conservative score of 0.0 so the caller
    triggers a re-run rather than silently accepting a bad synthesis.
    """
    client = get_client(provider)
    user = (
        f"Study context: {study_context}\n\n"
        f"Synthesis output:\n{json.dumps(output.model_dump(), indent=2, ensure_ascii=False)}\n\n"
        'Return JSON: {"score": ..., "rationale": "..."}'
    )
    req = LLMRequest(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user,
        payload={"temperature": 0},
    )
    try:
        resp = client.generate(req)
        raw = resp.text.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        rationale = str(data.get("rationale", ""))
        return SynthesisQualityScore(score=score, rationale=rationale)
    except Exception as exc:
        return SynthesisQualityScore(
            score=0.0,
            rationale=f"Quality scoring failed: {exc}. Defaulting to 0.0 to trigger re-run.",
        )
