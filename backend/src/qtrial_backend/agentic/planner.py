from __future__ import annotations

import json
import textwrap

from qtrial_backend.agentic.schemas import PlanSchema, PlanStep
from qtrial_backend.core.router import get_client
from qtrial_backend.core.types import LLMRequest, ProviderName


_SYSTEM = textwrap.dedent("""\
    You are a planning agent for a clinical trial data analysis pipeline.
    Your only job is to produce a structured JSON analysis plan.
    You must respond with ONLY valid JSON that matches the schema exactly.
    Do not include markdown fences, commentary, or extra keys.
""")

_USER = textwrap.dedent("""\
    Given the dataset preview and evidence below, produce a JSON plan with 4-5 ordered steps.

    Available agents (use exact names only):
      - DataQualityAgent
      - ClinicalSemanticsAgent
      - UnknownsAgent
      - InsightSynthesisAgent

    HARD RULES — violation causes automatic retry:
    1. InsightSynthesisAgent MUST be the LAST step.
    2. UnknownsAgent MUST appear after ClinicalSemanticsAgent and before InsightSynthesisAgent.
    3. inputs_used MUST contain ONLY keys from the ALLOWED KEYS list below.
       Raw column names like "bili", "albumin", "status" are FORBIDDEN in inputs_used.
       Use section-level keys only, e.g. "evidence.outlier_flags", "preview.schema".
    4. expected_output_keys must be field names from the agent schema:
       - DataQualityAgent:        issues, overall_quality_score, summary
       - ClinicalSemanticsAgent:  column_roles, clarifying_questions, trial_design_signals
       - UnknownsAgent:           ranked_unknowns, explicit_assumptions, required_documents, summary
       - InsightSynthesisAgent:   key_findings, risks_and_bias_signals,
                                  recommended_next_analyses, required_metadata_or_questions

    ALLOWED KEYS for inputs_used — use ONLY these verbatim:
{allowed_keys}

    Required JSON schema:
    {{
      "dataset_summary": "<one-line description>",
      "steps": [
        {{
          "step_number": 1,
          "name": "<short name>",
          "goal": "<narrow goal>",
          "inputs_used": ["<key from ALLOWED KEYS>", ...],
          "expected_output_keys": ["<field name>", ...],
          "agent_to_call": "<AgentName>"
        }}
      ]
    }}

    DATASET_PREVIEW (JSON):
    {preview}

    DATASET_EVIDENCE (JSON):
    {evidence}
""")


def _build_allowed_keys(preview: dict, evidence: dict) -> list[str]:
    """Return only section-level keys — never raw column names."""
    keys: list[str] = []
    for k in preview:
        keys.append(f"preview.{k}")
    for k in evidence:
        keys.append(f"evidence.{k}")
    return sorted(keys)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        ).strip()
    return text


def _validate_inputs_used(plan: PlanSchema, allowed_keys: list[str]) -> None:
    """Raise ValueError listing every step that contains a disallowed key."""
    allowed_set = set(allowed_keys)
    violations: list[str] = []
    for step in plan.steps:
        for key in step.inputs_used:
            if key not in allowed_set and not (
                key.startswith("preview.") or key.startswith("evidence.")
            ):
                violations.append(
                    f"  Step {step.step_number} ({step.agent_to_call}): "
                    f"disallowed key '{key}'"
                )
    if violations:
        raise ValueError(
            "Planner used disallowed inputs_used keys:\n" + "\n".join(violations)
        )


def _ensure_agent_order(plan: PlanSchema) -> PlanSchema:
    """
    Guarantee canonical agent order:
      DataQualityAgent → ClinicalSemanticsAgent → UnknownsAgent → InsightSynthesisAgent
    Inject missing agents with sensible defaults, then renumber 1..N.
    """
    buckets: dict[str, list[PlanStep]] = {
        "DataQualityAgent": [],
        "ClinicalSemanticsAgent": [],
        "UnknownsAgent": [],
        "InsightSynthesisAgent": [],
        "other": [],
    }
    for s in plan.steps:
        if s.agent_to_call in buckets:
            buckets[s.agent_to_call].append(s)
        else:
            buckets["other"].append(s)

    def _first_or_default(agent: str, default: PlanStep) -> PlanStep:
        return buckets[agent][0] if buckets[agent] else default

    dq_step = _first_or_default(
        "DataQualityAgent",
        PlanStep(
            step_number=1,
            name="Assess Data Quality",
            goal="Identify data quality issues such as outliers, missingness, and imbalance.",
            inputs_used=["evidence.outlier_flags", "evidence.missingness_pct",
                         "evidence.id_duplicates", "evidence.constant_columns",
                         "preview.schema", "preview.head"],
            expected_output_keys=["issues", "overall_quality_score", "summary"],
            agent_to_call="DataQualityAgent",
        ),
    )

    cs_step = _first_or_default(
        "ClinicalSemanticsAgent",
        PlanStep(
            step_number=2,
            name="Interpret Clinical Semantics",
            goal="Infer column roles and identify trial design signals.",
            inputs_used=["preview.schema", "preview.head", "preview.numeric_summary",
                         "evidence.cardinality", "evidence.categorical_distributions",
                         "evidence.top_correlations"],
            expected_output_keys=["column_roles", "clarifying_questions", "trial_design_signals"],
            agent_to_call="ClinicalSemanticsAgent",
        ),
    )

    ua_step = _first_or_default(
        "UnknownsAgent",
        PlanStep(
            step_number=3,
            name="Surface Unknowns and Assumptions",
            goal=(
                "Identify ranked open questions, explicit assumptions the pipeline relies on, "
                "and documents required to resolve ambiguities."
            ),
            inputs_used=[
                "preview.schema", "preview.head",
                "evidence.cardinality", "evidence.categorical_distributions",
                "evidence.outlier_flags", "evidence.top_correlations",
            ],
            expected_output_keys=[
                "ranked_unknowns", "explicit_assumptions",
                "required_documents", "summary",
            ],
            agent_to_call="UnknownsAgent",
        ),
    )

    is_step = _first_or_default(
        "InsightSynthesisAgent",
        PlanStep(
            step_number=4,
            name="Synthesise Final Insights",
            goal=(
                "Consolidate all prior agent outputs into a final structured report "
                "with resolvable evidence citations."
            ),
            inputs_used=[
                "evidence.outlier_flags", "evidence.top_correlations",
                "evidence.missingness_pct", "evidence.categorical_distributions",
                "preview.schema", "preview.numeric_summary",
            ],
            expected_output_keys=[
                "key_findings", "risks_and_bias_signals",
                "recommended_next_analyses", "required_metadata_or_questions",
            ],
            agent_to_call="InsightSynthesisAgent",
        ),
    )

    ordered = [dq_step] + buckets["other"] + [cs_step, ua_step, is_step]

    renumbered = [
        PlanStep(
            step_number=i + 1,
            name=s.name,
            goal=s.goal,
            inputs_used=s.inputs_used,
            expected_output_keys=s.expected_output_keys,
            agent_to_call=s.agent_to_call,
        )
        for i, s in enumerate(ordered)
    ]

    return PlanSchema(dataset_summary=plan.dataset_summary, steps=renumbered)


def call_planner(
    preview: dict,
    evidence: dict,
    provider: ProviderName,
    *,
    retry: bool = True,
) -> PlanSchema:
    client = get_client(provider)
    allowed_keys = _build_allowed_keys(preview, evidence)
    allowed_keys_block = "\n".join(f"      - {k}" for k in allowed_keys)

    user_content = _USER.format(
        allowed_keys=allowed_keys_block,
        preview=json.dumps(preview, indent=2, ensure_ascii=False),
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False),
    )

    req = LLMRequest(system_prompt=_SYSTEM, user_prompt=user_content, payload={})
    resp = client.generate(req)
    raw = _strip_fences(resp.text)

    try:
        data = json.loads(raw)
        plan = PlanSchema.model_validate(data)
        _validate_inputs_used(plan, allowed_keys)
        return _ensure_agent_order(plan)
    except Exception as exc:
        if not retry:
            raise ValueError(
                f"Planner returned invalid output after retry.\nRaw:\n{raw}\nError: {exc}"
            ) from exc

        fix_req = LLMRequest(
            system_prompt=_SYSTEM,
            user_prompt=(
                "Your previous response was rejected.\n"
                "Reasons may include:\n"
                "  - Not valid JSON\n"
                "  - inputs_used contained raw column names (forbidden)\n"
                "  - inputs_used key not in the allowed list\n"
                "  - UnknownsAgent missing or in wrong position\n\n"
                f"Error: {exc}\n\n"
                f"Allowed inputs_used keys (use ONLY these):\n{allowed_keys_block}\n\n"
                f"Previous response:\n{raw}\n\n"
                "Return ONLY corrected JSON. No markdown, no explanation."
            ),
            payload={},
        )
        fix_resp = client.generate(fix_req)
        fixed_raw = _strip_fences(fix_resp.text)
        try:
            data = json.loads(fixed_raw)
        except json.JSONDecodeError:
            from qtrial_backend.agentic.agents import _repair_truncated_json
            data = json.loads(_repair_truncated_json(fixed_raw))
        plan = PlanSchema.model_validate(data)
        _validate_inputs_used(plan, allowed_keys)
        return _ensure_agent_order(plan)
