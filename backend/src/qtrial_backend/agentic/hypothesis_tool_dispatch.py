"""
Task 4B — Hypothesis-driven deterministic tool dispatch.

After the LLM generates hypotheses with ToolDispatchRequest objects,
this module maps each request to an actual registered stats tool,
executes it against the DataFrame, and returns typed ToolDispatchResult
objects that are:
  - stored in ReasoningState.dispatched_tool_results (for the audit trail)
  - rendered as a text block injected into the final InsightSynthesisAgent
    re-run, grounding synthesis in real empirical numbers.

Tool-type → registered tool mapping
-------------------------------------
baseline_balance   → baseline_balance     (SMD Table-1 per treatment arm)
survival_analysis  → survival_analysis    (Kaplan-Meier + log-rank)
missing_by_group   → missing_data_patterns (missingness per column)
                     + group_by_summary   (aggregates showing NA counts)
group_statistics   → group_by_summary     (mean/median/SD per group)
distribution_check → distribution_info    (per-column, up to 3 columns)
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

# Import triggers @tool decorator registration for every stats tool
import qtrial_backend.tools.stats  # noqa: F401

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.agentic.schemas import ToolDispatchRequest, ToolDispatchResult
from qtrial_backend.tools.registry import ToolRegistry

_MAX_DISPATCHES = 4   # cap so a bad LLM can't generate 50 tool calls
_MAX_RESULT_CHARS = 2_000  # truncation for each result in the text block


# ── Argument builder ──────────────────────────────────────────────────────────

def _build_args(
    req: ToolDispatchRequest,
    df_columns: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Map one ToolDispatchRequest → list of (tool_name, args) pairs.
    Returns [] when the mapping is impossible (missing required columns).
    Each type may produce 1-3 actual tool calls (e.g. distribution_check
    runs one call per column).
    """
    valid_cols = [c for c in req.columns if c in df_columns]
    group = req.group_column if req.group_column in df_columns else None

    if req.tool_type == "baseline_balance":
        if not group or not valid_cols:
            return []
        return [("baseline_balance", {
            "treatment_column": group,
            "baseline_columns": valid_cols,
        })]

    if req.tool_type == "survival_analysis":
        if len(valid_cols) < 2:
            return []
        return [("survival_analysis", {
            "time_column": valid_cols[0],
            "event_column": valid_cols[1],
            **({"group_column": group} if group else {}),
        })]

    if req.tool_type == "missing_by_group":
        calls: list[tuple[str, dict[str, Any]]] = []
        if valid_cols:
            calls.append(("missing_data_patterns", {"columns": valid_cols}))
        if group and valid_cols:
            # Show count of non-nulls per group as a proxy for missing-by-group
            calls.append(("group_by_summary", {
                "group_columns": [group],
                "target_columns": valid_cols[:4],
                "aggregations": ["count"],
            }))
        return calls

    if req.tool_type == "group_statistics":
        if not group or not valid_cols:
            return []
        return [("group_by_summary", {
            "group_columns": [group],
            "target_columns": valid_cols[:6],
            "aggregations": ["mean", "median", "count", "std"],
        })]

    if req.tool_type == "distribution_check":
        if not valid_cols:
            return []
        # distribution_info takes one column at a time; cap at 3
        return [
            ("distribution_info", {"column": col})
            for col in valid_cols[:3]
        ]

    return []


# ── Main dispatcher ───────────────────────────────────────────────────────────

def run_tool_dispatch(
    requests: list[ToolDispatchRequest],
    df: pd.DataFrame,
) -> list[ToolDispatchResult]:
    """
    Execute up to _MAX_DISPATCHES hypothesis-driven tool calls against *df*.

    Returns one ToolDispatchResult per request (with error set when the
    call fails or the mapping is impossible).  Results include a stable
    ``citation_alias`` so InsightSynthesisAgent can cite them.
    """
    ctx = AgentContext(dataframe=df, dataset_name="hypothesis_dispatch")
    results: list[ToolDispatchResult] = []
    alias_index = 0

    for req in requests[:_MAX_DISPATCHES]:
        call_specs = _build_args(req, ctx.column_names)
        alias = f"dispatched[{alias_index}]"
        alias_index += 1

        if not call_specs:
            results.append(ToolDispatchResult(
                request=req,
                tool_called="(skipped)",
                args_used={},
                error=(
                    f"Cannot map tool_type='{req.tool_type}' — "
                    f"required columns {req.columns!r} or "
                    f"group_column {req.group_column!r} not found in dataset."
                ),
                citation_alias=alias,
            ))
            continue

        # For types that expand to multiple calls (distribution_check / missing_by_group),
        # we merge all sub-results under one ToolDispatchResult.
        combined: dict[str, Any] = {}
        last_tool = call_specs[0][0]
        last_args = call_specs[0][1]
        first_error: str | None = None

        for tool_name, args in call_specs:
            try:
                result_str = ToolRegistry.execute(tool_name, args, ctx)
                sub = json.loads(result_str)
                # Merge under sub-key = tool_name (avoids key collisions)
                key = f"{tool_name}__{len(combined)}"
                combined[key] = sub
                last_tool = tool_name
                last_args = args
            except Exception as exc:
                if first_error is None:
                    first_error = f"{tool_name}: {exc}"

        results.append(ToolDispatchResult(
            request=req,
            tool_called=last_tool,
            args_used=last_args,
            result=combined if combined else None,
            error=first_error,
            citation_alias=alias,
        ))

    return results


# ── Formatting for agents ─────────────────────────────────────────────────────

def format_dispatch_results_for_agents(
    results: list[ToolDispatchResult],
) -> str:
    """
    Render dispatch results as a compact, human-readable text block that
    can be injected into the InsightSynthesisAgent prompt so the agent
    can cite empirical findings using ``dispatched[i]`` aliases.
    """
    if not results:
        return ""

    lines: list[str] = [
        "=== HYPOTHESIS-DRIVEN TOOL RESULTS ===",
        "Cite each result via its alias, e.g. dispatched[0].\n",
    ]

    for r in results:
        lines.append(
            f"[{r.citation_alias}]  tool={r.tool_called}"
            f"  hypothesis={r.request.hypothesis_id}"
            f"  priority={r.request.priority}"
        )
        lines.append(f"  rationale: {r.request.rationale}")

        if r.error:
            lines.append(f"  STATUS: ERROR — {r.error}")
        else:
            compact = json.dumps(r.result, default=str, ensure_ascii=False)
            if len(compact) > _MAX_RESULT_CHARS:
                compact = compact[:_MAX_RESULT_CHARS] + "…(truncated)"
            lines.append(f"  result: {compact}")
        lines.append("")

    lines.append("=== END TOOL RESULTS ===")
    return "\n".join(lines)
