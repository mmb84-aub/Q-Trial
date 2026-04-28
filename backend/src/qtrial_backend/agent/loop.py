"""
AgentLoop — iterative LLM ↔ tool execution engine (Stage 4).

Input:  AgentContext (DataFrame, preview, evidence), system prompt, list of
        RegisteredTool objects, LLMClient, max_iterations.
Output: Markdown analysis_report (str) + tool_log (list[dict])
Does:   runs a while-loop where the LLM requests tool calls, tools execute
        against the DataFrame, results feed back to the LLM, and the loop
        terminates when the LLM produces a final text response or max_iterations
        is reached. Annotates confidence warnings post-loop.
"""
from __future__ import annotations

import json

from rich.console import Console
from rich.status import Status

from qtrial_backend.agent.context import AgentContext
from qtrial_backend.core.types import (
    AgentResponse,
    ChatResponse,
    Message,
    ToolResult,
)
from qtrial_backend.providers.base import LLMClient
from qtrial_backend.tools.registry import RegisteredTool, ToolRegistry

console = Console()


class AgentLoop:
    """While-loop orchestrator that drives the LLM ↔ tool conversation."""

    def __init__(
        self,
        client: LLMClient,
        tools: list[RegisteredTool],
        system_prompt: str,
        max_iterations: int = 25,
        max_consecutive_errors: int = 3,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_consecutive_errors = max_consecutive_errors
        self.verbose = verbose

    def run(self, initial_message: str, context: AgentContext) -> AgentResponse:
        messages: list[Message] = [
            Message(role="user", content=initial_message),
        ]

        iterations = 0
        total_tool_calls = 0
        consecutive_errors = 0
        tool_log: list[dict] = []

        with Status(
            "[bold green]Agent is analysing...", console=console
        ) as status:
            while iterations < self.max_iterations:
                iterations += 1
                status.update(
                    f"[bold green]Iteration {iterations}/{self.max_iterations}..."
                )

                # ── LLM call ──────────────────────────────────────────
                response: ChatResponse = self.client.chat(
                    messages=messages,
                    tools=self.tools,
                    system=self.system_prompt,
                )

                # ── No tool calls → final answer ─────────────────────
                if not response.has_tool_calls:
                    return AgentResponse(
                        provider=response.provider,
                        model=response.model,
                        text=response.content or "(No response text)",
                        tool_calls_made=total_tool_calls,
                        iterations=iterations,
                        tool_log=tool_log,
                    )

                # ── Record assistant message with tool requests ───────
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                # ── Execute each tool call ────────────────────────────
                for tc in response.tool_calls:
                    total_tool_calls += 1
                    status.update(
                        f"[bold cyan]Running tool: {tc.name} "
                        f"({total_tool_calls} calls so far)..."
                    )

                    # ── Deduplication check ────────────────────────────
                    cache_key = (
                        f"{tc.name}::"
                        + json.dumps(tc.arguments, sort_keys=True, default=str)
                    )
                    if cache_key in context._call_cache:
                        cached_result = context._call_cache[cache_key]
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=cached_result,
                            is_error=False,
                        )
                        if self.verbose:
                            console.print(
                                f"  [dim]Tool: {tc.name}({tc.arguments}) "
                                f"-> CACHED[/dim]"
                            )
                        messages.append(
                            Message(role="tool", tool_result=tool_result)
                        )
                        tool_log.append(
                            {
                                "iteration": iterations,
                                "tool": tc.name,
                                "args": tc.arguments,
                                "is_error": False,
                                "cached": True,
                            }
                        )
                        continue

                    try:
                        result_str = ToolRegistry.execute(
                            tc.name, tc.arguments, context
                        )
                        context._call_cache[cache_key] = result_str
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=result_str,
                            is_error=False,
                        )
                        consecutive_errors = 0
                    except Exception as exc:
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=json.dumps({"error": str(exc)}),
                            is_error=True,
                        )
                        consecutive_errors += 1

                    messages.append(
                        Message(role="tool", tool_result=tool_result)
                    )

                    if not tool_result.is_error and tc.name != "retrieve_evidence":
                        try:
                            context.index_tool_result(
                                tool_name=tc.name,
                                arguments=tc.arguments,
                                result_text=tool_result.content,
                                is_error=False,
                            )
                        except Exception:
                            # Retrieval indexing failures must never block analysis flow.
                            pass

                    tool_log.append(
                        {
                            "iteration": iterations,
                            "tool": tc.name,
                            "args": tc.arguments,
                            "result": tool_result.content,
                            "is_error": tool_result.is_error,
                            "cached": False,
                        }
                    )

                    if self.verbose:
                        if tool_result.is_error:
                            try:
                                err_detail = json.loads(tool_result.content).get("error", tool_result.content)
                            except Exception:
                                err_detail = tool_result.content
                            console.print(
                                f"  [red]Tool: {tc.name}({tc.arguments}) "
                                f"-> ERROR: {err_detail}[/red]"
                            )
                        else:
                            console.print(
                                f"  [dim]Tool: {tc.name}({tc.arguments}) "
                                f"-> OK[/dim]"
                            )

                # ── Too many consecutive errors → force conclusion ────
                if consecutive_errors >= self.max_consecutive_errors:
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                "Multiple consecutive tool errors occurred. "
                                "Please provide your analysis based on the "
                                "data gathered so far."
                            ),
                        )
                    )
                    final = self.client.chat(
                        messages=messages, tools=self.tools, system=self.system_prompt
                    )
                    return AgentResponse(
                        provider=final.provider,
                        model=final.model,
                        text=final.content or "(No response after errors)",
                        tool_calls_made=total_tool_calls,
                        iterations=iterations,
                        tool_log=tool_log,
                    )

        # ── Max iterations reached → force conclusion ─────────────────
        messages.append(
            Message(
                role="user",
                content=(
                    "You have reached the maximum number of analysis "
                    "iterations. Please synthesise your findings now and "
                    "produce the final report."
                ),
            )
        )
        final = self.client.chat(messages=messages, tools=self.tools, system=self.system_prompt)
        return AgentResponse(
            provider=final.provider,
            model=final.model,
            text=final.content or "(No response at max iterations)",
            tool_calls_made=total_tool_calls,
            iterations=iterations + 1,
            tool_log=tool_log,
        )
