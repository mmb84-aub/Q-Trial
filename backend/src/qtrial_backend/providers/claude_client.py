from __future__ import annotations

import json
from typing import Any

import anthropic

from qtrial_backend.config import settings
from qtrial_backend.core.types import (
    LLMRequest,
    LLMResponse,
    Message,
    ChatResponse,
    ToolCall,
)
from qtrial_backend.providers.base import LLMClient
from qtrial_backend.tools.converter import to_claude_tools
from qtrial_backend.tools.registry import RegisteredTool


class ClaudeClient(LLMClient):
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    # ── Legacy single-shot (unchanged) ────────────────────────────────

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=req.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        req.user_prompt
                        + "\n\nDATASET_PREVIEW_PAYLOAD (JSON):\n"
                        + payload_json
                    ),
                }
            ],
        )

        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        return LLMResponse(
            provider="claude", model=self.model, text="\n".join(text_parts)
        )

    # ── Agentic chat (Messages API with tools) ───────────────────────

    def chat(
        self,
        messages: list[Message],
        tools: list[RegisteredTool] | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        anth_messages = self._build_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anth_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_claude_tools(tools)

        resp = self.client.messages.create(**kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        stop_map = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
        }
        stop_reason = stop_map.get(resp.stop_reason or "", "error")

        return ChatResponse(
            provider="claude",
            model=self.model,
            content=content_text or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert canonical messages to Anthropic format.

        Claude requires tool results as tool_result content blocks inside
        a ``role: "user"`` message, grouped after the assistant turn.
        """
        result: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "user":
                result.append({"role": "user", "content": msg.content or ""})
                i += 1

            elif msg.role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                result.append({"role": "assistant", "content": content_blocks})
                i += 1

            elif msg.role == "tool" and msg.tool_result:
                tool_result_blocks: list[dict[str, Any]] = []
                while i < len(messages) and messages[i].role == "tool":
                    tr = messages[i].tool_result
                    if tr:
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                        )
                    i += 1
                result.append({"role": "user", "content": tool_result_blocks})

            else:
                i += 1

        return result
