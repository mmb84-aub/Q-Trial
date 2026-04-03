from __future__ import annotations

import json
from typing import Any

import anthropic

from qtrial_backend.config import settings
from qtrial_backend.core.types import ChatResponse, LLMRequest, LLMResponse, Message, ToolCall
from qtrial_backend.providers.base import LLMClient


class ClaudeClient(LLMClient):
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

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

        return LLMResponse(provider="claude", model=self.model, text="\n".join(text_parts))

    def chat(
        self,
        messages: list[Message],
        tools: list | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        """Multi-turn chat with optional tool calling for AgentLoop."""
        from qtrial_backend.tools.converter import to_claude_tools

        ant_messages: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "user":
                ant_messages.append({"role": "user", "content": msg.content or ""})
                i += 1

            elif msg.role == "assistant":
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                ant_messages.append({"role": "assistant", "content": content})
                i += 1

            elif msg.role == "tool":
                # Group consecutive tool results into a single user turn
                tool_content: list[dict[str, Any]] = []
                while i < len(messages) and messages[i].role == "tool":
                    tr = messages[i].tool_result
                    if tr is not None:
                        tool_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                        )
                    i += 1
                if tool_content:
                    ant_messages.append({"role": "user", "content": tool_content})

            else:
                i += 1

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": ant_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_claude_tools(tools)

        resp = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return ChatResponse(
            provider="claude",
            model=self.model,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
        )
