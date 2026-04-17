"""
OpenRouter LLMClient implementation.

Input:  LLMRequest or chat message list + OpenAI-compatible tool schemas.
Output: LLMResponse (text) or ChatResponse (text + tool_calls).
Does:   wraps the OpenRouter REST API (OpenAI-compatible) with API key rotation
        and thread-safe key cycling; supports any model available on OpenRouter.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from openai import OpenAI

from qtrial_backend.config import settings
from qtrial_backend.core.types import (
    LLMRequest,
    LLMResponse,
    Message,
    ChatResponse,
    ToolCall,
)
from qtrial_backend.providers.base import LLMClient
from qtrial_backend.tools.converter import to_openai_tools
from qtrial_backend.tools.registry import RegisteredTool

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Thread-local model override — set once per pipeline thread so all agents
# in that thread use the same user-selected model.
_tl = threading.local()


def set_thread_model(model: str | None) -> None:
    """Set the OpenRouter model for the current thread (None = use env default)."""
    _tl.model = model


def get_thread_model() -> str | None:
    return getattr(_tl, "model", None)


class OpenRouterClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )
        # Priority: constructor arg > thread-local > env default
        self.model = model or get_thread_model() or settings.openrouter_model

    # ── Legacy single-shot ────────────────────────────────────────────

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=settings.openrouter_max_tokens,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {
                    "role": "user",
                    "content": (
                        req.user_prompt
                        + "\n\nDATASET_PREVIEW_PAYLOAD (JSON):\n"
                        + payload_json
                    ),
                },
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(provider="openrouter", model=self.model, text=text)

    # ── Agentic chat (OpenAI-compatible) ──────────────────────────────

    def chat(
        self,
        messages: list[Message],
        tools: list[RegisteredTool] | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "user":
                oai_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                oai_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    oai_msg["content"] = msg.content
                if msg.tool_calls:
                    oai_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                oai_messages.append(oai_msg)
            elif msg.role == "tool" and msg.tool_result:
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                    }
                )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": settings.openrouter_max_tokens,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_map.get(choice.finish_reason or "", "error")

        return ChatResponse(
            provider="openrouter",
            model=self.model,
            content=choice.message.content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )
