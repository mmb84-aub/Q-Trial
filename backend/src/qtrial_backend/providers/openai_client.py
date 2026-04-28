"""
OpenAI LLMClient implementation.

Input:  LLMRequest or chat message list + OpenAI tool schemas.
Output: LLMResponse (text) or ChatResponse (text + tool_calls).
Does:   wraps openai.chat.completions.create; implements generate() and chat()
        from LLMClient base class for use in Stage 4 agent loop and Stages 5/7.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from qtrial_backend.config import settings
from qtrial_backend.core.types import ChatResponse, LLMRequest, LLMResponse, Message, ToolCall
from qtrial_backend.providers.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        resp = self.client.responses.create(
            model=self.model,
            input=[
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
        text = getattr(resp, "output_text", None) or ""
        return LLMResponse(provider="openai", model=self.model, text=text)

    def chat(
        self,
        messages: list[Message],
        tools: list | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        """Multi-turn chat with optional tool calling for AgentLoop."""
        from qtrial_backend.tools.converter import to_openai_tools

        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "user":
                oai_messages.append({"role": "user", "content": msg.content or ""})
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

        kwargs: dict[str, Any] = {"model": self.model, "messages": oai_messages}
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
        stop_reason = stop_map.get(choice.finish_reason or "", "end_turn")

        return ChatResponse(
            provider="openai",
            model=self.model,
            content=choice.message.content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )
