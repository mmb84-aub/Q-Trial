from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from google import genai
from google.genai import types

from qtrial_backend.config import settings
from qtrial_backend.core.types import (
    LLMRequest,
    LLMResponse,
    Message,
    ChatResponse,
    ToolCall,
)
from qtrial_backend.providers.base import LLMClient
from qtrial_backend.tools.converter import to_gemini_tools
from qtrial_backend.tools.registry import RegisteredTool


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    # ── Legacy single-shot (unchanged) ────────────────────────────────

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        prompt = (
            f"{req.system_prompt}\n\n"
            f"{req.user_prompt}\n\n"
            f"DATASET_PREVIEW_PAYLOAD (JSON):\n{payload_json}"
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = getattr(resp, "text", "") or ""
        return LLMResponse(provider="gemini", model=self.model, text=text)

    # ── Agentic chat (google-genai with tool calling) ─────────────────

    _MAX_RETRIES = 3
    _DEFAULT_WAIT = 60.0  # seconds to wait when delay cannot be parsed

    def chat(
        self,
        messages: list[Message],
        tools: list[RegisteredTool] | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        contents = self._build_contents(messages)

        config = types.GenerateContentConfig()
        if system:
            config.system_instruction = system
        if tools:
            config.tools = to_gemini_tools(tools)

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break  # success — exit retry loop
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_server_error = "ServerError" in type(exc).__name__ or "500" in err_str or "503" in err_str
                if (is_rate_limit or is_server_error) and attempt < self._MAX_RETRIES:
                    if is_rate_limit:
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                        wait = float(match.group(1)) if match else self._DEFAULT_WAIT
                        reason = "Rate limit hit"
                    else:
                        wait = min(5.0 * (attempt + 1), 30.0)  # 5s, 10s, 15s back-off
                        reason = "Server error (5xx)"
                    print(
                        f"[Gemini] {reason}. "
                        f"Waiting {wait:.0f}s before retry "
                        f"({attempt + 1}/{self._MAX_RETRIES})..."
                    )
                    time.sleep(wait)
                else:
                    raise

        content_text = ""
        tool_calls: list[ToolCall] = []

        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts or []:
                if part.text:
                    content_text += part.text
                elif part.function_call:
                    tool_calls.append(
                        ToolCall(
                            id=str(uuid.uuid4()),
                            name=part.function_call.name or "",
                            arguments=(
                                dict(part.function_call.args)
                                if part.function_call.args
                                else {}
                            ),
                        )
                    )

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return ChatResponse(
            provider="gemini",
            model=self.model,
            content=content_text or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_contents(messages: list[Message]) -> list[types.Content]:
        """Convert canonical messages to Gemini Content objects."""
        contents: list[types.Content] = []
        for msg in messages:
            if msg.role == "user":
                contents.append(
                    types.Content(
                        role="user", parts=[types.Part(text=msg.content or "")]
                    )
                )
            elif msg.role == "assistant":
                parts: list[types.Part] = []
                if msg.content:
                    parts.append(types.Part(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=tc.name, args=tc.arguments
                                )
                            )
                        )
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool" and msg.tool_result:
                try:
                    response_data = json.loads(msg.tool_result.content)
                except json.JSONDecodeError:
                    response_data = {"result": msg.tool_result.content}

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.tool_result.name,
                                    response=response_data,
                                )
                            )
                        ],
                    )
                )
        return contents
