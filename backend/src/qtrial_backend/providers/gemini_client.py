"""
Google Gemini LLMClient implementation.

Input:  LLMRequest or chat message list + Gemini tool schemas.
Output: LLMResponse (text) or ChatResponse (text + tool_calls).
Does:   wraps google.genai generate_content with GenerateContentConfig; groups
        consecutive function_response parts into one user turn; supports API
        key rotation and exponential-backoff retry on rate-limit errors.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

from google import genai

from qtrial_backend.config import settings
from qtrial_backend.core.types import (
    ChatResponse,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
)
from qtrial_backend.providers.base import LLMClient

_MAX_RETRIES = 4       # retries per key on RPM throttle
_BASE_BACKOFF = 10.0   # seconds before first RPM retry

# Thread-local so the SSE layer can inject an emit callback without
# changing any function signatures.
_tl = threading.local()


def set_thread_emit(fn: Callable | None) -> None:
    """Register an emit callback for rate-limit progress events on this thread."""
    _tl.emit = fn


def _thread_emit(event: dict) -> None:
    fn = getattr(_tl, "emit", None)
    if fn is not None:
        try:
            fn(event)
        except Exception:
            pass


def _is_daily_quota(exc: Exception) -> bool:
    """True when the error is the per-project DAILY cap (not a per-minute throttle)."""
    text = str(exc)
    return "GenerateRequestsPerDayPerProject" in text or "FreeTier" in text


def _retry_delay_from_error(exc: Exception) -> float | None:
    """Parse the retryDelay value from a 429 RPM error, e.g. '34s' → 34.0.
    Returns None if the error is NOT a retriable rate-limit."""
    text = str(exc)
    m = re.search(r"retryDelay['\"]:\s*['\"]([\\d.]+)s['\"]", text)
    if m:
        return float(m.group(1)) + 2.0
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return _BASE_BACKOFF
    return None


def _to_gemini_contents(messages: list[Message]) -> list[Any]:
    """Convert Message list → Gemini Content list for generate_content."""
    from google.genai import types

    contents: list[Any] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "user" and msg.content is not None:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.content)],
                )
            )
            i += 1

        elif msg.role == "assistant":
            parts: list[Any] = []
            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(
                            name=tc.name,
                            args=tc.arguments,
                        )
                    )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            i += 1

        elif msg.role == "tool":
            # Group ALL consecutive tool messages into one user turn
            fn_parts: list[Any] = []
            while i < len(messages) and messages[i].role == "tool":
                tr = messages[i].tool_result
                if tr is not None:
                    try:
                        response_data: Any = json.loads(tr.content)
                    except Exception:
                        response_data = {"result": tr.content}
                    fn_parts.append(
                        types.Part.from_function_response(
                            name=tr.name,
                            response=response_data,
                        )
                    )
                i += 1
            if fn_parts:
                contents.append(types.Content(role="user", parts=fn_parts))

        else:
            i += 1

    return contents


def _parse_gemini_chat_response(response: Any, model: str) -> ChatResponse:
    """Convert a Gemini generate_content response → ChatResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for idx, part in enumerate(candidate.content.parts):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"{fc.name}_{idx}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

    return ChatResponse(
        provider="gemini",
        model=model,
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        stop_reason="tool_use" if tool_calls else "end_turn",
    )


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        keys = settings.gemini_api_keys
        if not keys:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.model = settings.gemini_model
        # One genai.Client per key; rotate on daily-quota exhaustion
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._key_index = 0

    @property
    def _client(self) -> genai.Client:
        return self._clients[self._key_index]

    def _rotate_key(self) -> bool:
        """Advance to the next key. Returns True if a new key is available."""
        if self._key_index + 1 < len(self._clients):
            self._key_index += 1
            return True
        return False

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        prompt = (
            f"{req.system_prompt}\n\n"
            f"{req.user_prompt}\n\n"
            f"DATASET_PREVIEW_PAYLOAD (JSON):\n{payload_json}"
        )

        # Outer loop: rotate through keys on daily-quota exhaustion
        while True:
            last_exc: Exception | None = None

            # Inner loop: retry with backoff on per-minute throttle
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = self._client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                    )
                    text = getattr(resp, "text", "") or ""
                    return LLMResponse(provider="gemini", model=self.model, text=text)

                except Exception as exc:
                    last_exc = exc

                    # Daily quota hit — rotate to next key immediately
                    if _is_daily_quota(exc):
                        key_num = self._key_index + 1
                        if self._rotate_key():
                            msg = (
                                f"Gemini key {key_num} hit daily quota. "
                                f"Switching to key {self._key_index + 1}"
                                f"/{len(self._clients)}\u2026"
                            )
                            print(f"[GeminiClient] {msg}")
                            _thread_emit({
                                "type": "progress",
                                "stage": "key_rotation",
                                "message": msg,
                            })
                            break  # break inner → retry outer with new key
                        else:
                            raise RuntimeError(
                                f"All {len(self._clients)} Gemini key(s) have hit the "
                                "daily quota. Add more keys to GEMINI_API_KEY "
                                "(comma-separated) or wait until midnight UTC."
                            ) from exc

                    # Per-minute throttle — wait and retry same key
                    wait = _retry_delay_from_error(exc)
                    if wait is None:
                        raise  # non-rate-limit error — propagate immediately
                    if attempt == _MAX_RETRIES:
                        break
                    wait = min(wait * (2 ** (attempt - 1)), 120.0)
                    msg = (
                        f"Gemini rate-limit (attempt {attempt}/{_MAX_RETRIES}). "
                        f"Waiting {wait:.0f}s\u2026"
                    )
                    print(f"[GeminiClient] {msg}")
                    _thread_emit({
                        "type": "progress",
                        "stage": "rate_limit",
                        "message": msg,
                    })
                    time.sleep(wait)

            else:
                # Inner for-loop exhausted without a key-rotation break
                raise RuntimeError(
                    f"Gemini API rate-limit persisted after {_MAX_RETRIES} attempts. "
                    f"Last error: {last_exc}"
                ) from last_exc

    def chat(
        self,
        messages: list[Message],
        tools: list | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        """Multi-turn chat with optional tool calling for AgentLoop."""
        from google.genai import types
        from qtrial_backend.tools.converter import to_gemini_tools

        contents = _to_gemini_contents(messages)

        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = to_gemini_tools(tools)
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # Outer loop: rotate through keys on daily-quota exhaustion
        while True:
            last_exc: Exception | None = None

            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    if config is not None:
                        resp = self._client.models.generate_content(
                            model=self.model,
                            contents=contents,
                            config=config,
                        )
                    else:
                        resp = self._client.models.generate_content(
                            model=self.model,
                            contents=contents,
                        )
                    return _parse_gemini_chat_response(resp, self.model)

                except Exception as exc:
                    last_exc = exc

                    if _is_daily_quota(exc):
                        key_num = self._key_index + 1
                        if self._rotate_key():
                            msg = (
                                f"Gemini key {key_num} hit daily quota. "
                                f"Switching to key {self._key_index + 1}"
                                f"/{len(self._clients)}\u2026"
                            )
                            print(f"[GeminiClient.chat] {msg}")
                            _thread_emit({
                                "type": "progress",
                                "stage": "key_rotation",
                                "message": msg,
                            })
                            break
                        else:
                            raise RuntimeError(
                                f"All {len(self._clients)} Gemini key(s) have hit the "
                                "daily quota. Add more keys to GEMINI_API_KEY "
                                "(comma-separated) or wait until midnight UTC."
                            ) from exc

                    wait = _retry_delay_from_error(exc)
                    if wait is None:
                        raise
                    if attempt == _MAX_RETRIES:
                        break
                    wait = min(wait * (2 ** (attempt - 1)), 120.0)
                    msg = (
                        f"Gemini rate-limit (attempt {attempt}/{_MAX_RETRIES}). "
                        f"Waiting {wait:.0f}s\u2026"
                    )
                    print(f"[GeminiClient.chat] {msg}")
                    _thread_emit({
                        "type": "progress",
                        "stage": "rate_limit",
                        "message": msg,
                    })
                    time.sleep(wait)

            else:
                raise RuntimeError(
                    f"Gemini API rate-limit persisted after {_MAX_RETRIES} attempts. "
                    f"Last error: {last_exc}"
                ) from last_exc
