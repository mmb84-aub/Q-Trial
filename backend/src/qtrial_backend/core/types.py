from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any, Dict

ProviderName = Literal["openai", "gemini", "claude", "openrouter"]


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    provider: ProviderName
    model: str
    text: str
