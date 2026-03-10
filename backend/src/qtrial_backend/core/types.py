from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any, Dict

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "gemini", "claude", "openrouter"]


# ── Legacy single-shot types (unchanged) ──────────────────────────────


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


# ── Agentic flow types ────────────────────────────────────────────────


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool, sent back to the LLM."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """Provider-agnostic conversation message for the agent loop."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None


class ChatResponse(BaseModel):
    """Normalised single-turn response from any provider's chat() method."""

    provider: ProviderName
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error"]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class AgentResponse(BaseModel):
    """Final output of the complete agent loop."""

    provider: ProviderName
    model: str
    text: str
    tool_calls_made: int
    iterations: int
    tool_log: list[Dict[str, Any]] = Field(default_factory=list)
