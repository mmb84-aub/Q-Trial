"""
Core type definitions shared across the pipeline.

Input:  n/a (type module only).
Output: dataclasses and type aliases used by providers, AgentLoop, and orchestrator.
Does:   defines LLMRequest, LLMResponse, ChatResponse, Message, AgentResponse,
        and the ProviderName literal — the lingua franca of the Q-Trial pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any, Dict, List, Optional

ProviderName = Literal["openai", "gemini", "claude", "openrouter", "bedrock"]


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


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None


@dataclass
class ChatResponse:
    provider: str
    model: str
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class AgentResponse:
    provider: str
    model: str
    text: str
    tool_calls_made: int
    iterations: int
    tool_log: List[Dict[str, Any]] = field(default_factory=list)
