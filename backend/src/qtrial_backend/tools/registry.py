from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from qtrial_backend.agent.context import AgentContext

MAX_TOOL_RESULT_CHARS = 4_000


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for spec-compliant JSON.

    json.dumps() by default serialises NaN/Inf as the literals NaN/Infinity,
    which are valid Python but not valid JSON — Gemini (and strict parsers)
    reject them.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@dataclass
class RegisteredTool:
    name: str
    description: str
    params_model: type[BaseModel]
    func: Callable[..., dict[str, Any]]
    category: str  # "stats" or "literature"


class ToolRegistry:
    """Singleton registry of all available tools."""

    _tools: dict[str, RegisteredTool] = {}

    @classmethod
    def register(cls, tool: RegisteredTool) -> None:
        if tool.name in cls._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> RegisteredTool:
        if name not in cls._tools:
            raise KeyError(
                f"Unknown tool: '{name}'. Available: {list(cls._tools.keys())}"
            )
        return cls._tools[name]

    @classmethod
    def all_tools(cls) -> list[RegisteredTool]:
        return list(cls._tools.values())

    @classmethod
    def by_category(cls, category: str) -> list[RegisteredTool]:
        return [t for t in cls._tools.values() if t.category == category]

    @classmethod
    def execute(
        cls,
        name: str,
        arguments: dict[str, Any],
        context: AgentContext,
    ) -> str:
        """Validate arguments via Pydantic, call the tool, return JSON string."""
        tool = cls.get(name)
        params = tool.params_model.model_validate(arguments)
        result = tool.func(params, context)
        result_str = json.dumps(
            _sanitize_for_json(result), default=str, ensure_ascii=False
        )
        if len(result_str) > MAX_TOOL_RESULT_CHARS:
            result_str = (
                result_str[:MAX_TOOL_RESULT_CHARS]
                + f"\n... (truncated from {len(result_str)} chars)"
            )
        return result_str


def tool(
    name: str,
    description: str,
    params_model: type[BaseModel],
    category: str = "stats",
) -> Callable:
    """Decorator that registers a function as an agent tool."""

    def decorator(func: Callable) -> Callable:
        registered = RegisteredTool(
            name=name,
            description=description,
            params_model=params_model,
            func=func,
            category=category,
        )
        ToolRegistry.register(registered)
        return func

    return decorator
