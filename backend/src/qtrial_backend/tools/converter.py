"""
Tool schema converter — translates RegisteredTool objects to provider formats.

Input:  list[RegisteredTool] from ToolRegistry.
Output: provider-specific tool schema lists:
          to_openai_tools()  → list[dict] for OpenAI / OpenRouter
          to_claude_tools()  → list[dict] for Anthropic Claude
          to_gemini_tools()  → list for Google Gemini
Does:   each provider has a slightly different function-calling schema format;
        this module isolates those differences so AgentLoop stays provider-agnostic.
"""
from __future__ import annotations

from typing import Any

from qtrial_backend.tools.registry import RegisteredTool


def _clean_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic-specific keys that providers may not understand."""
    schema = {k: v for k, v in schema.items() if k not in ("title", "$defs")}
    if "properties" in schema:
        schema["properties"] = {
            prop_name: {k: v for k, v in prop_schema.items() if k != "title"}
            for prop_name, prop_schema in schema["properties"].items()
        }
    return schema


# ── OpenAI Chat Completions ──────────────────────────────────────────


def to_openai_tools(tools: list[RegisteredTool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": _clean_json_schema(
                    t.params_model.model_json_schema()
                ),
            },
        }
        for t in tools
    ]


# ── Anthropic Messages API ───────────────────────────────────────────


def to_claude_tools(tools: list[RegisteredTool]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": _clean_json_schema(
                t.params_model.model_json_schema()
            ),
        }
        for t in tools
    ]


# ── Google Gemini ────────────────────────────────────────────────────

_GEMINI_TYPE_MAP = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _json_schema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema dict to Gemini's expected format."""
    converted: dict[str, Any] = {}
    if "type" in schema:
        converted["type"] = _GEMINI_TYPE_MAP.get(schema["type"], schema["type"])
    if "description" in schema:
        converted["description"] = schema["description"]
    if "properties" in schema:
        converted["properties"] = {
            k: _json_schema_to_gemini(v)
            for k, v in schema["properties"].items()
        }
    if "required" in schema:
        converted["required"] = schema["required"]
    if "enum" in schema:
        converted["enum"] = schema["enum"]
    if "items" in schema:
        converted["items"] = _json_schema_to_gemini(schema["items"])
    return converted


def to_gemini_tools(tools: list[RegisteredTool]) -> list:
    from google.genai import types

    declarations = []
    for t in tools:
        schema = _clean_json_schema(t.params_model.model_json_schema())
        declarations.append(
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=_json_schema_to_gemini(schema),
            )
        )
    return [types.Tool(function_declarations=declarations)]
