"""
AWS Bedrock provider — uses the Converse API (supports tool use on all major models).

Supported model IDs (set AWS_BEDROCK_MODEL in .env):
  us.anthropic.claude-3-5-sonnet-20241022-v2:0  (default — cross-region profile)
  us.anthropic.claude-3-5-haiku-20241022-v1:0   (fast + cheap)
  us.amazon.nova-pro-v1:0
  us.amazon.nova-lite-v1:0
  us.meta.llama3-3-70b-instruct-v1:0
  mistral.mistral-large-2402-v1:0               (on-demand, no prefix needed)

NOTE: Newer Claude and Nova models require cross-region inference profile IDs
(prefixed with us./eu./ap.). The client auto-adds the prefix when needed.

Credentials — any of the standard boto3 auth methods work:
  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_REGION in .env
  - IAM role / instance profile (no keys needed)
  - AWS SSO / named profile (set AWS_PROFILE in .env)
"""
from __future__ import annotations

import json
import threading
import uuid
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from qtrial_backend.config import settings
from qtrial_backend.core.types import (
    ChatResponse,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
)
from qtrial_backend.providers.base import LLMClient
from qtrial_backend.tools.registry import RegisteredTool

# Thread-local model override — mirrors the OpenRouter pattern so the
# user-selected Bedrock model is used by all agents in the same thread.
_tl = threading.local()


def set_thread_model(model: str | None) -> None:
    """Set the Bedrock model for the current thread (None = use env default)."""
    _tl.model = model


def get_thread_model() -> str | None:
    return getattr(_tl, "model", None)

# Models that require a cross-region inference profile prefix.
# Bare IDs like "anthropic.claude-sonnet-4-5-20250929-v1:0" must become
# "us.anthropic.claude-sonnet-4-5-20250929-v1:0" (or eu./au.).
# Nova models use us./eu./ap. prefixes too.
_NEEDS_PROFILE_PREFIX = {
    # Claude 4.x
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-sonnet-4-6-20251120-v1:0",
    "anthropic.claude-opus-4-6-20251120-v1:0",
    # Claude 3.5/3.7
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    # Amazon Nova
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    # Meta Llama
    "meta.llama3-3-70b-instruct-v1:0",
}


def _resolve_model_id(model: str, region: str) -> str:
    """
    Resolve the correct cross-region inference profile ID for the given region.

    Strips any existing geographic prefix (us./eu./ap./au./global.) and
    re-applies the correct one for the configured region.  This means the
    frontend can send any prefixed ID and the backend will always use the
    right prefix for the account's region.
    """
    # Strip any existing geographic prefix so we work with the bare model ID
    bare = model
    for pfx in ("us.", "eu.", "ap.", "au.", "global."):
        if model.startswith(pfx):
            bare = model[len(pfx):]
            break

    if bare not in _NEEDS_PROFILE_PREFIX:
        # Model doesn't need a profile prefix (e.g. on-demand models)
        return bare

    if region.startswith("eu-"):
        prefix = "eu"
    elif region.startswith(("ap-southeast-2", "ap-southeast-4")):
        prefix = "au"
    elif region.startswith("ap-"):
        prefix = "ap"
    else:
        prefix = "us"
    return f"{prefix}.{bare}"


def _to_bedrock_tools(tools: list[RegisteredTool]) -> list[dict]:
    """Convert RegisteredTool list to Bedrock Converse toolSpec format."""
    specs = []
    for t in tools:
        schema = t.params_model.model_json_schema() if t.params_model else {"type": "object", "properties": {}}
        # Bedrock requires additionalProperties to be absent or False
        schema.pop("additionalProperties", None)
        specs.append({
            "toolSpec": {
                "name": t.name,
                "description": t.description or t.name,
                "inputSchema": {"json": schema},
            }
        })
    return specs


def _to_bedrock_messages(messages: list[Message]) -> list[dict]:
    """Convert internal Message list to Bedrock Converse messages format."""
    out: list[dict] = []
    for msg in messages:
        if msg.role == "user":
            out.append({"role": "user", "content": [{"text": msg.content or ""}]})

        elif msg.role == "assistant":
            content: list[dict] = []
            if msg.content:
                content.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append({
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    })
            out.append({"role": "assistant", "content": content})

        elif msg.role == "tool" and msg.tool_result:
            tr = msg.tool_result
            # Tool results must be wrapped in a user message in Bedrock
            try:
                result_content = json.loads(tr.content)
                if isinstance(result_content, dict) and "error" in result_content:
                    block = {"toolResult": {
                        "toolUseId": tr.tool_call_id,
                        "content": [{"text": result_content["error"]}],
                        "status": "error",
                    }}
                else:
                    block = {"toolResult": {
                        "toolUseId": tr.tool_call_id,
                        "content": [{"text": tr.content}],
                        "status": "success",
                    }}
            except (json.JSONDecodeError, TypeError):
                block = {"toolResult": {
                    "toolUseId": tr.tool_call_id,
                    "content": [{"text": tr.content}],
                    "status": "success" if not tr.is_error else "error",
                }}
            out.append({"role": "user", "content": [block]})

    # Bedrock requires alternating user/assistant roles — merge consecutive same-role messages
    merged: list[dict] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)
    return merged


class BedrockClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self.max_tokens = settings.bedrock_max_tokens
        region = settings.aws_region

        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=region,
            profile_name=settings.aws_profile or None,
        )
        self._client = session.client(
            "bedrock-runtime",
            config=BotoConfig(
                read_timeout=300,
                connect_timeout=10,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        # Priority: constructor arg > thread-local > env/config default
        raw_model = model or get_thread_model() or settings.bedrock_model
        self.model = _resolve_model_id(raw_model, region)

    # ── Legacy single-shot ────────────────────────────────────────────

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)
        user_text = (
            req.user_prompt
            + "\n\nDATASET_PREVIEW_PAYLOAD (JSON):\n"
            + payload_json
        )
        resp = self._client.converse(
            modelId=self.model,
            system=[{"text": req.system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        text = resp["output"]["message"]["content"][0].get("text", "")
        return LLMResponse(provider="bedrock", model=self.model, text=text)

    # ── Agentic chat ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[Message],
        tools: list[RegisteredTool] | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        bedrock_messages = _to_bedrock_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {"tools": _to_bedrock_tools(tools)}

        resp = self._client.converse(**kwargs)

        output_msg = resp["output"]["message"]
        stop_reason_raw = resp.get("stopReason", "end_turn")

        text_parts = []
        tool_calls: list[ToolCall] = []

        for block in output_msg.get("content", []):
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu["toolUseId"],
                    name=tu["name"],
                    arguments=tu["input"] if isinstance(tu["input"], dict) else json.loads(tu["input"]),
                ))

        stop_map = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
            "stop_sequence": "end_turn",
        }
        stop_reason = stop_map.get(stop_reason_raw, "end_turn")

        return ChatResponse(
            provider="bedrock",
            model=self.model,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )
