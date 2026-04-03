from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _parse_keys(raw: str | None) -> list[str]:
    """Split a comma-separated key string into a clean list."""
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
    openrouter_max_tokens: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))

    # AWS Bedrock
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_profile: str | None = os.getenv("AWS_PROFILE")
    bedrock_model: str = os.getenv("AWS_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    bedrock_max_tokens: int = int(os.getenv("AWS_BEDROCK_MAX_TOKENS", "8192"))

    # Literature / RAG
    ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")
    s2_api_key: str | None = os.getenv("S2_API_KEY")

    @property
    def gemini_api_keys(self) -> list[str]:
        """All Gemini API keys (supports comma-separated list in GEMINI_API_KEY)."""
        return _parse_keys(self.gemini_api_key)

    # Agent settings
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "25"))
    max_tool_result_chars: int = int(os.getenv("MAX_TOOL_RESULT_CHARS", "4000"))

settings = Settings()
