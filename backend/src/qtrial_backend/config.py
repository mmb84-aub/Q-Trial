from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


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

    # NCBI E-utilities (optional, increases PubMed rate limit)
    ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")

    # Semantic Scholar (optional, increases rate limit from ~100/5min to higher tiers)
    s2_api_key: str | None = os.getenv("S2_API_KEY")

    # Agent settings
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "25"))
    max_tool_result_chars: int = int(os.getenv("MAX_TOOL_RESULT_CHARS", "4000"))


settings = Settings()
