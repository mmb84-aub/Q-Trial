from __future__ import annotations

from qtrial_backend.core.types import ProviderName
from qtrial_backend.providers.openai_client import OpenAIClient
from qtrial_backend.providers.gemini_client import GeminiClient
from qtrial_backend.providers.claude_client import ClaudeClient


def get_client(provider: ProviderName):
    if provider == "openai":
        return OpenAIClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "claude":
        return ClaudeClient()
    raise ValueError(f"Unknown provider: {provider}")
