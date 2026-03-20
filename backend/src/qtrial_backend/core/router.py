from __future__ import annotations

from qtrial_backend.core.types import ProviderName
from qtrial_backend.providers.openai_client import OpenAIClient
from qtrial_backend.providers.gemini_client import GeminiClient
from qtrial_backend.providers.claude_client import ClaudeClient
from qtrial_backend.providers.openrouter_client import OpenRouterClient


def get_client(provider: ProviderName, model: str | None = None):
    if provider == "openai":
        return OpenAIClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "claude":
        return ClaudeClient()
    if provider == "openrouter":
        return OpenRouterClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")
