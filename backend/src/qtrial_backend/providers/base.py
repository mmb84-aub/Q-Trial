from __future__ import annotations

from abc import ABC, abstractmethod

from qtrial_backend.core.types import LLMRequest, LLMResponse, Message, ChatResponse
from qtrial_backend.tools.registry import RegisteredTool


class LLMClient(ABC):
    @abstractmethod
    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[RegisteredTool] | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        """Single-turn chat with optional tool definitions.

        The agent loop calls this repeatedly, managing conversation state
        externally.
        """
        raise NotImplementedError
