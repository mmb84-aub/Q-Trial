from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from qtrial_backend.core.types import LLMRequest, LLMResponse, ChatResponse, Message

if TYPE_CHECKING:
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
        raise NotImplementedError
