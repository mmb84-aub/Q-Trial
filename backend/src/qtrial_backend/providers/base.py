from __future__ import annotations

from abc import ABC, abstractmethod
from qtrial_backend.core.types import LLMRequest, LLMResponse


class LLMClient(ABC):
    @abstractmethod
    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError
