from __future__ import annotations

import json
from google import genai

from qtrial_backend.config import settings
from qtrial_backend.core.types import LLMRequest, LLMResponse
from qtrial_backend.providers.base import LLMClient


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        prompt = (
            f"{req.system_prompt}\n\n"
            f"{req.user_prompt}\n\n"
            f"DATASET_PREVIEW_PAYLOAD (JSON):\n{payload_json}"
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = getattr(resp, "text", "") or ""
        return LLMResponse(provider="gemini", model=self.model, text=text)
