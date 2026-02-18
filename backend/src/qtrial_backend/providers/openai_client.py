from __future__ import annotations

import json
from openai import OpenAI

from qtrial_backend.config import settings
from qtrial_backend.core.types import LLMRequest, LLMResponse
from qtrial_backend.providers.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": req.system_prompt},
                {
                    "role": "user",
                    "content": (
                        req.user_prompt
                        + "\n\nDATASET_PREVIEW_PAYLOAD (JSON):\n"
                        + payload_json
                    ),
                },
            ],
        )
        text = getattr(resp, "output_text", None) or ""
        return LLMResponse(provider="openai", model=self.model, text=text)
