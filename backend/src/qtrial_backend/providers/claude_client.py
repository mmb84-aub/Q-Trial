from __future__ import annotations

import json
import anthropic

from qtrial_backend.config import settings
from qtrial_backend.core.types import LLMRequest, LLMResponse
from qtrial_backend.providers.base import LLMClient


class ClaudeClient(LLMClient):
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    def generate(self, req: LLMRequest) -> LLMResponse:
        payload_json = json.dumps(req.payload, indent=2, ensure_ascii=False)

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=req.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        req.user_prompt
                        + "\n\nDATASET_PREVIEW_PAYLOAD (JSON):\n"
                        + payload_json
                    ),
                }
            ],
        )

        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        return LLMResponse(provider="claude", model=self.model, text="\n".join(text_parts))

