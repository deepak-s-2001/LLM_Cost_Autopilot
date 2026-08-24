from __future__ import annotations

import openai

from app.models.registry import ModelConfig
from app.providers.base import DEFAULT_MAX_TOKENS, RawResult


class OpenAIAdapter:
    def __init__(self) -> None:
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def _call(
        self, prompt: str, model_config: ModelConfig, system: str | None = None, max_tokens: int | None = None
    ) -> RawResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=model_config.model_id,
            max_completion_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            messages=messages,
        )
        return RawResult(
            text=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )
