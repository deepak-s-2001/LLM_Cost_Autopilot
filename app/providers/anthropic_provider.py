from __future__ import annotations

import anthropic

from app.models.registry import ModelConfig
from app.providers.base import DEFAULT_MAX_TOKENS, RawResult


class AnthropicAdapter:
    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _call(
        self, prompt: str, model_config: ModelConfig, system: str | None = None, max_tokens: int | None = None
    ) -> RawResult:
        return self._create(prompt, model_config, system, max_tokens)

    def call_with_web_search(
        self, prompt: str, model_config: ModelConfig, system: str | None = None, max_tokens: int | None = None
    ) -> RawResult:
        # Server-side tool: Claude runs the search and returns results as content blocks in the same response, no client-side loop needed.
        return self._create(
            prompt, model_config, system, max_tokens,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
        )

    def _create(
        self,
        prompt: str,
        model_config: ModelConfig,
        system: str | None,
        max_tokens: int | None,
        tools: list[dict] | None = None,
    ) -> RawResult:
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return RawResult(text=text, input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
