from __future__ import annotations

from app.models.registry import ModelConfig
from app.models.response import Response
from app.providers import base
from app.providers.anthropic_provider import AnthropicAdapter
from app.providers.ollama_provider import OllamaAdapter
from app.providers.openai_provider import OpenAIAdapter

_ADAPTERS = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "ollama": OllamaAdapter(),
}


def send_request(
    prompt: str, model_config: ModelConfig, system: str | None = None, max_tokens: int | None = None
) -> Response:
    adapter = _ADAPTERS[model_config.provider]
    return base.send(adapter, prompt, model_config, system, max_tokens)


def send_request_with_web_search(
    prompt: str, model_config: ModelConfig, system: str | None = None, max_tokens: int | None = None
) -> Response:
    adapter = _ADAPTERS[model_config.provider]
    if not hasattr(adapter, "call_with_web_search"):
        raise ValueError(f"{model_config.provider} does not support web search")
    return base.send(adapter, prompt, model_config, system, max_tokens, call_fn=adapter.call_with_web_search)
