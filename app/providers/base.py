from __future__ import annotations

import time
from typing import Callable, NamedTuple, Protocol

from app.models.registry import ModelConfig
from app.models.response import Response

DEFAULT_MAX_TOKENS = 2048


class RawResult(NamedTuple):
    text: str
    input_tokens: int
    output_tokens: int


class ProviderAdapter(Protocol):
    def _call(
        self, prompt: str, model_config: ModelConfig, system: str | None, max_tokens: int | None
    ) -> RawResult: ...


CallFn = Callable[[str, ModelConfig, "str | None", "int | None"], RawResult]


def send(
    adapter: ProviderAdapter,
    prompt: str,
    model_config: ModelConfig,
    system: str | None = None,
    max_tokens: int | None = None,
    call_fn: CallFn | None = None,
) -> Response:
    # call_fn lets a caller swap in a different adapter capability (e.g. web search) while reusing this cost/latency/error wrapping.
    call = call_fn or adapter._call
    start = time.perf_counter()
    try:
        result = call(prompt, model_config, system, max_tokens)
    except Exception as e:
        return Response(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=0.0,
            model_id=model_config.model_id,
            provider=model_config.provider,
            error=str(e),
        )
    latency_ms = (time.perf_counter() - start) * 1000
    cost_usd = (
        result.input_tokens / 1_000_000 * model_config.cost_per_1m_input
        + result.output_tokens / 1_000_000 * model_config.cost_per_1m_output
    )
    return Response(
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_id=model_config.model_id,
        provider=model_config.provider,
    )
