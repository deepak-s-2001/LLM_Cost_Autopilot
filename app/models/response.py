from dataclasses import dataclass


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str
    provider: str
    error: str | None = None
