from dataclasses import dataclass
from typing import Literal

QualityTier = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    display_name: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    avg_latency_ms: float
    quality_tier: QualityTier
    context_window: int


# Pricing verified live on 2026-08-20; claude-sonnet-5 uses its standard $3/$15 rate (not the $2/$10 intro price) so savings figures don't shift mid-project, and avg_latency_ms is measured from scripts/baseline_test.py.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama3.1:8b": ModelConfig(
        provider="ollama",
        model_id="llama3.1:8b",
        display_name="Llama 3.1 8B (local)",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=3339.0,
        quality_tier="low",
        context_window=128_000,
    ),
    "claude-haiku-4-5": ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        cost_per_1m_input=1.00,
        cost_per_1m_output=5.00,
        avg_latency_ms=3218.0,
        quality_tier="low",
        context_window=200_000,
    ),
    "gpt-5.4-mini": ModelConfig(
        provider="openai",
        model_id="gpt-5.4-mini",
        display_name="GPT-5.4 Mini",
        cost_per_1m_input=0.75,
        cost_per_1m_output=4.50,
        avg_latency_ms=952.0,
        quality_tier="medium",
        context_window=1_050_000,
    ),
    "claude-sonnet-5": ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        avg_latency_ms=2416.0,
        quality_tier="high",
        context_window=1_000_000,
    ),
    "gpt-5.4": ModelConfig(
        provider="openai",
        model_id="gpt-5.4",
        display_name="GPT-5.4",
        cost_per_1m_input=2.50,
        cost_per_1m_output=15.00,
        avg_latency_ms=1024.0,
        quality_tier="high",
        context_window=1_050_000,
    ),
}
