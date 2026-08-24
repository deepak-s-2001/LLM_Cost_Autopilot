from app.models.registry import MODEL_REGISTRY

# The flagship model every request is compared against for the "you saved $X" metric, matching the spec's original "vs. GPT-4o" framing.
BASELINE_MODEL_KEY = "gpt-5.4"


def baseline_cost_for_request(input_tokens: int, output_tokens: int) -> float:
    model = MODEL_REGISTRY[BASELINE_MODEL_KEY]
    return (
        input_tokens / 1_000_000 * model.cost_per_1m_input
        + output_tokens / 1_000_000 * model.cost_per_1m_output
    )
