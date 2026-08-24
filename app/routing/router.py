from app.classifier.features import needs_current_knowledge
from app.classifier.predict import classify_prompt, classify_prompt_detailed
from app.models.registry import MODEL_REGISTRY, ModelConfig
from app.models.response import Response
from app.providers.router_client import send_request, send_request_with_web_search
from app.routing.config import get_routing_config


def resolve_tier_to_model(tier: int) -> tuple[ModelConfig, ModelConfig | None]:
    config = get_routing_config()
    tier_cfg = config["tiers"][tier]
    model = MODEL_REGISTRY[tier_cfg["model"]]
    fallback_key = tier_cfg.get("fallback")
    fallback = MODEL_REGISTRY[fallback_key] if fallback_key else None
    return model, fallback


def tier_for_model_id(model_id: str) -> int | None:
    config = get_routing_config()
    for tier, cfg in config["tiers"].items():
        if cfg["model"] == model_id:
            return int(tier)
    return None


def generate_with_routing(
    prompt: str,
    context: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    classify_text: str | None = None,
) -> tuple[int, float, ModelConfig, Response, bool, dict[str, float], dict, bool]:
    """Classifies and routes to the appropriate model, retrying with the tier's fallback if the primary call fails (e.g. Ollama GPU OOM) and bypassing tier routing entirely for time-sensitive prompts in favor of web-search-augmented generation."""
    text_to_check = classify_text or prompt
    tier, confidence, tier_probabilities, features = classify_prompt_detailed(text_to_check, context)

    if needs_current_knowledge(text_to_check):
        model = get_escalation_model()
        response = send_request_with_web_search(prompt, model, system=system, max_tokens=max_tokens)
        return tier, confidence, model, response, False, tier_probabilities, features, True

    model, fallback = resolve_tier_to_model(tier)
    response = send_request(prompt, model, system=system, max_tokens=max_tokens)
    if response.error and fallback:
        fallback_response = send_request(prompt, fallback, system=system, max_tokens=max_tokens)
        if not fallback_response.error:
            return tier, confidence, fallback, fallback_response, True, tier_probabilities, features, False
    return tier, confidence, model, response, False, tier_probabilities, features, False


def get_judge_model() -> ModelConfig:
    config = get_routing_config()
    return MODEL_REGISTRY[config["verification"]["judge_model"]]


def get_escalation_model() -> ModelConfig:
    config = get_routing_config()
    return MODEL_REGISTRY[config["verification"]["escalation_model"]]
