from pathlib import Path

import yaml

from app.models.registry import MODEL_REGISTRY

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "routing.yaml"

_config_cache: dict | None = None


def load_routing_config() -> dict:
    global _config_cache
    with open(_CONFIG_PATH) as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache


def get_routing_config() -> dict:
    if _config_cache is None:
        return load_routing_config()
    return _config_cache


def normalize_tier_keys(config: dict) -> dict:
    # Normalizes tier keys to int, since JSON has no integer object keys but routing.yaml and the rest of the app do.
    if "tiers" in config:
        config["tiers"] = {int(k): v for k, v in config["tiers"].items()}
    return config


def validate_routing_config(config: dict) -> None:
    if "tiers" not in config or "verification" not in config:
        raise ValueError("routing config must have 'tiers' and 'verification' keys")
    for tier in (1, 2, 3):
        if tier not in config["tiers"]:
            raise ValueError(f"routing config missing tier {tier}")
        model_key = config["tiers"][tier].get("model")
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"tier {tier} references unknown model '{model_key}'")
        fallback_key = config["tiers"][tier].get("fallback")
        if fallback_key and fallback_key not in MODEL_REGISTRY:
            raise ValueError(f"tier {tier} fallback references unknown model '{fallback_key}'")
    for key in ("judge_model", "escalation_model"):
        model_key = config["verification"].get(key)
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"verification.{key} references unknown model '{model_key}'")


def save_routing_config(config: dict) -> None:
    config = normalize_tier_keys(config)
    validate_routing_config(config)
    with open(_CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    load_routing_config()
