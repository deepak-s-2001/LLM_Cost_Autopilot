from app.models.registry import MODEL_REGISTRY


def test_registry_has_five_models():
    assert len(MODEL_REGISTRY) == 5


def test_registry_entries_are_consistent():
    for key, cfg in MODEL_REGISTRY.items():
        assert cfg.model_id == key
        assert cfg.provider in {"openai", "anthropic", "ollama"}
        assert cfg.quality_tier in {"high", "medium", "low"}
        assert cfg.cost_per_1m_input >= 0
        assert cfg.cost_per_1m_output >= 0
