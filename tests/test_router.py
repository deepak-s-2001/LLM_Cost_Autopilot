from app.models.registry import MODEL_REGISTRY
from app.models.response import Response
from app.routing.router import (
    generate_with_routing,
    get_escalation_model,
    get_judge_model,
    resolve_tier_to_model,
    tier_for_model_id,
)


def test_resolve_tier_to_model_returns_configured_models():
    model, fallback = resolve_tier_to_model(1)
    assert model.model_id == "llama3.1:8b"
    assert fallback.model_id == "gpt-5.4-mini"


def test_tier_for_model_id_reverse_lookup():
    assert tier_for_model_id("llama3.1:8b") == 1
    assert tier_for_model_id("gpt-5.4-mini") == 2
    assert tier_for_model_id("claude-sonnet-5") == 3
    assert tier_for_model_id("nonexistent-model") is None


def test_judge_and_escalation_models_are_valid_registry_entries():
    judge = get_judge_model()
    escalation = get_escalation_model()
    assert judge.model_id in MODEL_REGISTRY
    assert escalation.model_id in MODEL_REGISTRY


def _fake_response(error=None, text="ok"):
    return Response(
        text=text, input_tokens=1, output_tokens=1, latency_ms=1.0,
        cost_usd=0.0, model_id="x", provider="y", error=error,
    )


def test_generate_with_routing_uses_primary_model_on_success(mocker):
    mocker.patch("app.routing.router.send_request", return_value=_fake_response())
    tier, confidence, model, response, used_fallback, tier_probs, features, used_web_search = generate_with_routing(
        "What is the capital of France?"
    )
    assert used_fallback is False
    assert used_web_search is False
    assert response.error is None
    assert set(tier_probs.keys()) == {"1", "2", "3"}
    assert "token_count" in features


def test_generate_with_routing_falls_back_when_primary_fails(mocker):
    _primary, expected_fallback = resolve_tier_to_model(1)
    responses = iter([_fake_response(error="boom"), _fake_response(text="recovered")])
    mock_send = mocker.patch("app.routing.router.send_request", side_effect=lambda *a, **k: next(responses))
    mocker.patch("app.routing.router.classify_prompt_detailed", return_value=(1, 0.9, {"1": 0.9}, {}))

    result = generate_with_routing("anything")
    _tier, _confidence, model, response, used_fallback, _tier_probs, _features, used_web_search = result

    assert used_fallback is True
    assert used_web_search is False
    assert response.error is None
    assert response.text == "recovered"
    assert model.model_id == expected_fallback.model_id
    assert mock_send.call_count == 2


def test_generate_with_routing_returns_error_when_both_fail(mocker):
    mocker.patch("app.routing.router.classify_prompt_detailed", return_value=(1, 0.9, {"1": 0.9}, {}))
    mocker.patch("app.routing.router.send_request", return_value=_fake_response(error="boom"))

    result = generate_with_routing("anything")
    _tier, _confidence, _model, response, used_fallback, _tier_probs, _features, used_web_search = result

    assert used_fallback is False
    assert used_web_search is False
    assert response.error == "boom"


def test_generate_with_routing_uses_web_search_for_time_sensitive_prompts(mocker):
    mocker.patch("app.routing.router.classify_prompt_detailed", return_value=(1, 0.95, {"1": 0.95}, {}))
    search_mock = mocker.patch(
        "app.routing.router.send_request_with_web_search", return_value=_fake_response(text="current answer")
    )
    plain_send_mock = mocker.patch("app.routing.router.send_request")

    result = generate_with_routing("What is the current state of the government shutdown in 2026?")
    _tier, _confidence, model, response, used_fallback, _tier_probs, _features, used_web_search = result

    assert used_web_search is True
    assert used_fallback is False
    assert response.text == "current answer"
    assert model.model_id == get_escalation_model().model_id
    search_mock.assert_called_once()
    plain_send_mock.assert_not_called()
