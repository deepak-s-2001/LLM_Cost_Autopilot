import pytest
from fastapi.testclient import TestClient

import app.routing.config as routing_config
from app.api.main import app
from app.models.response import Response

client = TestClient(app)


def _fake_response(text="4", cost_usd=0.001):
    return Response(
        text=text, input_tokens=10, output_tokens=5, latency_ms=100.0,
        cost_usd=cost_usd, model_id="llama3.1:8b", provider="ollama",
    )


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))


def test_get_completion_verification_pending_before_worker_runs(mocker):
    mocker.patch("app.routing.router.send_request", return_value=_fake_response())
    post_response = client.post("/v1/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    request_id = post_response.json()["id"]

    response = client.get(f"/v1/completions/{request_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_get_completion_verification_escalated_after_worker_runs(mocker):
    from app.logging import repository

    mocker.patch("app.routing.router.send_request", return_value=_fake_response())
    post_response = client.post("/v1/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    request_id = post_response.json()["id"]

    # Simulate what the worker does once it processes the verification job.
    repository.update_request_verification(request_id, quality_score=2.0, escalated=True)
    job_id = repository.create_verification_job(request_id)
    repository.log_escalation(
        request_id=request_id, verification_job_id=job_id,
        original_model_id="llama3.1:8b", escalated_model_id="claude-sonnet-5",
        original_cost_usd=0.0, escalated_cost_usd=0.01, quality_gap=3.0,
        escalated_response_text="a better, revised answer",
    )

    response = client.get(f"/v1/completions/{request_id}")
    body = response.json()
    assert body["status"] == "escalated"
    assert body["escalated_content"] == "a better, revised answer"
    assert body["cost_delta_usd"] == pytest.approx(0.01)


def test_create_completion_success(mocker):
    mocker.patch("app.routing.router.send_request", return_value=_fake_response())
    response = client.post("/v1/completions", json={"messages": [{"role": "user", "content": "What is 2+2?"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "4"
    assert body["routing"]["tier"] in (1, 2, 3)
    assert body["routing"]["escalated"] is False
    assert "id" in body


def test_create_completion_requires_non_system_message():
    response = client.post("/v1/completions", json={"messages": [{"role": "system", "content": "be helpful"}]})
    assert response.status_code == 400


def test_create_completion_propagates_model_error(mocker):
    mocker.patch(
        "app.routing.router.send_request",
        return_value=Response(
            text="", input_tokens=0, output_tokens=0, latency_ms=10.0, cost_usd=0.0,
            model_id="x", provider="y", error="boom",
        ),
    )
    response = client.post("/v1/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 502


def test_list_models_returns_registry():
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert {m["model_id"] for m in body} == {
        "llama3.1:8b", "claude-haiku-4-5", "gpt-5.4-mini", "claude-sonnet-5", "gpt-5.4",
    }


def test_get_stats_shape(mocker):
    mocker.patch("app.routing.router.send_request", return_value=_fake_response())
    client.post("/v1/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    response = client.get("/v1/stats")
    assert response.status_code == 200
    body = response.json()
    assert "savings" in body and "routing_distribution" in body and "escalation_rate_over_time" in body
    assert body["savings"]["request_count"] == 1


def test_update_routing_config_rejects_invalid_model(mocker):
    bad_config = {
        "tiers": {1: {"model": "not-a-real-model", "fallback": None}, 2: {"model": "gpt-5.4-mini"}, 3: {"model": "gpt-5.4"}},
        "verification": {"judge_model": "claude-sonnet-5", "escalation_model": "claude-sonnet-5"},
    }
    response = client.put("/v1/routing-config", json=bad_config)
    assert response.status_code == 400


def test_update_routing_config_accepts_valid_config(tmp_path, monkeypatch):
    monkeypatch.setattr(routing_config, "_CONFIG_PATH", tmp_path / "routing.yaml")
    try:
        new_config = {
            "tiers": {
                1: {"model": "llama3.1:8b", "fallback": "gpt-5.4-mini"},
                2: {"model": "gpt-5.4-mini", "fallback": "claude-sonnet-5"},
                3: {"model": "claude-sonnet-5", "fallback": "gpt-5.4"},
            },
            "verification": {"judge_model": "claude-sonnet-5", "escalation_model": "claude-sonnet-5"},
        }
        response = client.put("/v1/routing-config", json=new_config)
        assert response.status_code == 200
        assert (tmp_path / "routing.yaml").exists()
    finally:
        routing_config._config_cache = None
