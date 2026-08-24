import pytest

from app.costs import stats
from app.logging import repository
from app.models.response import Response


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))

    def _log(cost_usd, baseline_cost_usd, model_id="gpt-5.4-mini", provider="openai"):
        response = Response(
            text="ok", input_tokens=100, output_tokens=100, latency_ms=500.0,
            cost_usd=cost_usd, model_id=model_id, provider=provider,
        )
        return repository.log_request(
            prompt="test prompt", response=response, complexity_tier=1,
            classifier_confidence=0.9, use_case=None, baseline_cost_usd=baseline_cost_usd,
        )

    request_ids = [
        _log(0.01, 0.10, model_id="gpt-5.4-mini"),
        _log(0.02, 0.10, model_id="gpt-5.4-mini"),
        _log(0.05, 0.10, model_id="claude-sonnet-5", provider="anthropic"),
    ]
    return request_ids


def test_total_savings_computes_exact_totals(seeded_db):
    result = stats.total_savings()
    assert result["actual_total_usd"] == pytest.approx(0.08)
    assert result["baseline_total_usd"] == pytest.approx(0.30)
    assert result["savings_usd"] == pytest.approx(0.22)
    assert result["savings_pct"] == pytest.approx(0.22 / 0.30 * 100)
    assert result["request_count"] == 3


def test_total_savings_on_empty_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "empty.db"))
    result = stats.total_savings()
    assert result["actual_total_usd"] == 0
    assert result["savings_pct"] == 0.0
    assert result["request_count"] == 0


def test_total_savings_includes_escalation_rerun_cost(seeded_db):
    # Regression: a savings figure that only sums requests.cost_usd silently omits escalation reruns, which are billed separately.
    request_id = seeded_db[0]
    job_id = repository.create_verification_job(request_id)
    repository.complete_verification_job(
        job_id, judge_model_id="claude-sonnet-5", quality_score=2.0, passed=False, divergence_notes=""
    )
    repository.log_escalation(
        request_id=request_id, verification_job_id=job_id,
        original_model_id="gpt-5.4-mini", escalated_model_id="claude-sonnet-5",
        original_cost_usd=0.01, escalated_cost_usd=0.09, quality_gap=3.0,
    )

    result = stats.total_savings()
    assert result["escalation_cost_usd"] == pytest.approx(0.09)
    assert result["actual_total_usd"] == pytest.approx(0.08 + 0.09)
    assert result["savings_usd"] == pytest.approx(0.30 - (0.08 + 0.09))


def test_routing_distribution_counts_per_model(seeded_db):
    dist = stats.routing_distribution()
    assert dist == {"gpt-5.4-mini": 2, "claude-sonnet-5": 1}


def test_quality_score_distribution_ignores_null_scores(seeded_db):
    request_id = seeded_db[0]
    job_id = repository.create_verification_job(request_id)
    repository.complete_verification_job(
        job_id, judge_model_id="claude-sonnet-5", quality_score=4.5, passed=True, divergence_notes=""
    )
    scores = stats.quality_score_distribution()
    assert scores == [4.5]
