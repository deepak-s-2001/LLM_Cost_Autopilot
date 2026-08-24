import json

from app.models.response import Response
from app.verification.quality_checks import classification_check, extraction_check
from app.verification.verifier import process_verification_job


def _fake_response(text: str, cost_usd: float = 0.001) -> Response:
    return Response(
        text=text, input_tokens=10, output_tokens=10, latency_ms=100.0,
        cost_usd=cost_usd, model_id="claude-sonnet-5", provider="anthropic",
    )


# ---------- quality_checks ----------

def test_extraction_check_passes_with_all_keys_present():
    assert extraction_check('{"name": "John", "age": 34}', ["name", "age"]) is True


def test_extraction_check_fails_with_missing_key():
    assert extraction_check('{"name": "John"}', ["name", "age"]) is False


def test_extraction_check_falls_back_to_substring_match_on_invalid_json():
    assert extraction_check("The name is John and age is 34", ["name", "age"]) is True
    assert extraction_check("The name is John", ["name", "age"]) is False


def test_classification_check_matches_case_and_whitespace_insensitively():
    assert classification_check(" Positive.\n", "positive") is True
    assert classification_check("negative", "positive") is False


def test_classification_check_matches_terse_label_inside_verbose_reference():
    # Regression: exact string equality previously scored this as a mismatch even though both answers agree on the label.
    reference = "**Category: Billing**\n\nReasoning: this ticket concerns a payment discrepancy."
    assert classification_check("billing", reference) is True
    assert classification_check("technical", reference) is False


# ---------- process_verification_job ----------

def _base_request(**overrides):
    request = {
        "id": "req-1",
        "prompt_full": "Extract the name from this text.",
        "response_text": '{"name": "John"}',
        "required_keys_json": json.dumps(["name"]),
        "use_case": "extraction",
        "routed_model_id": "llama3.1:8b",
        "cost_usd": 0.0,
    }
    request.update(overrides)
    return request


def test_extraction_pass_does_not_call_any_provider(mocker):
    mocker.patch("app.verification.verifier.repository.get_request", return_value=_base_request())
    complete_mock = mocker.patch("app.verification.verifier.repository.complete_verification_job")
    escalate_mock = mocker.patch("app.verification.verifier.escalate")
    log_training_mock = mocker.patch("app.verification.verifier.repository.log_training_example")
    send_request_mock = mocker.patch("app.verification.verifier.send_request")

    process_verification_job({"id": "job-1", "request_id": "req-1"})

    send_request_mock.assert_not_called()
    escalate_mock.assert_not_called()
    log_training_mock.assert_not_called()
    _, kwargs = complete_mock.call_args
    assert kwargs["passed"] is True


def test_extraction_failure_triggers_escalation_and_training_example(mocker):
    failing_request = _base_request(response_text='{"other_field": "x"}')
    mocker.patch("app.verification.verifier.repository.get_request", return_value=failing_request)
    complete_mock = mocker.patch("app.verification.verifier.repository.complete_verification_job")
    escalate_mock = mocker.patch(
        "app.verification.verifier.escalate", return_value=("esc-1", _fake_response("escalated answer"))
    )
    log_training_mock = mocker.patch("app.verification.verifier.repository.log_training_example")
    mocker.patch("app.verification.verifier.tier_for_model_id", return_value=3)

    process_verification_job({"id": "job-2", "request_id": "req-1"})

    _, kwargs = complete_mock.call_args
    assert kwargs["passed"] is False
    escalate_mock.assert_called_once()
    log_training_mock.assert_called_once()
    _, train_kwargs = log_training_mock.call_args
    assert train_kwargs["tier_label"] == 3
    assert train_kwargs["source"] == "routing_failure"


def test_general_use_case_uses_llm_judge(mocker):
    request = _base_request(use_case="summarization", response_text="A short summary.")
    mocker.patch("app.verification.verifier.repository.get_request", return_value=request)
    complete_mock = mocker.patch("app.verification.verifier.repository.complete_verification_job")
    mocker.patch("app.verification.verifier.repository.update_request_verification")
    mocker.patch("app.verification.verifier.escalate")
    mocker.patch(
        "app.verification.llm_judge.send_request",
        return_value=_fake_response('{"score": 4.5, "rationale": "solid summary"}'),
    )

    process_verification_job({"id": "job-3", "request_id": "req-1"})

    _, kwargs = complete_mock.call_args
    assert kwargs["passed"] is True
    assert kwargs["quality_score"] == 4.5


def test_ambiguous_score_triggers_real_output_comparison(mocker):
    # A score inside the ambiguous band should trigger a real comparison against the escalation model's own answer, not be trusted alone.
    request = _base_request(use_case="summarization", response_text="A so-so summary.")
    mocker.patch("app.verification.verifier.repository.get_request", return_value=request)
    complete_mock = mocker.patch("app.verification.verifier.repository.complete_verification_job")
    mocker.patch("app.verification.verifier.repository.update_request_verification")
    escalate_mock = mocker.patch("app.verification.verifier.escalate", return_value=("esc-1", _fake_response("x")))
    mocker.patch("app.verification.verifier.tier_for_model_id", return_value=3)
    mocker.patch("app.verification.verifier.repository.log_training_example")

    # First call (absolute judge) lands in the ambiguous band; second call (comparison) fails.
    mocker.patch(
        "app.verification.llm_judge.send_request",
        side_effect=[
            _fake_response('{"score": 3.5, "rationale": "borderline"}'),
        ],
    )
    reference_response = _fake_response("the real reference answer")
    mocker.patch("app.verification.verifier.send_request", return_value=reference_response)
    mocker.patch(
        "app.verification.verifier.compare_with_reference",
        return_value=mocker.Mock(passed=False, score=2.0, rationale="misses key point"),
    )

    process_verification_job({"id": "job-4", "request_id": "req-1"})

    _, kwargs = complete_mock.call_args
    assert kwargs["passed"] is False
    assert kwargs["quality_score"] == 2.0
    assert "[comparison]" in kwargs["divergence_notes"]

    # The comparison's reference answer must be reused for escalation, not regenerated with a second paid call.
    escalate_mock.assert_called_once()
    _, escalate_kwargs = escalate_mock.call_args
    assert escalate_kwargs["precomputed_response"] is reference_response
