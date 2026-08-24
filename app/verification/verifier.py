import json

from app.classifier.features import extract_features
from app.logging import repository
from app.models.registry import MODEL_REGISTRY
from app.models.response import Response
from app.providers.router_client import send_request
from app.routing.router import get_escalation_model, get_judge_model, tier_for_model_id
from app.verification.escalation import escalate
from app.verification.llm_judge import compare_with_reference, judge_response
from app.verification.quality_checks import classification_check, extraction_check

# Scores between CLEAR_FAIL and CLEAR_PASS get a real output-comparison check against the escalation model instead of trusting the absolute judge alone, without paying for that extra call on every request.
CLEAR_PASS = 4.0
CLEAR_FAIL = 3.0


def enqueue_verification(request_id: str) -> str:
    return repository.create_verification_job(request_id)


def process_verification_job(job: dict) -> None:
    request = repository.get_request(job["request_id"])
    judge_model = get_judge_model()
    precomputed_response: Response | None = None

    if request["use_case"] == "extraction":
        required_keys = json.loads(request["required_keys_json"] or "[]")
        passed = extraction_check(request["response_text"], required_keys)
        score = 5.0 if passed else 1.0
        notes = "" if passed else f"missing required keys: {required_keys}"
    elif request["use_case"] == "classification":
        reference = send_request(request["prompt_full"], judge_model)
        passed = classification_check(request["response_text"], reference.text)
        score = 5.0 if passed else 1.0
        notes = "" if passed else f"reference label: {reference.text.strip()[:100]}"
    else:
        result = judge_response(request["prompt_full"], request["response_text"], judge_model, threshold=CLEAR_PASS)
        if result.score >= CLEAR_PASS or result.score < CLEAR_FAIL:
            passed, score, notes = result.passed, result.score, result.rationale
        else:
            escalation_model = get_escalation_model()
            precomputed_response = send_request(request["prompt_full"], escalation_model)
            comparison = compare_with_reference(
                request["prompt_full"], request["response_text"], precomputed_response.text, judge_model
            )
            passed, score, notes = comparison.passed, comparison.score, f"[comparison] {comparison.rationale}"

    repository.complete_verification_job(
        job["id"], judge_model_id=judge_model.model_id, quality_score=score, passed=passed, divergence_notes=notes
    )
    repository.update_request_verification(request["id"], quality_score=score, escalated=not passed)

    if not passed:
        _handle_failure(request, job_id=job["id"], quality_score=score, precomputed_response=precomputed_response)


def _handle_failure(
    request: dict, *, job_id: str, quality_score: float, precomputed_response: Response | None = None
) -> None:
    original_model = MODEL_REGISTRY[request["routed_model_id"]]
    escalation_model = get_escalation_model()

    _escalation_id, _escalated_response = escalate(
        request_id=request["id"],
        verification_job_id=job_id,
        prompt=request["prompt_full"],
        original_model=original_model,
        escalated_model=escalation_model,
        original_cost_usd=request["cost_usd"],
        quality_gap=5.0 - quality_score,
        precomputed_response=precomputed_response,
    )

    true_tier = tier_for_model_id(escalation_model.model_id) or 3
    features = extract_features(request["prompt_full"])
    repository.log_training_example(
        prompt_text=request["prompt_full"],
        features_json=json.dumps(features),
        tier_label=true_tier,
        source="routing_failure",
    )
