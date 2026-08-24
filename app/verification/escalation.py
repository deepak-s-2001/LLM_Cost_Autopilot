from app.logging import repository
from app.models.registry import ModelConfig
from app.models.response import Response
from app.providers.router_client import send_request


def escalate(
    *,
    request_id: str,
    verification_job_id: str,
    prompt: str,
    original_model: ModelConfig,
    escalated_model: ModelConfig,
    original_cost_usd: float,
    quality_gap: float | None,
    system: str | None = None,
    precomputed_response: Response | None = None,
) -> tuple[str, Response]:
    # Reuses a precomputed response (e.g. from an ambiguous-band comparison) instead of paying for a second identical call.
    result = precomputed_response if precomputed_response is not None else send_request(prompt, escalated_model, system=system)
    escalation_id = repository.log_escalation(
        request_id=request_id,
        verification_job_id=verification_job_id,
        original_model_id=original_model.model_id,
        escalated_model_id=escalated_model.model_id,
        original_cost_usd=original_cost_usd,
        escalated_cost_usd=result.cost_usd,
        quality_gap=quality_gap,
        escalated_response_text=result.text,
    )
    return escalation_id, result
