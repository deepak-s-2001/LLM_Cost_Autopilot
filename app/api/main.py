from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.api.schemas import CompletionRequest, CompletionResponse, Message, RoutingMetadata, VerificationStatus
from app.config import load_env
from app.costs import stats
from app.costs.baseline import baseline_cost_for_request
from app.logging import repository
from app.models.registry import MODEL_REGISTRY
from app.routing.config import get_routing_config, save_routing_config
from app.routing.router import generate_with_routing
from app.verification.verifier import enqueue_verification

load_env()

app = FastAPI(title="LLM Cost Autopilot")


def _split_messages(messages: list[Message]) -> tuple[list[Message], str | None]:
    system_parts = [m.content for m in messages if m.role == "system"]
    system = "\n".join(system_parts) if system_parts else None
    turns = [m for m in messages if m.role != "system"]
    return turns, system


def _flatten_transcript(turns: list[Message]) -> str:
    if len(turns) == 1:
        return turns[0].content
    return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in turns)


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks) -> CompletionResponse:
    turns, system = _split_messages(request.messages)
    if not turns:
        raise HTTPException(status_code=400, detail="messages must include at least one non-system message")

    last_content = turns[-1].content
    context = _flatten_transcript(turns[:-1]) if len(turns) > 1 else None
    full_prompt = _flatten_transcript(turns)

    tier, confidence, model, response, used_fallback, tier_probabilities, features, used_web_search = (
        generate_with_routing(
            full_prompt, context=context, system=system, max_tokens=request.max_tokens, classify_text=last_content
        )
    )

    if response.error:
        raise HTTPException(status_code=502, detail=f"model call failed: {response.error}")

    baseline_cost = baseline_cost_for_request(response.input_tokens, response.output_tokens)
    request_id = repository.log_request(
        prompt=full_prompt,
        response=response,
        complexity_tier=tier,
        classifier_confidence=confidence,
        use_case=request.use_case,
        baseline_cost_usd=baseline_cost,
    )
    # Verification runs after the response is sent, so "escalated" below is always False here (see app/verification/verifier.py).
    background_tasks.add_task(enqueue_verification, request_id)

    if used_web_search:
        reason = "detected as needing current information — routed directly to Claude Sonnet 5 with live web search, bypassing normal tier routing"
    else:
        reason = f"classified as tier {tier} (confidence {confidence:.0%})"
        if used_fallback:
            reason += " — primary model failed, used configured fallback"

    return CompletionResponse(
        id=request_id,
        content=response.text,
        routing=RoutingMetadata(
            tier=tier,
            model_id=model.model_id,
            provider=model.provider,
            reason=reason,
            cost_usd=response.cost_usd,
            baseline_cost_usd=baseline_cost,
            latency_ms=response.latency_ms,
            escalated=False,
            tier_probabilities=tier_probabilities,
            features=features,
        ),
    )


@app.get("/v1/completions/{request_id}", response_model=VerificationStatus)
def get_completion_verification(request_id: str) -> VerificationStatus:
    return VerificationStatus(**repository.get_verification_status(request_id))


@app.get("/v1/models")
def list_models() -> list[dict]:
    return [
        {
            "model_id": cfg.model_id,
            "provider": cfg.provider,
            "display_name": cfg.display_name,
            "cost_per_1m_input": cfg.cost_per_1m_input,
            "cost_per_1m_output": cfg.cost_per_1m_output,
            "quality_tier": cfg.quality_tier,
        }
        for cfg in MODEL_REGISTRY.values()
    ]


@app.get("/v1/stats")
def get_stats() -> dict:
    return {
        "savings": stats.total_savings(),
        "routing_distribution": stats.routing_distribution(),
        "escalation_rate_over_time": stats.escalation_rate_over_time(),
    }


@app.put("/v1/routing-config")
def update_routing_config(config: dict) -> dict:
    try:
        save_routing_config(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return get_routing_config()


# Mounted last at "/" so Starlette matches the explicit /v1/* routes first; html=True serves frontend/index.html for "/".
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
