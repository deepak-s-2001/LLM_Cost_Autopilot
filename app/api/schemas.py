from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str = Field(max_length=50_000)


class CompletionRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)
    use_case: str | None = None
    max_tokens: int | None = Field(default=None, gt=0, le=8192)


class RoutingMetadata(BaseModel):
    tier: int
    model_id: str
    provider: str
    reason: str
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    escalated: bool = False
    tier_probabilities: dict[str, float] = {}
    features: dict = {}


class CompletionResponse(BaseModel):
    id: str
    content: str
    routing: RoutingMetadata


class VerificationStatus(BaseModel):
    status: str  # "pending" | "passed" | "escalated"
    quality_score: float | None = None
    escalated_content: str | None = None
    cost_delta_usd: float | None = None
    quality_gap: float | None = None
