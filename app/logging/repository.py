import hashlib
import json
import time
import uuid

from app.logging.db import get_connection
from app.models.response import Response


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_request(
    *,
    prompt: str,
    response: Response,
    complexity_tier: int,
    classifier_confidence: float | None,
    use_case: str | None,
    baseline_cost_usd: float,
    required_keys: list[str] | None = None,
    status: str = "completed",
) -> str:
    request_id = _new_id()
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO requests (
                id, timestamp, prompt_hash, prompt_preview, prompt_full, response_text,
                required_keys_json, use_case, complexity_tier, classifier_confidence,
                routed_provider, routed_model_id, input_tokens, output_tokens, cost_usd,
                latency_ms, baseline_cost_usd, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                request_id,
                _now(),
                hashlib.sha256(prompt.encode()).hexdigest(),
                prompt[:200],
                prompt,
                response.text,
                json.dumps(required_keys) if required_keys is not None else None,
                use_case,
                complexity_tier,
                classifier_confidence,
                response.provider,
                response.model_id,
                response.input_tokens,
                response.output_tokens,
                response.cost_usd,
                response.latency_ms,
                baseline_cost_usd,
                status,
            ),
        )
    conn.close()
    return request_id


def get_request(request_id: str) -> dict:
    conn = get_connection()
    row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    conn.close()
    return dict(row)


def create_verification_job(request_id: str) -> str:
    job_id = _new_id()
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO verification_jobs (id, request_id, status, created_at) VALUES (?,?,?,?)",
            (job_id, request_id, "pending", _now()),
        )
    conn.close()
    return job_id


def get_pending_verification_jobs() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM verification_jobs WHERE status = 'pending'").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def complete_verification_job(
    job_id: str, *, judge_model_id: str, quality_score: float, passed: bool, divergence_notes: str
) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE verification_jobs
            SET status = 'completed', completed_at = ?, judge_model_id = ?, quality_score = ?,
                passed = ?, divergence_notes = ?
            WHERE id = ?
            """,
            (_now(), judge_model_id, quality_score, int(passed), divergence_notes, job_id),
        )
    conn.close()


def update_request_verification(request_id: str, *, quality_score: float, escalated: bool) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE requests SET quality_score = ?, escalated = ? WHERE id = ?",
            (quality_score, int(escalated), request_id),
        )
    conn.close()


def log_escalation(
    *,
    request_id: str,
    verification_job_id: str,
    original_model_id: str,
    escalated_model_id: str,
    original_cost_usd: float,
    escalated_cost_usd: float,
    quality_gap: float | None,
    escalated_response_text: str | None = None,
) -> str:
    escalation_id = _new_id()
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO escalations (
                id, request_id, verification_job_id, original_model_id, escalated_model_id,
                original_cost_usd, escalated_cost_usd, cost_delta_usd, quality_gap, timestamp,
                escalated_response_text
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                escalation_id,
                request_id,
                verification_job_id,
                original_model_id,
                escalated_model_id,
                original_cost_usd,
                escalated_cost_usd,
                escalated_cost_usd - original_cost_usd,
                quality_gap,
                _now(),
                escalated_response_text,
            ),
        )
    conn.close()
    return escalation_id


def get_escalation_for_request(request_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM escalations WHERE request_id = ? ORDER BY timestamp DESC LIMIT 1", (request_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_verification_status(request_id: str) -> dict:
    conn = get_connection()
    row = conn.execute("SELECT quality_score, escalated FROM requests WHERE id = ?", (request_id,)).fetchone()
    conn.close()

    if row is None or row["quality_score"] is None:
        return {"status": "pending", "quality_score": None, "escalated_content": None, "cost_delta_usd": None, "quality_gap": None}

    if not row["escalated"]:
        return {"status": "passed", "quality_score": row["quality_score"], "escalated_content": None, "cost_delta_usd": None, "quality_gap": None}

    escalation = get_escalation_for_request(request_id)
    return {
        "status": "escalated",
        "quality_score": row["quality_score"],
        "escalated_content": escalation["escalated_response_text"] if escalation else None,
        "cost_delta_usd": escalation["cost_delta_usd"] if escalation else None,
        "quality_gap": escalation["quality_gap"] if escalation else None,
    }


def log_training_example(*, prompt_text: str, features_json: str, tier_label: int, source: str) -> str:
    example_id = _new_id()
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO training_examples (id, prompt_text, features_json, tier_label, source, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (example_id, prompt_text, features_json, tier_label, source, _now()),
        )
    conn.close()
    return example_id
