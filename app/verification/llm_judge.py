import json
import re
from dataclasses import dataclass

from app.models.registry import ModelConfig
from app.providers.router_client import send_request

JUDGE_SYSTEM_PROMPT = (
    'You are a strict quality judge. Given a user prompt and a candidate response, score how well '
    'the response answers the prompt from 1 (poor) to 5 (excellent). Reply with ONLY a JSON object: '
    '{"score": <number>, "rationale": "<one sentence>"}.'
)

COMPARISON_SYSTEM_PROMPT = (
    "You are a strict quality judge. You are given a user prompt, a candidate response, and a "
    "reference response from a stronger model. Score how well the candidate agrees with the "
    "reference in substance (not wording) from 1 (diverges badly, missing or wrong information) "
    "to 5 (fully agrees, no meaningful gap) — the candidate does not need to match the reference's "
    "style or length, only its correctness and completeness. Reply with ONLY a JSON object: "
    '{"score": <number>, "rationale": "<one sentence>"}.'
)


@dataclass
class JudgeResult:
    score: float
    rationale: str
    passed: bool


def judge_response(prompt: str, response_text: str, judge_model: ModelConfig, threshold: float = 4.0) -> JudgeResult:
    judge_prompt = f"User prompt:\n{prompt}\n\nCandidate response:\n{response_text}"
    result = send_request(judge_prompt, judge_model, system=JUDGE_SYSTEM_PROMPT)
    if result.error:
        return JudgeResult(score=0.0, rationale=f"judge call failed: {result.error}", passed=False)
    score, rationale = _parse_judge_output(result.text)
    return JudgeResult(score=score, rationale=rationale, passed=score >= threshold)


def compare_with_reference(
    prompt: str, candidate_text: str, reference_text: str, judge_model: ModelConfig, threshold: float = 4.0
) -> JudgeResult:
    judge_prompt = (
        f"User prompt:\n{prompt}\n\nCandidate response:\n{candidate_text}\n\n"
        f"Reference response:\n{reference_text}"
    )
    result = send_request(judge_prompt, judge_model, system=COMPARISON_SYSTEM_PROMPT)
    if result.error:
        return JudgeResult(score=0.0, rationale=f"judge call failed: {result.error}", passed=False)
    score, rationale = _parse_judge_output(result.text)
    return JudgeResult(score=score, rationale=rationale, passed=score >= threshold)


def _parse_judge_output(text: str) -> tuple[float, str]:
    try:
        data = json.loads(text)
        return float(data["score"]), str(data.get("rationale", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    match = re.search(r'"?score"?\s*[:=]\s*([\d.]+)', text)
    score = float(match.group(1)) if match else 0.0
    return score, text[:200]
