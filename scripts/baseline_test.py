import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_env
from app.models.registry import MODEL_REGISTRY
from app.providers.router_client import send_request

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Extract the name and age from this text: 'John is 34 years old.' Return as JSON with keys name and age.",
    "Summarize in one sentence: The stock market fell 2% today after inflation data came in higher than "
    "expected, prompting investors to reassess rate-cut expectations.",
    "Classify the sentiment of this review as positive, negative, or neutral: 'The food was okay, nothing special.'",
    "List three prime numbers between 10 and 30.",
    "Reformat this date to YYYY-MM-DD: March 5th, 2026.",
    "Compare the pros and cons of remote work vs office work in two sentences.",
    "Write a two-sentence story about a robot learning to paint.",
    "What does 'ubiquitous' mean? Answer in one sentence.",
    "A bus holds 40 people and 137 people need seats. How many buses are needed?",
]


def estimate_cost() -> float:
    avg_input_tokens = 40
    avg_output_tokens = 80
    total = 0.0
    for cfg in MODEL_REGISTRY.values():
        per_prompt = (
            avg_input_tokens / 1_000_000 * cfg.cost_per_1m_input
            + avg_output_tokens / 1_000_000 * cfg.cost_per_1m_output
        )
        total += per_prompt * len(PROMPTS)
    return total


def main() -> None:
    load_env()
    estimated = estimate_cost()
    print(f"Estimated cost for {len(PROMPTS)} prompts x {len(MODEL_REGISTRY)} models: ${estimated:.4f}")
    if "--yes" in sys.argv:
        confirmed = True
    else:
        confirmed = input("Proceed with real API calls? [y/N] ").strip().lower() == "y"
    if not confirmed:
        print("Aborted.")
        return

    log_path = Path(__file__).resolve().parent.parent / "data" / "logs" / "requests.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    actual_total = 0.0
    with log_path.open("a") as log_file:
        for model_key, cfg in MODEL_REGISTRY.items():
            for prompt in PROMPTS:
                resp = send_request(prompt, cfg)
                actual_total += resp.cost_usd
                status = "ERROR" if resp.error else "ok"
                print(f"[{status}] {model_key:20s} cost=${resp.cost_usd:.6f} latency={resp.latency_ms:.0f}ms")
                log_file.write(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "model_id": resp.model_id,
                            "provider": resp.provider,
                            "prompt": prompt,
                            "output": resp.text,
                            "input_tokens": resp.input_tokens,
                            "output_tokens": resp.output_tokens,
                            "cost_usd": resp.cost_usd,
                            "latency_ms": resp.latency_ms,
                            "error": resp.error,
                        }
                    )
                    + "\n"
                )

    print(f"\nDone. Actual total cost: ${actual_total:.4f}. Log: {log_path}")


if __name__ == "__main__":
    main()
