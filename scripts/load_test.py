import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.classifier.predict import classify_prompt
from app.config import load_env
from app.routing.router import resolve_tier_to_model

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "prompts" / "load_test_prompts.json"
REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"

# Rough output-length assumptions per tier, used only for the pre-flight estimate.
ESTIMATED_OUTPUT_TOKENS = {1: 50, 2: 200, 3: 600}


def load_prompts(limit: int | None) -> list[dict]:
    prompts = json.loads(PROMPTS_PATH.read_text())
    return prompts[:limit] if limit else prompts


def estimate_cost(prompts: list[dict]) -> tuple[float, dict]:
    total = 0.0
    tier_counts = {1: 0, 2: 0, 3: 0}
    for item in prompts:
        tier, _confidence = classify_prompt(item["prompt"])
        tier_counts[tier] += 1
        model, _fallback = resolve_tier_to_model(tier)
        input_tokens = max(len(item["prompt"].split()), 1)
        output_tokens = ESTIMATED_OUTPUT_TOKENS[tier]
        total += (
            input_tokens / 1_000_000 * model.cost_per_1m_input
            + output_tokens / 1_000_000 * model.cost_per_1m_output
        )
    return total, tier_counts


def send_one(base_url: str, item: dict) -> dict:
    payload = {"messages": [{"role": "user", "content": item["prompt"]}]}
    if item.get("use_case"):
        payload["use_case"] = item["use_case"]
    start = time.perf_counter()
    try:
        resp = httpx.post(f"{base_url}/v1/completions", json=payload, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        return {
            "prompt": item["prompt"][:100],
            "tier": body["routing"]["tier"],
            "model_id": body["routing"]["model_id"],
            "cost_usd": body["routing"]["cost_usd"],
            "latency_ms": body["routing"]["latency_ms"],
            "wall_ms": (time.perf_counter() - start) * 1000,
            "error": None,
        }
    except Exception as e:
        return {
            "prompt": item["prompt"][:100], "tier": None, "model_id": None,
            "cost_usd": 0.0, "latency_ms": 0.0, "wall_ms": (time.perf_counter() - start) * 1000,
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    load_env()
    prompts = load_prompts(args.limit)
    estimated, tier_counts = estimate_cost(prompts)

    print(f"Load test: {len(prompts)} prompts against {args.base_url}")
    print(f"Predicted tier distribution: {tier_counts}")
    print(f"Estimated cost: ${estimated:.4f}")

    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return

    try:
        httpx.get(f"{args.base_url}/v1/models", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Cannot reach {args.base_url} — is the API running? ({e})")
        return

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send_one, args.base_url, item) for item in prompts]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 25 == 0 or i == len(prompts):
                print(f"  {i}/{len(prompts)} completed")
    elapsed = time.perf_counter() - start

    errors = [r for r in results if r["error"]]
    successes = [r for r in results if not r["error"]]
    actual_cost = sum(r["cost_usd"] for r in successes)
    tier_dist = {}
    for r in successes:
        tier_dist[r["tier"]] = tier_dist.get(r["tier"], 0) + 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"load_test_report_{int(time.time())}.json"
    report_path.write_text(json.dumps(results, indent=2))

    print(f"\nDone in {elapsed:.1f}s. {len(successes)} succeeded, {len(errors)} failed.")
    print(f"Actual cost: ${actual_cost:.4f} (estimated ${estimated:.4f})")
    print(f"Actual tier distribution: {tier_dist}")
    print(f"Full report: {report_path}")
    if errors:
        print(f"First error: {errors[0]['error']}")


if __name__ == "__main__":
    main()
