import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_env
from app.logging import repository
from app.verification.verifier import process_verification_job

POLL_INTERVAL_SECONDS = 5
# Processes jobs concurrently since a sequential loop can't keep up with bursty traffic (a ~500-request load test left 373 jobs queued minutes later).
CONCURRENCY = 8


def _process(job: dict) -> None:
    print(f"Processing verification job {job['id']} for request {job['request_id']}")
    try:
        process_verification_job(job)
    except Exception as e:
        print(f"Verification job {job['id']} failed: {e}")


def main() -> None:
    load_env()
    print(f"Verifier worker started, polling every {POLL_INTERVAL_SECONDS}s, concurrency={CONCURRENCY}")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        while True:
            jobs = repository.get_pending_verification_jobs()
            if jobs:
                list(pool.map(_process, jobs))
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
