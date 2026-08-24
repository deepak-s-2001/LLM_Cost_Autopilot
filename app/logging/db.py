import os
import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    prompt_full TEXT NOT NULL,
    response_text TEXT NOT NULL,
    required_keys_json TEXT,
    use_case TEXT,
    complexity_tier INTEGER NOT NULL,
    classifier_confidence REAL,
    routed_provider TEXT NOT NULL,
    routed_model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    baseline_cost_usd REAL NOT NULL,
    status TEXT NOT NULL,
    escalated INTEGER NOT NULL DEFAULT 0,
    quality_score REAL
);

CREATE TABLE IF NOT EXISTS verification_jobs (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES requests(id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    judge_model_id TEXT,
    quality_score REAL,
    passed INTEGER,
    divergence_notes TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES requests(id),
    verification_job_id TEXT NOT NULL REFERENCES verification_jobs(id),
    original_model_id TEXT NOT NULL,
    escalated_model_id TEXT NOT NULL,
    original_cost_usd REAL NOT NULL,
    escalated_cost_usd REAL NOT NULL,
    cost_delta_usd REAL NOT NULL,
    quality_gap REAL,
    timestamp TEXT NOT NULL,
    escalated_response_text TEXT
);

CREATE TABLE IF NOT EXISTS training_examples (
    id TEXT PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    features_json TEXT NOT NULL,
    tier_label INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


# CREATE TABLE IF NOT EXISTS only helps fresh files, so pre-existing DBs get missing columns backfilled here via a table_info check, safe to re-run every connection.
_COLUMN_MIGRATIONS = [
    ("requests", "escalated", "ALTER TABLE requests ADD COLUMN escalated INTEGER NOT NULL DEFAULT 0"),
    ("requests", "quality_score", "ALTER TABLE requests ADD COLUMN quality_score REAL"),
    ("escalations", "escalated_response_text", "ALTER TABLE escalations ADD COLUMN escalated_response_text TEXT"),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, column, ddl in _COLUMN_MIGRATIONS:
        existing_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing_columns:
            conn.execute(ddl)


def get_connection() -> sqlite3.Connection:
    db_path = Path(os.environ.get("DATABASE_PATH", "./data/autopilot.db"))
    if not db_path.is_absolute():
        db_path = _PROJECT_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _apply_migrations(conn)
    return conn
