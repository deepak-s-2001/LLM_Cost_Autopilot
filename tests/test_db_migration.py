import sqlite3

from app.logging.db import get_connection


def test_migration_backfills_columns_on_pre_existing_db(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    # Simulates a DB created before these columns existed, by creating the old-shape tables directly.
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE requests (
            id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, prompt_hash TEXT NOT NULL,
            prompt_preview TEXT NOT NULL, prompt_full TEXT NOT NULL, response_text TEXT NOT NULL,
            required_keys_json TEXT, use_case TEXT, complexity_tier INTEGER NOT NULL,
            classifier_confidence REAL, routed_provider TEXT NOT NULL, routed_model_id TEXT NOT NULL,
            input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL,
            latency_ms REAL NOT NULL, baseline_cost_usd REAL NOT NULL, status TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE escalations (
            id TEXT PRIMARY KEY, request_id TEXT NOT NULL, verification_job_id TEXT NOT NULL,
            original_model_id TEXT NOT NULL, escalated_model_id TEXT NOT NULL,
            original_cost_usd REAL NOT NULL, escalated_cost_usd REAL NOT NULL,
            cost_delta_usd REAL NOT NULL, quality_gap REAL, timestamp TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO requests VALUES ('r1','t','h','p','p','r',NULL,NULL,1,0.9,'ollama','llama3.1:8b',1,1,0.0,1.0,0.0,'completed')"
    )
    conn.commit()
    conn.close()

    conn = get_connection()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
    assert {"escalated", "quality_score"} <= columns
    esc_columns = {row[1] for row in conn.execute("PRAGMA table_info(escalations)")}
    assert "escalated_response_text" in esc_columns

    row = conn.execute("SELECT escalated, quality_score FROM requests WHERE id = 'r1'").fetchone()
    conn.close()
    assert row["escalated"] == 0
    assert row["quality_score"] is None
