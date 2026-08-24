from app.logging.db import get_connection


def _date_clauses(start: str | None, end: str | None) -> tuple[str, list]:
    clauses, params = [], []
    if start:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        clauses.append("timestamp <= ?")
        params.append(end)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def total_savings(start: str | None = None, end: str | None = None) -> dict:
    where, params = _date_clauses(start, end)
    conn = get_connection()
    routed_total, baseline_total, count = conn.execute(
        f"SELECT COALESCE(SUM(cost_usd),0), COALESCE(SUM(baseline_cost_usd),0), COUNT(*) FROM requests{where}",
        params,
    ).fetchone()
    # Escalation reruns cost money too, but that cost lives in escalations.escalated_cost_usd, not requests.cost_usd, so it must be added in here or true spend is understated.
    esc_where, esc_params = _date_clauses(start, end)
    escalation_total = conn.execute(
        f"SELECT COALESCE(SUM(escalated_cost_usd),0) FROM escalations{esc_where}", esc_params
    ).fetchone()[0]
    conn.close()
    actual_total = routed_total + escalation_total
    savings_usd = baseline_total - actual_total
    savings_pct = (savings_usd / baseline_total * 100) if baseline_total > 0 else 0.0
    return {
        "actual_total_usd": actual_total,
        "baseline_total_usd": baseline_total,
        "escalation_cost_usd": escalation_total,
        "savings_usd": savings_usd,
        "savings_pct": savings_pct,
        "request_count": count,
    }


def routing_distribution(start: str | None = None, end: str | None = None) -> dict[str, int]:
    where, params = _date_clauses(start, end)
    conn = get_connection()
    rows = conn.execute(
        f"SELECT routed_model_id, COUNT(*) FROM requests{where} GROUP BY routed_model_id", params
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def quality_score_distribution() -> list[float]:
    conn = get_connection()
    rows = conn.execute("SELECT quality_score FROM verification_jobs WHERE quality_score IS NOT NULL").fetchall()
    conn.close()
    return [row[0] for row in rows]


def escalation_rate_over_time() -> list[dict]:
    # Buckets by hour rather than day, since demo/portfolio traffic tends to arrive in short bursts that would otherwise collapse into one unreadable day-level point.
    conn = get_connection()
    bucket = "strftime('%Y-%m-%d %H:00', timestamp)"
    total_rows = conn.execute(f"SELECT {bucket} AS bucket, COUNT(*) FROM requests GROUP BY bucket").fetchall()
    esc_rows = conn.execute(f"SELECT {bucket} AS bucket, COUNT(*) FROM escalations GROUP BY bucket").fetchall()
    conn.close()
    totals = {row[0]: row[1] for row in total_rows}
    escalations = {row[0]: row[1] for row in esc_rows}
    return [
        {
            "date": bucket,
            "total_requests": totals.get(bucket, 0),
            "escalations": escalations.get(bucket, 0),
            "escalation_rate": (escalations.get(bucket, 0) / totals[bucket]) if totals.get(bucket) else 0.0,
        }
        for bucket in sorted(set(totals) | set(escalations))
    ]
