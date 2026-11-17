"""Cost/token logging. Pure code: just records what llm_client reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402


def make_logger(conn=None):
    own_conn = conn is None
    conn = conn or get_conn()

    def log(stage, model, input_tokens, output_tokens, usd_cost):
        conn.execute(
            "INSERT INTO cost_log (stage, model, input_tokens, output_tokens, usd_cost) VALUES (?, ?, ?, ?, ?)",
            (stage, model, input_tokens, output_tokens, usd_cost),
        )
        conn.commit()

    return log


def summarize(conn=None):
    own_conn = conn is None
    conn = conn or get_conn()
    rows = conn.execute(
        "SELECT stage, COUNT(*) AS n, SUM(input_tokens) AS it, SUM(output_tokens) AS ot, SUM(usd_cost) AS cost "
        "FROM cost_log GROUP BY stage ORDER BY cost DESC"
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) AS n, SUM(input_tokens) AS it, SUM(output_tokens) AS ot, SUM(usd_cost) AS cost FROM cost_log"
    ).fetchone()
    return {"by_stage": [dict(r) for r in rows], "total": dict(total) if total else {}}


if __name__ == "__main__":
    import json
    print(json.dumps(summarize(), indent=2))
