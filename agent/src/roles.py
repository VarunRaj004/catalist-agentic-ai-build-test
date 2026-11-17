"""Speaker -> role mapping, one short LLM call per team.

Why this is an LLM job and not regex: self-introductions in the AMI corpus
are phrased in wildly different ways ("I'm the Marketing Expert", "I'm our
marketing expert", a third-person roll-call with no stated roles at all, or
-- for the two teams missing meeting 'a' -- no introduction in the available
meetings at all). A handful of transcripts we sampled didn't match ANY
reasonable keyword regex. This is exactly the kind of fuzzy natural-language
classification an LLM is good at and hand-rolled string matching is not; the
cost is trivial (35 short calls, once, cached in the `roles` table).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from llm_client import LLMClient, LLMError  # noqa: E402

SYSTEM = (
    "You are extracting the role of each speaker from the start of a business meeting transcript. "
    "The four roles are always: Project Manager, Industrial Designer, User Interface Designer, Marketing. "
    "Each of Speaker A, B, C, D has exactly one of these roles (one role per speaker, no repeats), unless "
    "the transcript genuinely gives no evidence for a speaker, in which case use \"unknown\". "
    "Reply with ONLY a JSON object: {\"A\": \"role or unknown\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}."
)


def build_roles(log_fn=None, conn=None):
    # Accept an existing connection so a caller that also needs it for cost-logging shares the
    # same one, instead of two sqlite3 connections to the same file fighting each other.
    own_conn = conn is None
    conn = conn or get_conn()
    client = LLMClient(log_fn=log_fn)
    teams = [r["team"] for r in conn.execute("SELECT DISTINCT team FROM meetings ORDER BY team").fetchall()]

    results = {}
    for team in teams:
        if conn.execute("SELECT 1 FROM roles WHERE team = ?", (team,)).fetchone():
            continue  # already processed -- supports resuming across days under a daily rate cap
        earliest = conn.execute(
            "SELECT meeting_id FROM meetings WHERE team = ? ORDER BY letter LIMIT 1", (team,)
        ).fetchone()
        meeting_id = earliest["meeting_id"]
        turns = conn.execute(
            "SELECT speaker, text FROM turns WHERE meeting_id = ? ORDER BY turn_idx LIMIT 80",
            (meeting_id,),
        ).fetchall()
        transcript_snippet = "\n".join(f"Speaker {t['speaker']}: {t['text']}" for t in turns)
        user = f"Transcript (start of meeting {meeting_id}):\n\n{transcript_snippet}"
        try:
            # retries=1: fail fast on rate-limit/quota backoff once the daily free-tier cap is hit
            # (truncation retries are unaffected -- they're on their own budget, see llm_client.py).
            roles = client.call_json(SYSTEM, user, max_tokens=1200, stage="roles", retries=1)
        except LLMError as e:
            # Do NOT write anything on failure. This team must stay unprocessed (no row at all) so
            # the "already processed" resume-check in the caller correctly retries it next time,
            # instead of permanently locking in a bogus all-"unknown" result.
            print(f"[roles] {team}: LLM error, leaving unprocessed for retry ({e})")
            continue
        for speaker in ["A", "B", "C", "D"]:
            role = roles.get(speaker, "unknown") if isinstance(roles, dict) else "unknown"
            conn.execute(
                "INSERT OR REPLACE INTO roles (team, speaker, role, confidence, source_meeting) "
                "VALUES (?, ?, ?, ?, ?)",
                (team, speaker, role, "llm", meeting_id),
            )
        results[team] = roles
        conn.commit()
    if own_conn:
        conn.close()
    return results


if __name__ == "__main__":
    print(build_roles())
