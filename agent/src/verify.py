"""Citation verification. Pure code -- this is the one place we never trust the model.

A citation is only valid if its quote is an exact (whitespace/case-insensitive)
substring of a single turn in the named meeting, and has at least six words,
matching the grading rule in questions.json.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


_TURN_CACHE = {}


def _turns_for_meeting(conn, meeting_id: str):
    if meeting_id not in _TURN_CACHE:
        rows = conn.execute(
            "SELECT turn_idx, speaker, text FROM turns WHERE meeting_id = ? ORDER BY turn_idx",
            (meeting_id,),
        ).fetchall()
        _TURN_CACHE[meeting_id] = [dict(r) for r in rows]
    return _TURN_CACHE[meeting_id]


def verify_citation(conn, meeting_id: str, quote: str):
    """Return (ok: bool, reason: str, matched_turn: dict|None)."""
    if not quote or len(quote.split()) < 6:
        return False, "quote has fewer than 6 words", None
    turns = _turns_for_meeting(conn, meeting_id)
    if not turns:
        return False, f"unknown meeting_id {meeting_id}", None
    nq = _norm(quote)
    for t in turns:
        if nq in _norm(t["text"]):
            return True, "ok", t
    return False, "quote not found verbatim in any turn of this meeting", None


def verify_answer_citations(conn, answer: dict):
    """Verify every citation in an {"citations": [...]} style dict. Returns list of results."""
    results = []
    for c in answer.get("citations", []):
        ok, reason, turn = verify_citation(conn, c.get("meeting_id", ""), c.get("quote", ""))
        results.append({"citation": c, "ok": ok, "reason": reason})
    return results


if __name__ == "__main__":
    conn = get_conn()
    ok, reason, turn = verify_citation(conn, "TS3009d", "Titanium-coloured plastic")
    print(ok, reason, turn)
    ok, reason, turn = verify_citation(conn, "TS3009d", "this quote does not exist anywhere")
    print(ok, reason, turn)
