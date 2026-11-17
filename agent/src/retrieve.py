"""Hybrid lexical retrieval over the memory store. Pure code, no LLM.

Two sources are searched: raw turns (ground truth for citations) and the
ledger (pre-digested decisions/actions/disagreements, cheap to scan and
already time-aware). FTS5's bm25() gives real ranking, not just substring
matching, with no extra dependency.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402

TEAM_RE = re.compile(r"\b(ES20\d\d|IS10\d\d|TS30\d\d)\b")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "what", "which",
    "who", "when", "did", "does", "do", "is", "was", "were", "are", "their", "its", "they",
    "team", "question", "meeting", "meetings",
}


def team_from_text(text: str):
    m = TEAM_RE.search(text)
    return m.group(1) if m else None


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query over its content words."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if not words:
        return None
    # dedupe keeping order
    seen = set()
    terms = []
    for w in words:
        if w not in seen:
            seen.add(w)
            terms.append(f'"{w}"')
    return " OR ".join(terms[:25])


def search_turns(conn, query: str, team: str = None, top_k: int = 15):
    fts_q = _fts_query(query)
    if not fts_q:
        return []
    sql = """
        SELECT t.meeting_id, t.team, t.letter, t.turn_idx, t.start_time, t.speaker, t.text,
               bm25(turns_fts) AS score
        FROM turns_fts
        JOIN turns t ON t.turn_id = turns_fts.turn_id
        WHERE turns_fts MATCH ?
    """
    params = [fts_q]
    if team:
        sql += " AND turns_fts.team = ?"
        params.append(team)
    sql += " ORDER BY score LIMIT ?"
    params.append(top_k)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_ledger(conn, query: str, team: str = None, top_k: int = 10):
    fts_q = _fts_query(query)
    if not fts_q:
        return []
    sql = """
        SELECT l.entry_id, l.team, l.kind, l.subject, l.detail, l.owner, l.status,
               l.meeting_id, l.letter, l.quote, l.supersedes_id,
               bm25(ledger_fts) AS score
        FROM ledger_fts
        JOIN ledger l ON l.entry_id = ledger_fts.entry_id
        WHERE ledger_fts MATCH ?
    """
    params = [fts_q]
    if team:
        sql += " AND ledger_fts.team = ?"
        params.append(team)
    sql += " ORDER BY score LIMIT ?"
    params.append(top_k)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def team_meetings_chronological(conn, team: str):
    rows = conn.execute(
        "SELECT meeting_id, letter FROM meetings WHERE team = ? ORDER BY letter", (team,)
    ).fetchall()
    return [dict(r) for r in rows]


def hybrid_search(conn, query: str, team: str = None, top_k_turns: int = 15, top_k_ledger: int = 10):
    return {
        "turns": search_turns(conn, query, team=team, top_k=top_k_turns),
        "ledger": search_ledger(conn, query, team=team, top_k=top_k_ledger),
    }


if __name__ == "__main__":
    conn = get_conn()
    res = hybrid_search(conn, "titanium case material plastic", team="TS3009")
    for t in res["turns"][:5]:
        print(t["meeting_id"], t["speaker"], round(t["score"], 2), t["text"][:80])
