"""Generates the pre-final-meeting brief for Team TS3012.

Mostly reads straight from the ledger (already time-aware: open vs
superseded, decision vs action vs disagreement) rather than re-deriving
"what's current" from scratch -- that's the point of building the ledger
during ingestion instead of redoing this reasoning at answer time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from llm_client import LLMClient, LLMError  # noqa: E402
from verify import verify_citation  # noqa: E402

SYSTEM = """You are writing a short internal brief for Team TS3012 to read before their fourth and \
final meeting, designing a TV remote control. You are given the team's full ledger (decisions, \
actions, disagreements, numbers) built from their first three meetings, already marked with which \
entries are current ("open" or "closed") vs superseded by a later decision.

Produce a list of brief "points", grouped into four sections, in this order:
- "decided": what has been decided (the current, non-superseded decisions)
- "changed": what changed along the way (a decision that replaced an earlier one, and why if known)
- "actions": open actions and who owns them
- "disagreements": disagreements still open (not resolved as of the last meeting)

Each point is 1-2 plain sentences, with 1-3 citations. Reply with ONLY a JSON object:
{"points": [{"section": "decided|changed|actions|disagreements", "text": "...",
"citations": [{"meeting_id": "...", "quote": "..."}]}]}
Citation rule: quotes must be copied EXACTLY from one ledger quote or transcript turn given below, at \
least 6 words, from the stated meeting. Every point must have at least one citation.
"""

SECTION_TITLES = {
    "decided": "What has been decided",
    "changed": "What changed along the way",
    "actions": "Open actions and owners",
    "disagreements": "Disagreements still open",
}


def _format_ledger_for_brief(entries):
    lines = []
    for e in entries:
        lines.append(
            f"- entry_id={e['entry_id']} [{e['kind']}/{e['status']}] meeting {e['meeting_id']}: "
            f"{e['subject']} -- {e['detail']} (owner: {e['owner']}) (quote: \"{e['quote']}\")"
        )
    return "\n".join(lines) if lines else "(ledger is empty)"


def build_brief(team: str = "TS3012", log_fn=None, verbose=True, conn=None):
    # Accept an existing connection so a caller that also needs it for cost-logging shares the
    # same one, instead of two sqlite3 connections to the same file fighting each other.
    conn = conn or get_conn()
    client = LLMClient(log_fn=log_fn)

    entries = conn.execute(
        "SELECT entry_id, kind, subject, detail, owner, status, meeting_id, quote FROM ledger "
        "WHERE team = ? ORDER BY entry_id", (team,)
    ).fetchall()
    entries = [dict(e) for e in entries]

    meetings = conn.execute(
        "SELECT meeting_id, letter FROM meetings WHERE team = ? ORDER BY letter", (team,)
    ).fetchall()
    last_meeting_known = [m["meeting_id"] for m in meetings]

    user = (
        f"Team {team}'s ledger from meetings {', '.join(last_meeting_known)} (they are about to have "
        f"their final meeting; write the brief for whichever meeting comes after these):\n\n"
        f"{_format_ledger_for_brief(entries)}"
    )

    try:
        result = client.call_json(SYSTEM, user, max_tokens=3500, stage="brief", retries=2)
    except LLMError as e:
        if verbose:
            print(f"[brief] LLM error: {e}")
        return {"brief": "", "citations": []}

    points = result.get("points", []) if isinstance(result, dict) else []
    by_section = {k: [] for k in SECTION_TITLES}
    all_citations = []
    dropped = 0

    for p in points:
        section = p.get("section")
        if section not in SECTION_TITLES:
            continue
        text = (p.get("text") or "").strip()
        verified = []
        for c in (p.get("citations", []) or [])[:3]:
            mid, quote = c.get("meeting_id", ""), c.get("quote", "")
            ok, reason, turn = verify_citation(conn, mid, quote)
            if ok:
                verified.append({"meeting_id": mid, "quote": quote})
            elif verbose:
                print(f"[brief] dropped unverifiable citation: {mid} / {quote!r} ({reason})")
        if not text or not verified:
            dropped += 1
            continue
        cite_str = "; ".join(f'{c["meeting_id"]}: "{c["quote"]}"' for c in verified)
        by_section[section].append(f"{text} ({cite_str})")
        all_citations.extend(verified)

    if verbose and dropped:
        print(f"[brief] dropped {dropped} point(s) with no verifiable citation")

    parts = []
    for key, title in SECTION_TITLES.items():
        if by_section[key]:
            parts.append(f"{title}:\n" + "\n".join(f"- {line}" for line in by_section[key]))
    brief_text = "\n\n".join(parts)

    return {"brief": brief_text, "citations": all_citations}


if __name__ == "__main__":
    import json
    print(json.dumps(build_brief(), indent=2))
