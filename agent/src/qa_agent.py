"""The question-answering agent: retrieve -> LLM decides search-more-or-answer -> verify -> repair.

This is the "real agent behaviour" piece: the model does not see the whole
corpus. Each iteration it only sees what plain-code retrieval handed it, and
it can ask for another retrieval pass with a refined query before
committing to an answer. The loop is bounded (MAX_ITERS) so it can't run
away, and every citation the model proposes is checked against the actual
transcript text in code before being accepted -- if a citation is wrong, we
show the model the real text and give it one chance to fix it rather than
silently keeping a bad quote or silently dropping the point.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from llm_client import LLMClient, LLMError  # noqa: E402
from retrieve import hybrid_search, team_from_text, team_meetings_chronological  # noqa: E402
from verify import verify_citation  # noqa: E402

MAX_ITERS = 4

SYSTEM = """You are answering a question about a product design team's meeting transcripts, using only \
evidence retrieved for you below -- you cannot see the full transcripts. The team met several times \
(letters a, b, c, d in order); later meetings can change or overturn what was decided earlier, and \
people often refer back to earlier decisions without re-stating them ("like we said", "the one we \
discussed"), so pay attention to meeting order and to the pre-digested LEDGER entries, which already \
track what is current.

At each step, decide:
- If the evidence below is enough to answer confidently, reply with:
  {"action": "answer", "answer": "1-2 plain sentences", "citations": [{"meeting_id": "...", "quote": "..."}]}
- Otherwise, reply with:
  {"action": "search", "query": "a short, different, more specific search query", "reason": "why"}

Citation rules (checked automatically, word for word): each quote must be copied EXACTLY (same words, \
same order) from ONE turn shown below, at least 6 words long, from the meeting you name as meeting_id. \
Never combine words from two different turns into one quote. Give 1 to 3 citations. Prefer quotes that \
contain the concrete fact (the decision, the name, the number), not filler.

Reply with ONLY the JSON object, nothing else.
"""


def _format_evidence(evidence_turns, evidence_ledger):
    lines = []
    if evidence_ledger:
        lines.append("LEDGER (pre-digested, already tracks what's current):")
        for e in evidence_ledger:
            lines.append(
                f"- [{e['kind']}/{e['status']}] meeting {e['meeting_id']}: {e['subject']} -- {e['detail']} "
                f"(quote: \"{e['quote']}\")"
            )
    lines.append("\nRAW TRANSCRIPT TURNS:")
    for t in evidence_turns:
        lines.append(f"- meeting {t['meeting_id']} (letter {t['letter']}), Speaker {t['speaker']}: \"{t['text']}\"")
    return "\n".join(lines)


def _dedup(items, key):
    seen = set()
    out = []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def answer_question(conn, client: LLMClient, question: str, team: str = None, verbose=True):
    team = team or team_from_text(question)
    meetings = team_meetings_chronological(conn, team) if team else []

    seed = hybrid_search(conn, question, team=team, top_k_turns=15, top_k_ledger=10)
    evidence_turns = list(seed["turns"])
    evidence_ledger = list(seed["ledger"])
    queries_tried = [question]

    final = None
    for it in range(MAX_ITERS):
        evidence_turns = _dedup(evidence_turns, lambda t: (t["meeting_id"], t["turn_idx"]))
        evidence_ledger = _dedup(evidence_ledger, lambda e: e["entry_id"])
        evidence_text = _format_evidence(evidence_turns, evidence_ledger)
        meeting_list = ", ".join(f"{m['meeting_id']}" for m in meetings) if meetings else "unknown"
        user = (
            f"Question (about team {team or 'unknown'}, meetings in order: {meeting_list}):\n"
            f"{question}\n\nEVIDENCE RETRIEVED SO FAR:\n{evidence_text}"
        )
        try:
            # retries=2: on a tight free-tier budget, don't let one call stall ~7.5 minutes --
            # fail this iteration fast and let the agent either answer with what it has or move on.
            step = client.call_json(SYSTEM, user, max_tokens=1500, stage="qa", retries=2)
        except LLMError as e:
            if verbose:
                print(f"[qa] LLM error on iter {it}: {e}")
            break

        action = step.get("action")
        if action == "search" and it < MAX_ITERS - 1:
            q = step.get("query") or question
            if q in queries_tried:
                q = question + " " + q
            queries_tried.append(q)
            more = hybrid_search(conn, q, team=team, top_k_turns=10, top_k_ledger=6)
            evidence_turns += more["turns"]
            evidence_ledger += more["ledger"]
            if verbose:
                print(f"[qa] iter {it}: searching again -> {q!r} ({step.get('reason', '')})")
            continue
        else:
            final = step
            break

    if final is None or final.get("action") != "answer":
        final = {"action": "answer", "answer": "", "citations": []}

    # verify + one repair round per citation
    clean_citations = []
    for c in final.get("citations", [])[:3]:
        mid = c.get("meeting_id", "")
        quote = c.get("quote", "")
        ok, reason, turn = verify_citation(conn, mid, quote)
        if not ok:
            # try to repair using whatever evidence turn is from that meeting and close in content
            candidates = [t for t in evidence_turns if t["meeting_id"] == mid]
            if candidates:
                repair_user = (
                    f'Your citation quote "{quote}" for meeting {mid} did not match any turn exactly.\n'
                    "Here are the actual turns from that meeting you retrieved:\n" +
                    "\n".join(f'- "{t["text"]}"' for t in candidates[:10]) +
                    '\n\nReply with ONLY {"quote": "an exact verbatim substring of at least 6 words, '
                    'copied from one of those turns"}'
                )
                try:
                    fix = client.call_json(SYSTEM, repair_user, max_tokens=500, stage="qa_repair", retries=2)
                    fixed_quote = (fix.get("quote") or "").strip()
                    ok2, reason2, turn2 = verify_citation(conn, mid, fixed_quote)
                    if ok2:
                        clean_citations.append({"meeting_id": mid, "quote": fixed_quote})
                        continue
                except LLMError:
                    pass
            if verbose:
                print(f"[qa] dropped unverifiable citation: {mid} / {quote!r} ({reason})")
            continue
        clean_citations.append({"meeting_id": mid, "quote": quote})

    return {
        "answer": final.get("answer", ""),
        "citations": clean_citations,
        "_debug": {"team": team, "iterations": it + 1, "queries": queries_tried},
    }


if __name__ == "__main__":
    conn = get_conn()
    client = LLMClient()
    out = answer_question(
        conn, client,
        "Team TS3009 planned a titanium back for their remote. Who first proposed replacing the "
        "titanium, what did they propose instead, and was the titanium kept in the end?",
        team="TS3009",
    )
    import json
    print(json.dumps(out, indent=2))
