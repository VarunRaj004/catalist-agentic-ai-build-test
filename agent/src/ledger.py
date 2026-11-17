"""Builds the per-team decision/action/disagreement ledger.

This is the "memory over time" mechanism. For each team we walk its meetings
in chronological order (a -> b -> c -> d) and, for each meeting, show the
model the transcript AND the team's ledger so far, and ask it to emit new or
updated entries -- including marking which earlier entries a new one
supersedes. That supersession judgment (e.g. "the lithium battery decision
from meeting b is overturned by the solar-panel decision in meeting c") is
semantic and context-dependent, which is why it is an LLM call and not a
string match. Everything downstream (verifying the quote is real, writing it
to SQLite, flipping old rows to status='superseded') is plain code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from llm_client import LLMClient, LLMError  # noqa: E402
from verify import verify_citation  # noqa: E402

SYSTEM = """You maintain a running memory ledger for a product design team (they are designing a TV \
remote control over several meetings). You will be shown one meeting's transcript and the team's \
existing ledger entries from earlier meetings. Extract NEW ledger entries for THIS meeting only.

An entry has:
- "kind": one of "decision", "action", "disagreement", "number"
- "subject": a short topic label, e.g. "case material", "energy source", "selling price"
- "detail": one sentence of what was decided / asked / disputed / what the number was
- "owner": the speaker letter (A/B/C/D) most responsible, or null
- "status": "open" or "closed" (closed = resolved/decided by the end of this meeting; open = still pending)
- "turn_idx": the integer turn_idx of ONE turn in THIS transcript that best supports the entry
- "quote": copied EXACTLY (verbatim, same words, same order) from that turn's text, at least 6 words long
- "supersedes": a list of entry_id values from the EXISTING LEDGER (given below) that this new entry \
overturns or replaces (e.g. a later material choice replacing an earlier one on the same subject). \
Empty list if nothing is overturned.

Only extract things that are actually new in this meeting (new decisions, new action items, new \
disagreements, new numbers/costs/prices, or a reversal of something from before). Do not restate \
existing ledger entries that are still unchanged. It is fine to return few or no entries for a quiet meeting.

Reply with ONLY a JSON object: {"entries": [ { ... }, ... ]}
"""


def _format_existing_ledger(entries):
    if not entries:
        return "(none yet -- this is the team's first meeting)"
    lines = []
    for e in entries:
        # No quote here -- it's not needed to decide new entries or supersession, and dropping it
        # measurably shrinks the prompt for teams with many accumulated entries (see chunking below).
        lines.append(
            f"- entry_id={e['entry_id']} [{e['kind']}/{e['status']}] {e['subject']}: {e['detail']} "
            f"(from {e['meeting_id']})"
        )
    return "\n".join(lines)


def _format_transcript(turns):
    return "\n".join(f"[{t['turn_idx']}] Speaker {t['speaker']}: {t['text']}" for t in turns)


# Conservative chars-per-token estimate (actual tokenizers run ~3.3-4 chars/token for English;
# using 3.5 here errs toward overestimating tokens, i.e. chunking a bit more than strictly needed,
# which is safer than hitting a provider's hard per-request token ceiling (seen in practice: Groq's
# free tier rejects any single request over ~8000 tokens with a 413, regardless of model).
CHARS_PER_TOKEN = 3.5
TARGET_PROMPT_TOKENS = 2800  # leave real headroom under Groq's 8000 TPM ceiling even when the
# rolling window still has leftover usage from a recent call -- 4500 was hugging the edge too
# closely and causing frequent 429s on otherwise-valid requests.


def _chunk_turns(turns, existing_ledger_text, system_len):
    """Split turns into chunks small enough that each (system + existing-ledger + chunk) prompt
    stays under TARGET_PROMPT_TOKENS. One LLM call per chunk; chunks of the same meeting are still
    processed in order, each seeing what the previous chunks of THIS meeting already extracted."""
    budget_chars = int(TARGET_PROMPT_TOKENS * CHARS_PER_TOKEN) - system_len - len(existing_ledger_text)
    budget_chars = max(budget_chars, 2000)  # never chunk down to nothing even if context is huge

    chunks, cur_chunk, cur_len = [], [], 0
    for t in turns:
        line = f"[{t['turn_idx']}] Speaker {t['speaker']}: {t['text']}"
        line_len = len(line) + 1
        if cur_chunk and cur_len + line_len > budget_chars:
            chunks.append(cur_chunk)
            cur_chunk, cur_len = [], 0
        cur_chunk.append(t)
        cur_len += line_len
    if cur_chunk:
        chunks.append(cur_chunk)
    return chunks


def _extract_from_chunk(conn, client, meeting_id, chunk_turns, turns_by_idx, context_text, verbose):
    """One LLM call (+ repair loop) over one chunk of a meeting's turns. Returns a list of
    validated entry dicts (not yet written to DB), (repaired, rejected) counts, and whether the
    call itself succeeded -- a failed call must NOT look the same as "the model found nothing
    here", or the caller can't tell a quiet chunk from a lost one."""
    user = (
        f"Meeting: {meeting_id}.\n\n{context_text}\n\n"
        f"THIS MEETING'S TRANSCRIPT{' (part of it -- see note above)' if 'part' in context_text else ''}:\n"
        f"{_format_transcript(chunk_turns)}"
    )
    try:
        # retries=2: on a tight free-tier TPM budget, one retry (one real wait) is enough to
        # recover from transient contention; exhausting all 4 default retries can stack to ~7.5
        # minutes of silence for one chunk, which isn't worth it -- better to fail this chunk
        # quickly (the caller discards the whole meeting rather than committing a gap) than
        # stall the whole run on one stubborn request.
        result = client.call_json(SYSTEM, user, max_tokens=1500, stage="ledger", retries=2)
    except LLMError as e:
        if verbose:
            print(f"[ledger] {meeting_id}: LLM error on chunk, failing meeting for retry later ({e})")
        return [], 0, 0, False

    entries = result.get("entries", []) if isinstance(result, dict) else []
    validated, repaired, rejected = [], 0, 0

    for e in entries:
        quote = (e.get("quote") or "").strip()
        turn_idx = e.get("turn_idx")
        ok, reason, matched_turn = verify_citation(conn, meeting_id, quote)

        if not ok:
            candidate_turn = turns_by_idx.get(turn_idx)
            if candidate_turn is not None:
                repair_user = (
                    f"Your quote \"{quote}\" was not found verbatim in turn [{turn_idx}].\n"
                    f"The actual text of turn [{turn_idx}] (Speaker {candidate_turn['speaker']}) is:\n"
                    f"\"{candidate_turn['text']}\"\n\n"
                    "Reply with ONLY a JSON object: {\"quote\": \"an exact verbatim substring of "
                    "at least 6 words copied from that turn's text, that best supports the entry\"}"
                )
                try:
                    fix = client.call_json(SYSTEM, repair_user, max_tokens=300, stage="ledger_repair", retries=2)
                    fixed_quote = (fix.get("quote") or "").strip()
                    ok2, reason2, matched_turn2 = verify_citation(conn, meeting_id, fixed_quote)
                    if ok2:
                        quote = fixed_quote
                        matched_turn = matched_turn2
                        ok = True
                        repaired += 1
                except LLMError:
                    pass

        if not ok:
            rejected += 1
            continue

        validated.append({
            "kind": e.get("kind", "decision"), "subject": e.get("subject", ""),
            "detail": e.get("detail", ""), "owner": e.get("owner"),
            "status": e.get("status", "open"), "quote": quote,
            "turn_idx": matched_turn["turn_idx"], "supersedes": e.get("supersedes", []) or [],
        })

    return validated, repaired, rejected, True


def build_ledger_for_team(conn, client: LLMClient, team: str, log_fn=None, verbose=True):
    meetings = conn.execute(
        "SELECT meeting_id, letter FROM meetings WHERE team = ? ORDER BY letter", (team,)
    ).fetchall()

    for m in meetings:
        meeting_id, letter = m["meeting_id"], m["letter"]
        if conn.execute(
            "SELECT 1 FROM ledger_progress WHERE meeting_id = ?", (meeting_id,)
        ).fetchone():
            if verbose:
                print(f"[ledger] {meeting_id}: already processed, skipping")
            continue
        turns = conn.execute(
            "SELECT turn_idx, speaker, text FROM turns WHERE meeting_id = ? ORDER BY turn_idx",
            (meeting_id,),
        ).fetchall()
        turns = [dict(t) for t in turns]
        turns_by_idx = {t["turn_idx"]: t for t in turns}
        existing = conn.execute(
            "SELECT entry_id, kind, subject, detail, status, meeting_id, quote FROM ledger "
            "WHERE team = ? AND status != 'superseded' ORDER BY entry_id",
            (team,),
        ).fetchall()
        existing = [dict(e) for e in existing]
        existing_text = _format_existing_ledger(existing)

        chunks = _chunk_turns(turns, existing_text, len(SYSTEM))

        # Accumulate across chunks in memory; only commit to DB once the WHOLE meeting succeeds,
        # so a mid-meeting failure (e.g. a later chunk hits a transient error) can't leave the
        # meeting half-written while still being marked "done" by ledger_progress.
        all_validated = []
        meeting_summary_lines = []  # informational only, not supersedable (see note in docstring)
        total_repaired, total_rejected, chunk_failed = 0, 0, False

        for i, chunk in enumerate(chunks):
            note = ""
            if len(chunks) > 1:
                note = (
                    f"(This meeting's transcript is long and split into parts for processing; "
                    f"this is part {i + 1} of {len(chunks)}. Already noted earlier in this same "
                    f"meeting -- informational only, do not re-extract or try to supersede these "
                    f"by id, just avoid repeating them:\n"
                    f"{chr(10).join(meeting_summary_lines) if meeting_summary_lines else '(none yet)'})\n\n"
                )
            context_text = f"{note}EXISTING LEDGER (from earlier meetings):\n{existing_text}"

            validated, repaired, rejected, success = _extract_from_chunk(
                conn, client, meeting_id, chunk, turns_by_idx, context_text, verbose
            )
            if not success:
                # Don't burn more calls on a meeting we're about to discard anyway -- leave it
                # fully unprocessed (nothing inserted, not marked done) for a clean retry later.
                chunk_failed = True
                break
            total_repaired += repaired
            total_rejected += rejected
            all_validated.extend(validated)
            meeting_summary_lines.extend(f"- [{v['kind']}] {v['subject']}: {v['detail']}" for v in validated)

        if chunk_failed:
            if verbose:
                print(f"[ledger] {meeting_id}: a chunk failed, leaving meeting unprocessed for retry")
            continue

        for v in all_validated:
            cur = conn.execute(
                "INSERT INTO ledger (team, kind, subject, detail, owner, status, meeting_id, letter, "
                "quote, turn_idx, supersedes_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    team, v["kind"], v["subject"], v["detail"], v["owner"], v["status"],
                    meeting_id, letter, v["quote"], v["turn_idx"], None,
                ),
            )
            new_id = cur.lastrowid
            conn.execute(
                "INSERT INTO ledger_fts (subject, detail, team, entry_id) VALUES (?, ?, ?, ?)",
                (v["subject"], v["detail"], team, new_id),
            )
            for sup_id in v["supersedes"]:
                conn.execute(
                    "UPDATE ledger SET status = 'superseded' WHERE entry_id = ? AND team = ?",
                    (sup_id, team),
                )

        conn.execute("INSERT OR IGNORE INTO ledger_progress (meeting_id) VALUES (?)", (meeting_id,))
        conn.commit()
        if verbose:
            chunk_note = f" across {len(chunks)} chunks" if len(chunks) > 1 else ""
            print(
                f"[ledger] {meeting_id}: +{len(all_validated)} entries{chunk_note} "
                f"({total_repaired} repaired, {total_rejected} dropped)"
            )


def build_ledger(log_fn=None, only_team: str = None, conn=None):
    # Accept an existing connection so a caller that also needs it for cost-logging shares the
    # same one -- two separate sqlite3 connections to the same file in one process is exactly
    # what caused the "database is locked" contention we hit in practice.
    own_conn = conn is None
    conn = conn or get_conn()
    client = LLMClient(log_fn=log_fn)
    teams = [only_team] if only_team else [
        r["team"] for r in conn.execute("SELECT DISTINCT team FROM meetings ORDER BY team").fetchall()
    ]
    for team in teams:
        build_ledger_for_team(conn, client, team, log_fn=log_fn)
    if own_conn:
        conn.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--team", default=None)
    args = p.parse_args()
    build_ledger(only_team=args.team)
