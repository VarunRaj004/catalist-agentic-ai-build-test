"""Ingestion: parse transcripts/*.txt into the SQLite memory store.

Pure code, no LLM. The transcript format is fixed and mechanical (a 7-line
header, then blocks of "HH:MM:SS Speaker X" followed by exactly one text
line, separated by blank lines) so a model has nothing to decide here --
regex + string splitting is strictly more reliable and orders of magnitude
cheaper than asking an LLM to parse it turn by turn.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn, reset_db  # noqa: E402

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "data_raw" / "transcripts"

TURN_HEADER_RE = re.compile(r"^(\d\d:\d\d:\d\d) Speaker ([A-D])$")
MEETING_ID_RE = re.compile(r"^Meeting:\s*(\S+)")
MEETING_NUM_RE = re.compile(r"^Meeting number:\s*(\d+)\s*\(([a-d])\)")


def parse_transcript(path: Path):
    """Return (meeting_id, letter, meeting_number, turns) for one file.

    turns is a list of dicts: {turn_idx, start_time, speaker, text}
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.split("\n")

    meeting_id = path.stem
    letter = meeting_id[-1]
    meeting_number = None

    body_start = 0
    for i, line in enumerate(lines[:10]):
        m = MEETING_ID_RE.match(line.strip())
        if m:
            meeting_id = m.group(1)
        m = MEETING_NUM_RE.match(line.strip())
        if m:
            meeting_number = int(m.group(1))
            letter = m.group(2)

    turns = []
    i = 0
    turn_idx = 0
    n = len(lines)
    while i < n:
        header = TURN_HEADER_RE.match(lines[i].strip())
        if header:
            start_time, speaker = header.group(1), header.group(2)
            j = i + 1
            text_parts = []
            while j < n and lines[j].strip() != "" and not TURN_HEADER_RE.match(lines[j].strip()):
                text_parts.append(lines[j].strip())
                j += 1
            text = " ".join(t for t in text_parts if t)
            if text:
                turns.append({
                    "turn_idx": turn_idx,
                    "start_time": start_time,
                    "speaker": speaker,
                    "text": text,
                })
                turn_idx += 1
            i = j
        else:
            i += 1

    return meeting_id, letter, meeting_number, turns


def ingest_all(fresh: bool = True):
    if fresh:
        reset_db()
    conn = get_conn()
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No transcripts found under {TRANSCRIPTS_DIR}")

    total_turns = 0
    for path in files:
        meeting_id, letter, meeting_number, turns = parse_transcript(path)
        team = meeting_id[:-1]
        conn.execute(
            "INSERT OR REPLACE INTO meetings (meeting_id, team, letter, meeting_number, source_file, n_turns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (meeting_id, team, letter, meeting_number, path.name, len(turns)),
        )
        rows = [
            (meeting_id, team, letter, t["turn_idx"], t["start_time"], t["speaker"], t["text"])
            for t in turns
        ]
        conn.executemany(
            "INSERT INTO turns (meeting_id, team, letter, turn_idx, start_time, speaker, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        # populate FTS index with the turn_id we just assigned
        cur = conn.execute(
            "SELECT turn_id, text, team, meeting_id FROM turns WHERE meeting_id = ? ORDER BY turn_idx",
            (meeting_id,),
        )
        fts_rows = [(r["text"], r["team"], r["meeting_id"], r["turn_id"]) for r in cur.fetchall()]
        conn.executemany(
            "INSERT INTO turns_fts (text, team, meeting_id, turn_id) VALUES (?, ?, ?, ?)",
            fts_rows,
        )
        total_turns += len(turns)

    conn.commit()
    n_meetings = conn.execute("SELECT COUNT(*) AS c FROM meetings").fetchone()["c"]
    n_teams = conn.execute("SELECT COUNT(DISTINCT team) AS c FROM meetings").fetchone()["c"]
    conn.close()
    return {"meetings": n_meetings, "teams": n_teams, "turns": total_turns}


if __name__ == "__main__":
    stats = ingest_all()
    print(f"Ingested {stats['meetings']} meetings, {stats['teams']} teams, {stats['turns']} turns.")
