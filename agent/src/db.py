"""SQLite schema and connection helper for the meeting memory store.

Everything here is plain code: no LLM involved in defining or writing this schema.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id TEXT PRIMARY KEY,
    team TEXT NOT NULL,
    letter TEXT NOT NULL,
    meeting_number INTEGER,
    source_file TEXT,
    n_turns INTEGER
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL,
    team TEXT NOT NULL,
    letter TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    start_time TEXT,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id)
);
CREATE INDEX IF NOT EXISTS idx_turns_meeting ON turns(meeting_id);
CREATE INDEX IF NOT EXISTS idx_turns_team ON turns(team);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    text, team UNINDEXED, meeting_id UNINDEXED, turn_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS roles (
    team TEXT NOT NULL,
    speaker TEXT NOT NULL,
    role TEXT,
    confidence TEXT,
    source_meeting TEXT,
    PRIMARY KEY (team, speaker)
);

CREATE TABLE IF NOT EXISTS ledger (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    kind TEXT NOT NULL,            -- decision | action | disagreement | number
    subject TEXT,
    detail TEXT,
    owner TEXT,
    status TEXT,                   -- open | closed | superseded
    meeting_id TEXT NOT NULL,
    letter TEXT NOT NULL,
    quote TEXT NOT NULL,
    turn_idx INTEGER,
    supersedes_id INTEGER,
    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id)
);
CREATE INDEX IF NOT EXISTS idx_ledger_team ON ledger(team);

CREATE VIRTUAL TABLE IF NOT EXISTS ledger_fts USING fts5(
    subject, detail, team UNINDEXED, entry_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS answer_cache (
    question_id TEXT PRIMARY KEY,
    answer TEXT,
    citations_json TEXT  -- JSON list of {"meeting_id", "quote"}
);

CREATE TABLE IF NOT EXISTS ledger_progress (
    meeting_id TEXT PRIMARY KEY  -- marks a meeting as already processed by build_ledger,
                                  -- even if it produced zero entries, so reruns (e.g. resuming
                                  -- across days under a rate cap) don't redo or duplicate work
);

CREATE TABLE IF NOT EXISTS cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    usd_cost REAL,
    ts TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # timeout=30: SQLite's default busy-timeout is ~0, so any brief lock contention (another
    # connection, a backup/AV scan touching the file) raises "database is locked" immediately
    # instead of waiting. WAL mode also lets readers and a writer coexist instead of blocking.
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


def reset_db(db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
