"""Orchestration CLI.

    python run.py ingest          # parse transcripts -> SQLite (no LLM, free)
    python run.py roles           # speaker->role mapping (LLM, ~35 short calls)
    python run.py ledger          # build the memory ledger (LLM, ~138 calls)
    python run.py answer          # answer questions.json's 25 questions + the TS3012 brief -> answers.json
    python run.py eval            # score the agent against eval/custom_questions.json
    python run.py all             # ingest -> roles -> ledger -> answer
    python run.py cost            # print cumulative $ / token spend so far
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from ingest import ingest_all  # noqa: E402
from roles import build_roles  # noqa: E402
from ledger import build_ledger  # noqa: E402
from qa_agent import answer_question  # noqa: E402
from brief import build_brief  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from cost import make_logger, summarize  # noqa: E402

AGENT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = AGENT_DIR.parent / "data_raw" / "questions.json"
ANSWERS_PATH = AGENT_DIR / "answers.json"


def cmd_ingest():
    stats = ingest_all()
    print(f"Ingested {stats['meetings']} meetings, {stats['teams']} teams, {stats['turns']} turns.")


def cmd_roles():
    conn = get_conn()
    log = make_logger(conn)
    t0 = time.time()
    build_roles(log_fn=log, conn=conn)
    print(f"Roles built in {time.time() - t0:.1f}s")


def cmd_ledger():
    t0 = time.time()
    conn = get_conn()
    log = make_logger(conn)
    build_ledger(log_fn=log, conn=conn)
    print(f"Ledger built in {time.time() - t0:.1f}s")


def cmd_answer():
    conn = get_conn()
    log = make_logger(conn)
    client = LLMClient(log_fn=log)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    answers = []
    t0 = time.time()
    for q in questions:
        print(f"[answer] {q['question_id']} ({q['team']}) ...")
        result = answer_question(conn, client, q["question"], team=q["team"], verbose=False)
        answers.append({
            "question_id": q["question_id"],
            "answer": result["answer"],
            "citations": result["citations"],
        })
    print(f"[answer] 25 questions done in {time.time() - t0:.1f}s")

    print("[brief] building TS3012 brief ...")
    brief_result = build_brief("TS3012", log_fn=log, verbose=True, conn=conn)

    out = {"answers": answers, "brief": brief_result["brief"]}
    ANSWERS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {ANSWERS_PATH}")


def cmd_cost():
    print(json.dumps(summarize(), indent=2))


def cmd_all():
    cmd_ingest()
    cmd_roles()
    cmd_ledger()
    cmd_answer()
    cmd_cost()


COMMANDS = {
    "ingest": cmd_ingest,
    "roles": cmd_roles,
    "ledger": cmd_ledger,
    "answer": cmd_answer,
    "cost": cmd_cost,
    "all": cmd_all,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
