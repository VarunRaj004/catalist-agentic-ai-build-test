"""Answer the 25 official questions, split across 2 Groq keys (if GROQ_API_KEY_2 is set), with
each answer cached to the DB as soon as it lands -- so stopping/restarting mid-run never loses
completed work, same lesson learned from the ledger stage.
"""
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from llm_client import LLMClient, _load_dotenv  # noqa: E402
from cost import make_logger  # noqa: E402
from qa_agent import answer_question  # noqa: E402

_load_dotenv()
PRINT_LOCK = threading.Lock()

AGENT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = AGENT_DIR.parent / "data_raw" / "questions.json"


def _run_from_queue(q_queue, api_key, label):
    import queue as queue_mod
    conn = get_conn()
    log = make_logger(conn)
    client = LLMClient(api_key=api_key, log_fn=log)
    while True:
        try:
            q = q_queue.get_nowait()
        except queue_mod.Empty:
            break
        try:
            result = answer_question(conn, client, q["question"], team=q["team"], verbose=False)
            if not result["citations"]:
                # answer_question() degrades gracefully to an empty answer on LLM failure rather
                # than raising -- that must NOT look like a cached success, or it's never retried.
                with PRINT_LOCK:
                    print(f"[{label}] {q['question_id']} ({q['team']}): 0 citations, "
                          f"leaving unprocessed for retry")
                continue
            conn.execute(
                "INSERT OR REPLACE INTO answer_cache (question_id, answer, citations_json) "
                "VALUES (?, ?, ?)",
                (q["question_id"], result["answer"], json.dumps(result["citations"])),
            )
            conn.commit()
            with PRINT_LOCK:
                print(f"[{label}] {q['question_id']} ({q['team']}): done, "
                      f"{len(result['citations'])} citations")
        except Exception as e:  # noqa: BLE001
            with PRINT_LOCK:
                print(f"[{label}] {q['question_id']}: ERROR {e}")
    conn.close()


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    conn = get_conn()
    cached = {r["question_id"] for r in conn.execute("SELECT question_id FROM answer_cache").fetchall()}
    conn.close()

    remaining = [q for q in questions if q["question_id"] not in cached]
    print(f"{len(remaining)}/{len(questions)} questions remaining (already cached: {len(cached)})")

    import queue as queue_mod
    q_queue = queue_mod.Queue()
    for q in remaining:
        q_queue.put(q)

    key2 = os.environ.get("GROQ_API_KEY_2")
    provider = os.environ.get("LLM_PROVIDER", "openrouter")

    if provider == "groq" and key2:
        print("Shared queue, 2 Groq keys")
        t1 = threading.Thread(target=_run_from_queue, args=(q_queue, None, "key1"))
        t2 = threading.Thread(target=_run_from_queue, args=(q_queue, key2, "key2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    else:
        _run_from_queue(q_queue, None, "key1")

    print("Done.")


if __name__ == "__main__":
    main()
