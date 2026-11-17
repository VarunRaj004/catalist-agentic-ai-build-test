"""Run build_ledger_for_team across teams concurrently.

Safe because teams are independent (team X's ledger never depends on team Y's), while each
team's own meetings stay strictly sequential inside build_ledger_for_team (meeting b needs a's
resulting ledger state). Each worker thread gets its OWN sqlite connection -- safe with the
WAL + busy_timeout settings in db.py, which make concurrent writers wait instead of crashing.
Per-meeting resumability (ledger_progress) means a team that's partway done just continues from
where it left off, and re-running this after an interruption never duplicates finished work.

Groq's free tier caps tokens-per-minute PER ACCOUNT, not per connection -- so concurrency within
one key doesn't help (it just means more requests racing the same tiny budget and 429ing). If a
second GROQ_API_KEY_2 is set (a second free account = its own independent quota), we split the
remaining teams into two groups and run each group sequentially against its own key -- two
independent budgets, not one budget fought over by many threads.
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_conn  # noqa: E402
from llm_client import LLMClient, _load_dotenv  # noqa: E402
from cost import make_logger  # noqa: E402
from ledger import build_ledger_for_team  # noqa: E402

_load_dotenv()
PRINT_LOCK = threading.Lock()


def _run_from_queue(team_queue, api_key, label):
    """Pull teams from a shared queue until empty -- so if one team takes unusually long (lots
    of chunks/repairs), the OTHER key doesn't sit idle on a fixed pre-assigned half; it just keeps
    pulling the next team instead."""
    import queue as queue_mod
    conn = get_conn()
    log = make_logger(conn)
    client = LLMClient(api_key=api_key, log_fn=log)
    while True:
        try:
            team = team_queue.get_nowait()
        except queue_mod.Empty:
            break
        try:
            build_ledger_for_team(conn, client, team, log_fn=log, verbose=True)
            with PRINT_LOCK:
                print(f"[{label}] {team}: done")
        except Exception as e:  # noqa: BLE001
            with PRINT_LOCK:
                print(f"[{label}] {team}: ERROR {e}")
    conn.close()


def main():
    conn = get_conn()
    all_meetings = {
        r["meeting_id"]: r["team"]
        for r in conn.execute("SELECT meeting_id, team FROM meetings").fetchall()
    }
    done = {r["meeting_id"] for r in conn.execute("SELECT meeting_id FROM ledger_progress").fetchall()}
    conn.close()

    remaining_teams = sorted({team for mid, team in all_meetings.items() if mid not in done})
    print(f"{len(remaining_teams)} teams with unfinished meetings: {remaining_teams}")

    key2 = os.environ.get("GROQ_API_KEY_2")
    provider = os.environ.get("LLM_PROVIDER", "openrouter")
    t0 = time.time()

    import queue as queue_mod
    team_queue = queue_mod.Queue()
    for team in remaining_teams:
        team_queue.put(team)

    if provider == "groq" and key2:
        # Groq free tier: TPM is a per-ACCOUNT token budget, so two independent keys = two
        # independent budgets. Concurrency within one key would just mean more requests racing
        # the same tiny budget. Gated on provider=="groq" too -- GROQ_API_KEY_2 being set in
        # .env from an earlier run must never get fed to a different provider as its key.
        print(f"Shared queue, 2 keys pulling from {len(remaining_teams)} teams")
        t1 = threading.Thread(target=_run_from_queue, args=(team_queue, None, "key1"))
        t2 = threading.Thread(target=_run_from_queue, args=(team_queue, key2, "key2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    elif provider == "openrouter":
        # OpenRouter free tier limits REQUESTS/min, not tokens/min -- unlike Groq, concurrency
        # on the SAME key is actually useful here, since each call is slow (~20-30s, a reasoning
        # model) but the request-rate ceiling (20/min) has plenty of headroom under that latency.
        n_workers = 3
        print(f"Shared queue, {n_workers} concurrent workers on one OpenRouter key, "
              f"{len(remaining_teams)} teams")
        threads = [
            threading.Thread(target=_run_from_queue, args=(team_queue, None, f"w{i+1}"))
            for i in range(n_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        _run_from_queue(team_queue, None, "key1")

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
