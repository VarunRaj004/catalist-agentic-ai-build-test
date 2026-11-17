"""Run the agent against our self-authored eval set and report what it got right/wrong.

Automated part (code): did the agent produce verified citations at all, and do the
meetings it cited overlap with the meetings our hand-checked ground truth relies on.
That's a proxy, not a full correctness check -- the actual right/wrong call on content
is made by a human (us) reading `eval_report.md` next to the ground truth, which is why
this script prints both side by side rather than trying to auto-grade free text.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from db import get_conn  # noqa: E402
from qa_agent import answer_question  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from cost import make_logger  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "custom_questions.json"
REPORT_PATH = Path(__file__).resolve().parent / "eval_report.md"


def main():
    conn = get_conn()
    log = make_logger(conn)
    client = LLMClient(log_fn=log)
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    lines = ["# Self-authored eval report\n"]
    n_cited_overlap = 0
    t0 = time.time()
    for q in data["questions"]:
        result = answer_question(conn, client, q["question"], team=q["team"], verbose=False)
        expected_meetings = {c["meeting_id"] for c in q["expected_citations"]}
        got_meetings = {c["meeting_id"] for c in result["citations"]}
        overlap = bool(expected_meetings & got_meetings)
        n_cited_overlap += overlap

        lines.append(f"## {q['id']} ({q['team']})")
        lines.append(f"**Question:** {q['question']}\n")
        lines.append(f"**Expected:** {q['expected_answer']}")
        lines.append(f"**Expected citation meetings:** {sorted(expected_meetings)}\n")
        lines.append(f"**Agent answer:** {result['answer']}")
        lines.append(f"**Agent citations:** {json.dumps(result['citations'])}")
        lines.append(f"**Cited-meeting overlap with ground truth:** {'YES' if overlap else 'NO'}\n")
        lines.append("---\n")

    elapsed = time.time() - t0
    lines.insert(1, f"Ran {len(data['questions'])} questions in {elapsed:.1f}s. "
                     f"Cited-meeting overlap with ground truth: {n_cited_overlap}/{len(data['questions'])}.\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Cited-meeting overlap: {n_cited_overlap}/{len(data['questions'])}")


if __name__ == "__main__":
    main()
