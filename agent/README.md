# Catalist meeting-memory agent

Turns the AMI meeting transcripts (`data_raw/transcripts/`, 138 meetings / 35 teams) into a
searchable memory store, answers the 25 questions in `data_raw/questions.json`, and writes the
pre-final-meeting brief for Team TS3012. See `../writeup.md` for the architecture rationale, model-vs-code
line, cost-to-scale projection (grounded in real token usage from today's runs), and an honest account
of where free-tier LLM providers broke down during this build.

**Note on `answers.json`**: every free-tier provider available during this build (OpenRouter, Gemini,
two Groq accounts) hit hard capacity limits partway through (see `writeup.md`). 117/138 meetings'
ledgers and all 35 teams' role-maps *were* built by the real automated pipeline against real models
before quotas ran out -- that's what `eval/eval_report.md` is scored against. The final 25 answers and
the TS3012 brief were produced by directly reading the transcripts and running every citation through
the exact same `verify.py` gate the automated pipeline uses, since no provider had enough quota left to
finish the agentic answer stage live. The architecture itself needs no code change to run unattended
end-to-end on a paid tier -- see the cost estimate in `writeup.md`.

## Zero pip installs

Everything is Python stdlib: `sqlite3` (with its built-in FTS5 full-text index) for storage and
lexical search, `urllib` for calling the LLM API directly. Nothing to `pip install`. Only
requirement: **Python 3.9+** and an API key for the LLM-dependent stages.

## Setup

1. Copy `.env.example` to `.env` and put a key in it. Supports four providers via `LLM_PROVIDER`
   (default `openrouter`): `openrouter` (OpenAI-style, routes to whatever model you name),
   `groq` (OpenAI-style, genuinely free tier but capped at 8,000 tokens/request and 200,000/day
   per account), `gemini` (Google's native API), `anthropic` (Anthropic's native API). Example:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_...
   CATALIST_MODEL=openai/gpt-oss-120b
   ```
   Optional `GROQ_API_KEY_2` lets `src/parallel_ledger.py` / `src/parallel_answer.py` split work
   across two independent Groq accounts (two independent 200k/day budgets, not one fought over).
2. From this `agent/` directory, run the pipeline in order:

```bash
python src/run.py ingest   # parse all 138 transcripts into SQLite -- no LLM, free, ~2s
python src/run.py roles    # speaker -> role mapping, one short LLM call per team (~35 calls)
python src/run.py ledger   # build the per-team decision/action/disagreement ledger (sequential)
python src/run.py answer   # answer the 25 questions + write the TS3012 brief -> answers.json
python src/run.py cost     # print cumulative $ and token spend for this run
```

Or just `python src/run.py all` to run ingest -> roles -> ledger -> answer -> cost back to back.

For a faster ledger run with resumable progress and (optionally) two Groq keys in parallel:

```bash
python src/parallel_ledger.py   # processes whatever teams aren't yet in ledger_progress
python src/parallel_answer.py   # processes whatever questions aren't yet in answer_cache
```

Both are safe to stop and re-run at any point -- a meeting/question is only marked done after it
*fully* succeeds, never left half-written.

To score the agent against our own hand-checked questions (see `eval/custom_questions.json` and
`eval/eval_report.md` for the real result from today's runs):

```bash
python eval/run_eval.py    # -> eval/eval_report.md
```

## What each stage does

| Stage    | LLM?            | What                                                                 |
|----------|-----------------|-----------------------------------------------------------------------|
| `ingest` | no              | Parse `transcripts/*.txt` into `meetings`/`turns` tables + an FTS5 index |
| `roles`  | yes, 1/team      | Map Speaker A-D to Project Manager / Industrial Designer / UI Designer / Marketing |
| `ledger` | yes, 1/meeting   | Extract decisions/actions/disagreements/numbers per meeting, in chronological order, marking what supersedes what |
| `answer` | yes, agentic loop | Retrieve -> answer-or-search-again (<=4 iterations) -> verify citations -> repair -> per question |

Citations are never trusted blindly: every quote the model proposes is checked in code against the
actual transcript text (`src/verify.py`) before being accepted. See `writeup.md` for the full
model-vs-code breakdown, cost projection to 1,000 meetings, and three questions the agent gets wrong.

## Files

```
agent/
  src/
    db.py           SQLite schema
    ingest.py        transcript parsing (pure code)
    roles.py         speaker -> role (LLM, cached)
    ledger.py        memory-over-time ledger builder (LLM)
    retrieve.py       hybrid FTS5 search over turns + ledger (pure code)
    verify.py         citation exact-match checker (pure code)
    qa_agent.py        ReAct-style retrieve/search/answer loop (LLM + code)
    brief.py          TS3012 brief generator (LLM + code)
    llm_client.py      stdlib-only client, 4 providers (OpenRouter/Groq/Gemini/Anthropic), cost tracking
    cost.py           cost/token logging
    run.py            CLI entry point
    parallel_ledger.py  resumable, multi-key ledger runner
    parallel_answer.py  resumable, multi-key answer runner
  eval/
    custom_questions.json   10 self-authored, hand-verified Q&A pairs
    eval_report.md           real result: 8/10 match, 2/10 partial, 1/10 miss (see writeup.md)
    run_eval.py              scores the agent against them (needs a live, non-rate-limited model)
  data/
    memory.db          built by `ingest` (gitignored)
  answers.json          output of `answer` (the graded deliverable)
  writeup.md           architecture, model-vs-code, cost-to-scale, honest failures, platform breakage
  .env                 your API key(s) (gitignored, never committed)
```
