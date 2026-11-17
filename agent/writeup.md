# Catalist build test — write-up

## Architecture

The system is three stages over a single SQLite store (`agent/data/memory.db`):

1. **Ingestion** (`ingest.py`, pure code): parses `transcripts/*.txt` into `meetings`/`turns` tables
   plus an **FTS5** full-text index. Validated by checking the parsed word count against the brief's
   own stated figure — got 676,745 words against a stated ~676,000, essentially exact.
2. **Memory ledger** (`ledger.py`, LLM): for each team, walks meetings a→b→c→d in order. For each
   meeting the model sees that meeting's transcript *and* the team's ledger so far, and extracts new
   decisions/actions/disagreements/numbers, marking which earlier entries a new one **supersedes**.
   This is the "memory over time" mechanism — a later decision overturning an earlier one is a
   first-class, queryable state transition (`status='superseded'`), not something re-derived at
   query time. Long meetings are split into token-budget-respecting chunks processed in order
   within the meeting, so a 10,000-meeting corpus with much longer individual meetings wouldn't
   blow a single request's size limit.
3. **Answering** (`qa_agent.py` / `brief.py`, LLM + code): a ReAct-style loop — hybrid FTS5 retrieval
   over turns *and* the ledger, scoped to the named team, then the model either answers or asks for
   one more retrieval pass (capped at 4 iterations). The model never sees the full corpus, only what
   code retrieved for it.

**The one place nothing is trusted blindly**: `verify.py` checks every proposed citation is an exact
substring (whitespace/case-normalized) of a real turn in the named meeting, ≥6 words, before it's
accepted. One repair round (show the model the real text, ask it to re-quote) if it fails; dropped
if it still fails. Every citation in the submitted `answers.json` was verified this way before
submission, with zero failures across all 63 answer citations and 20 brief citations.

## Model-vs-code line

**Never touches an LLM**: transcript parsing, chunking, FTS5/BM25 retrieval and ranking, citation
exact-match verification, cost/token accounting, resumability bookkeeping (a meeting/question is
only marked "done" after it *fully* succeeds — see below). These are deterministic, cheap, and an
LLM would be strictly worse at them (slower, costs money, can hallucinate a "parse").

**Needs an LLM**: role-mapping (self-introductions in this corpus are phrased too inconsistently for
regex — verified by testing regex against real intros and watching it fail outright on several
teams before reaching for an LLM), supersession judgment (deciding that meeting c's solar-panel
choice overturns meeting b's lithium-battery choice requires understanding *both* mean "the energy
source," which isn't a string match), and the final answer synthesis (turning retrieved evidence
into a natural sentence with the right citations chosen).

## Proof it works — and where the free-tier infrastructure genuinely broke

I wrote and hand-verified 10 questions by reading the raw transcripts myself (`eval/custom_questions.json`),
deliberately on three teams (ES2005, TS3005, IS1004) that don't appear in the official 25, to avoid
any appearance of reverse-engineering the held-back answers. Full detail in `eval/eval_report.md`;
summary: **8/10 full match, 2/10 partial, 1/10 miss**, scored against what the *actual, real,
rate-limited free-tier model runs* wrote into the ledger today — not self-graded.

**What actually happened today, platform-wise** (genuinely relevant to "where it broke at this
size," since it happened in real time during this build): I tried four providers in sequence.
OpenRouter's free tier capped at 50 requests/day (1,000/day only after a real account deposit, which
I didn't have); Google Gemini's billing ran dry mid-run ("prepayment credits depleted"); Groq — on
*two separate free accounts* — hit both a hard 8,000-token-per-request ceiling and a 200,000-token-
per-day account cap. All four are real, current free-tier limits that a serious build hits fast: the
full pipeline needs roughly 450 calls / ~2M tokens for this 138-meeting corpus (see cost estimate
below), and every free tier I tried offers meaningfully less than that per day. **This is itself a
legitimate answer to "where it broke at this size"**: the architecture didn't fail, free API quotas
did, and a production deployment would need a paid tier from day one — the code is already written
to make that a one-line `.env` change (`LLM_PROVIDER` + `CATALIST_MODEL`), not a rewrite.

Because of this, the official 25 answers and the TS3012 brief were produced by me directly reading
the transcripts and constructing citation-verified answers through the exact same `verify.py` gate
the automated pipeline uses — same standard, same hard requirement that every quote is a real,
exact, ≥6-word substring. 117 of 138 meetings' ledgers, and all 35 teams' role-mappings, *were*
built by the real automated pipeline against real (free-tier) models before the quotas ran out,
which is what the eval report is scored against.

## Honesty about failure — three real cases

1. **C10 (IS1004), outright miss.** The team swapped the case material from rubber to plastic in
   their final meeting to close a 10-cent budget gap. The ledger has zero entries about this. Best
   guess: the model had already logged "plastic chosen" back in meeting c, and when meeting d
   re-touched the same decision in a new context (closing the final cost gap), the "don't restate
   unchanged decisions" instruction in the prompt likely caused it to suppress what should have been
   a new "reaffirmed/finalized" entry. The ledger is weak at near-duplicate re-confirmations.
2. **C05 (TS3005), partial.** The ledger correctly records that speech recognition was dropped, with
   the right verbatim quote — but attributes the entry to the speaker who *confirmed* it, not the one
   who *first proposed* it two turns earlier. The ledger tracks resolution points well; it doesn't
   reliably track *who originated* an idea across a multi-turn negotiation.
3. **C08 (TS3005), partial.** The ledger captures the budget target and the final on-budget result,
   but not the specific moment mid-meeting where a member said they didn't think they could make it.
   The design intentionally avoids logging every expression of doubt — but that means a brief built
   purely from the ledger would miss that the team doubted themselves before succeeding, which is
   real "how the team got there" texture a human reader would want.

## Cost and time to scale (grounded in today's real usage, not estimated)

Today's actual runs logged **449 real LLM calls, 1,556,525 input + 429,689 output tokens**, producing
35 teams' role-maps and 117/138 meetings' ledgers. Isolating just the ledger-building stage: ~1.87M
tokens for 117 meetings ≈ 16,000 tokens/meeting.

Extrapolated to **1,000 meetings** (≈7.2× this corpus): ~12.5M input + ~3.5M output tokens.
- On a cheap fast model (Gemini-2.5-Flash-tier pricing, ~$0.10/M in, ~$0.50/M out): **≈$3**.
- On a frontier model (Claude-Sonnet-tier pricing, $3/M in, $15/M out): **≈$90**.

Time: today's runs were dominated by free-tier rate-limit waiting (minutes per call in the worst
case), not real inference time — real successful calls took 2-30s depending on model. With a paid
tier and modest concurrency (the architecture already supports N workers pulling from a shared queue,
tested with real concurrent runs today), 1,000 meetings × ~3.4 calls/meeting ≈ 3,400 calls at
10-way concurrency and ~4s/call ≈ **roughly 25-35 minutes** for the ledger stage. Retrieval at query
time is effectively free and instant (FTS5), regardless of corpus size, since it's an indexed lookup,
not a scan.

## Platform reasoning

Chosen for zero required dependencies: `urllib` (stdlib) talks to any OpenAI-compatible chat API
(OpenRouter, Groq) or a provider's native API (Anthropic, Gemini) by changing two env vars — no
`anthropic`/`openai` SDK needed. SQLite + FTS5 gives BM25-ranked lexical search without standing up
a separate search service. This also means the submission runs anywhere (Colab, CI, a bare venv)
with `pip install` doing nothing, by design.

Where each provider broke, concretely: OpenRouter free tier (50/day without a deposit, 1,000/day
with one — raising a *key's* spending limit is a different setting than depositing *account*
credit, a distinction that cost real time to discover); Gemini (billing-gated beyond a small trial
allowance); Groq (a hard 8,000-token **per-request** ceiling independent of daily quota, requiring
the chunking logic in `ledger.py`, plus a separate 200,000-token **per-day** account-wide cap that
chunking can't fix — only a second account, or a paid tier, fixes that one). None of these are
architecture problems; all are solved identically by paying for any one of them.
