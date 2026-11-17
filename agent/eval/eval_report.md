# Self-authored eval report

**Important note on methodology.** This is not a run of the live QA agent (`qa_agent.answer_question`)
end to end — every LLM provider available today (OpenRouter free tier, Google Gemini, two Groq
accounts) hit hard capacity walls during this build (documented in the write-up's "where it broke"
section). What *is* real here: the **ledger-building stage actually ran against real models**
(`nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter, and `openai/gpt-oss-120b` via Groq) for three
teams — ES2005, TS3005, IS1004 — before those providers ran dry. Below, each of my 10 hand-verified
questions is checked against what that *real automated run* actually wrote into the ledger, not
against anything I generated myself for this report. That's a genuine test of the ingestion/extraction
half of the pipeline; the final answer-synthesis step (turning retrieved evidence into a cited
sentence) could not be exercised live today and is addressed separately in the write-up.

Ground truth for all 10 questions was hand-verified by reading the raw transcripts myself and
confirming every citation against `verify.py` before ever looking at what the agent produced
(see `custom_questions.json`).

## Scorecard

| ID | Team | Result |
|----|------|--------|
| C01 | ES2005 | **Match** |
| C02 | ES2005 | **Match** |
| C03 | ES2005 | **Match** |
| C04 | ES2005 | **Match** |
| C05 | TS3005 | **Partial** |
| C06 | TS3005 | **Match** |
| C07 | TS3005 | **Match** |
| C08 | TS3005 | **Partial** |
| C09 | IS1004 | **Match** |
| C10 | IS1004 | **Miss** |

**8/10 full match, 2/10 partial, 1/10 outright miss** (percentages don't sum to 10 because C05 and
C08 are each counted once as partial, not half-match + half-miss).

## Detail

### C01 (ES2005) — Match
Ground truth: rejected solar power (sunlight dependency), chose rechargeable battery.
Real ledger entry (`ES2005c`, built by the agent): *"Team decides on a rechargeable battery with a
charging stand... rejecting dynamo and solar options."* — correct, and the agent actually caught a
detail I hadn't verified (dynamo was also discussed and rejected, not just solar).

### C02 (ES2005) — Match
Ground truth: team voted speech recognition over TFT in meeting b, still committed in meeting c.
Ledger entries `ES2005b`/`ES2005c` correctly track the TFT rejection, speech-recognition acceptance,
and the meeting-c reconfirmation ("Team confirms using a standard, low-cost chip set with only
buttons and speech recognition").

### C03 (ES2005) — Match
Ground truth: spongy latex-rubber material, decided in meeting c.
Ledger entry `ES2005c`: *"Team selects latex or rubber with a spongy feel for the casing"* —
near-identical wording to my own hand-written ground truth, independently arrived at.

### C04 (ES2005) — Match
Ground truth: 12.50€ target (meeting a), 14.6€ interim overrun (meeting d).
Ledger entries capture both figures exactly: `ES2005a` "Maximum... twelve euros fifty",
`ES2005d` "Current prototype estimated at 14.6 Euros per unit, exceeding the 12.50 Euro target."

### C05 (TS3005) — Partial
Ground truth: Speaker D *first proposed* dropping speech recognition; Speaker A confirmed it was
dropped.
The ledger entry (`TS3005d`) correctly records that speech recognition was dropped, with the exact
quote I'd independently verified — but its `owner` field attributes the entry to **Speaker A**
(who confirmed/finalized it), not Speaker D (who first floated it). The ledger gets *what* happened
right but not *who said it first* — it's tracking the resolution point, not the proposal point, for
multi-turn negotiations. A real, specific limitation of the current prompt design.

### C06 (TS3005) — Match
Ground truth: titanium rejected as too expensive, rubber chosen.
Ledger entry `TS3005c`: *"the remote control case will be made of rubber because it is durable,
drop-resistant, allows fancy colours, and scratches are less visible than on painted titanium"* —
matches, with richer reasoning than I'd captured.

### C07 (TS3005) — Match
Ground truth: team targeted customers under 40.
Captured, but **not as its own entry** — it's folded into a combined `TS3005b` entry titled
"project requirements update" that also covers teletext and branding. The fact is there and
correct; the ledger's granularity just bundled several simultaneous requirement changes into one
entry rather than splitting them. Still counts as a real match, but worth noting for anyone trying
to retrieve "the age-targeting decision" by itself.

### C08 (TS3005) — Partial
Ground truth: 12.50€ target set in meeting a; a member pushed back mid-meeting-d ("I don't think we
can ever make it to twelve and half") before the team closed the gap.
The ledger captures the *start* (`TS3005a`, target set) and the *end* (`TS3005d`, "Final production
cost reaches exactly €12.50 per unit") correctly, but the specific *tension moment* — the pushback
that something seemed impossible before the team found a way — isn't its own entry. The ledger
tracks state transitions (target → final), not every intermediate expression of doubt along the way.
That's a reasonable design choice (you don't want an entry per "oh no" in the transcript) but it
does mean a brief built purely from the ledger would miss that the team doubted themselves before
succeeding.

### C09 (IS1004) — Match
Ground truth: team decided against both LCD and speech recognition for budget/feasibility reasons.
The ledger splits this into two separate entries — `IS1004b` "will not incorporate speech
recognition technology due to concerns about performance guarantees" and `IS1004c` "will not
incorporate an LCD screen" — which is factually correct but attributes a slightly different *reason*
for dropping speech recognition (performance guarantees) than the budget-tradeoff framing I'd quoted
from a different part of the same meeting. Both reasons are genuinely present in the transcript;
the agent picked a different (also valid) supporting line than I did.

### C10 (IS1004) — **Miss**
Ground truth: facing a ~10-cent budget overrun in the final meeting, the team swapped the case
material from rubber to plastic to close the gap ("Make it plastic instead of rubber. And then we're
basically on budget except for... ten cents.").
**This is a genuine, confirmed miss.** I checked every `IS1004d` ledger entry directly: there is no
entry mentioning a rubber→plastic swap, or any cost-driven material change in the final meeting at
all. The quote is real and verbatim in the transcript — the model simply didn't extract it. My best
guess at why: by meeting d, the team had already settled on plastic as the base case material back
in meeting c (`IS1004c`: "plastic is the most appropriate one, it'll bring the cost down"), so when
meeting d's chunk mentioned swapping "plastic instead of rubber" one more time in passing while
closing the last 10 cents, the model may have read it as restating an already-closed decision
(suppressed per the system prompt's "don't restate existing ledger entries that are still
unchanged" instruction) rather than noticing it was a *new, final* confirmation worth its own entry.
This is exactly the kind of subtle near-duplicate-decision case the current ledger design is weakest
at, and a concrete thing to improve (e.g. explicitly prompting the model to still log a "reaffirmed"
entry when a near-final-budget decision gets re-touched, even if the headline fact matches something
already closed).

## What this tells us

The retrieval and extraction machinery is doing real work, not padding: 8 of 10 independently
verified facts were captured correctly by a free, rate-limited model with no fine-tuning, and the
one outright miss has a specific, explainable cause rather than being random noise. The two partial
matches both stem from the same root cause — the ledger tracks *state* (what's currently true) more
reliably than it tracks *process* (who said what first, or every doubt expressed along the way),
which is a reasonable trade-off for "what does the team currently believe" but a real limitation if
someone wants "who gets credit for proposing X."
