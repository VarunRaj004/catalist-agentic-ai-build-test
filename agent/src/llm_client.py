"""Minimal LLM client, stdlib only (no pip packages). This is the one deliberate seam in the
whole project where swapping providers/models is a one-line env change, not a code change --
exactly so the agent isn't married to any single vendor. Supports:
  - "openrouter": OpenAI-style chat completions, routes to whatever model you point it at
  - "groq": Groq's OpenAI-compatible API (genuinely free tier, no prepaid billing)
  - "anthropic": calling Anthropic's Messages API directly
  - "gemini": calling Google's Generative Language API directly

Deliberate choice: the only third-party dependency this whole project would need is an HTTP
client, and `urllib` already does that. Keeping this to stdlib means the submission runs anywhere
(Colab, a bare venv, CI) with zero `pip install`.

Provider is chosen by LLM_PROVIDER env var ("openrouter" [default], "groq", "anthropic", "gemini").
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# $ per million tokens (input, output), used only as a fallback when the API response doesn't
# report cost directly (OpenRouter does, via `usage: {"include": true}`; Anthropic/Gemini/Groq
# don't, so this table drives the cost estimate on those paths). Update if pricing changes.
PRICING = {
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-5": (5.00, 25.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "openai/gpt-oss-120b": (0.0, 0.0),  # Groq free tier
    "openai/gpt-oss-20b": (0.0, 0.0),   # Groq free tier
}
FALLBACK_PRICING = (3.00, 15.00)  # used for unrecognized model slugs

def _load_dotenv():
    """Tiny .env loader (KEY=VALUE per line) so users don't have to export vars."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()  # must run BEFORE reading CATALIST_MODEL/LLM_PROVIDER below, or .env is ignored

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")
_PROVIDER_DEFAULT_MODEL = {
    "openrouter": "anthropic/claude-sonnet-4.5",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
}
# Override any time with CATALIST_MODEL -- the model is a config value, never hardcoded in logic.
DEFAULT_MODEL = os.environ.get(
    "CATALIST_MODEL", _PROVIDER_DEFAULT_MODEL.get(DEFAULT_PROVIDER, "anthropic/claude-sonnet-4.5")
)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str = None, api_key: str = None, provider: str = None, log_fn=None):
        self.provider = provider or DEFAULT_PROVIDER
        self.model = model or DEFAULT_MODEL
        if self.provider == "openrouter":
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            self._key_env_hint = "OPENROUTER_API_KEY=sk-or-..."
        elif self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            self._key_env_hint = "GEMINI_API_KEY=AIza..."
        elif self.provider == "groq":
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
            self._key_env_hint = "GROQ_API_KEY=gsk_..."
        else:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._key_env_hint = "ANTHROPIC_API_KEY=sk-ant-..."
        self.log_fn = log_fn  # optional callback(stage, model, input_tokens, output_tokens, usd_cost)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _fallback_cost(self, input_tokens, output_tokens):
        in_rate, out_rate = PRICING.get(self.model, FALLBACK_PRICING)
        return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate

    def _call_openrouter(self, system, user, max_tokens, temperature, retries):
        return self._call_openai_compatible(
            OPENROUTER_API_URL, system, user, max_tokens, temperature, retries,
            provider_label="OpenRouter", request_usage_cost=True,
        )

    def _call_groq(self, system, user, max_tokens, temperature, retries):
        return self._call_openai_compatible(
            GROQ_API_URL, system, user, max_tokens, temperature, retries,
            provider_label="Groq", request_usage_cost=False,
        )

    def _call_openai_compatible(self, url, system, user, max_tokens, temperature, retries,
                                  provider_label, request_usage_cost):
        def build_req(tokens):
            payload = {
                "model": self.model,
                "max_tokens": tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if request_usage_cost:
                # ask OpenRouter to report actual $ cost in usage, instead of a price-table guess
                payload["usage"] = {"include": True}
            return urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {self.api_key}",
                    # Python's default urllib User-Agent gets fingerprint-blocked by Cloudflare
                    # (seen as a 403 "error code: 1010" from Groq, which sits behind Cloudflare).
                    "user-agent": "Mozilla/5.0 (compatible; catalist-agent/1.0)",
                },
                method="POST",
            )

        last_err = None
        cur_max_tokens = max_tokens
        # Truncation retries are on their own small budget, independent of `retries` -- a caller
        # that passes retries=1 to fail fast on rate limits/quota should still get a free shot at
        # fixing a truncated response, since that costs no sleep and isn't the thing it's avoiding.
        length_retries_left = 2
        attempt = 0
        while True:
            req = build_req(cur_max_tokens)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice = (body.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = body.get("usage", {}) or {}
                it = usage.get("prompt_tokens", 0)
                ot = usage.get("completion_tokens", 0)
                cost = usage.get("cost")
                if cost is None:
                    cost = self._fallback_cost(it, ot)
                if choice.get("finish_reason") == "length" and length_retries_left > 0:
                    # hit the token budget before finishing -- either reasoning ate it all (text
                    # empty) or the JSON itself got cut off mid-string (text present but truncated,
                    # which fails json.loads downstream). Either way: double the budget and retry
                    # immediately (no sleep needed), don't just repeat the same request and get the
                    # same truncation again.
                    cur_max_tokens *= 2
                    length_retries_left -= 1
                    continue
                return text, it, ot, cost
            except urllib.error.HTTPError as e:
                last_err = e
                body_text = e.read().decode("utf-8", errors="replace")
                retryable = e.code == 429 or e.code >= 500 or e.code == 402
                if retryable and attempt < retries - 1:
                    wait = e.headers.get("Retry-After")
                    if not wait and e.code in (429, 402):
                        # Groq puts the wait hint in the message text, not a Retry-After header,
                        # e.g. "...Please try again in 52.0875s." Parse it rather than guess.
                        m = re.search(r"try again in (\d+(?:\.\d+)?)s", body_text)
                        if m:
                            wait = m.group(1)
                    if e.code in (429, 402) and wait:
                        sleep_s = min(float(wait) + 1, 150)
                    elif e.code == 402:
                        sleep_s = min(2 ** attempt * 5, 150)
                    else:
                        sleep_s = min(2 ** attempt, 20)
                    print(f"[llm_client] {provider_label} {e.code}, attempt {attempt+1}/{retries}, "
                          f"sleeping {sleep_s:.0f}s: {body_text[:150]}", flush=True)
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                print(f"[llm_client] {provider_label} {e.code}, giving up after {attempt+1} attempts: "
                      f"{body_text[:150]}", flush=True)
                raise LLMError(f"{provider_label} API error {e.code}: {body_text}") from e
            except urllib.error.URLError as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 20))
                    attempt += 1
                    continue
                raise LLMError(f"{provider_label} API call failed after {retries} retries: {last_err}") from e

    def _call_anthropic(self, system, user, max_tokens, temperature, retries):
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        last_err = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = "".join(b.get("text", "") for b in body.get("content", []))
                usage = body.get("usage", {})
                it, ot = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                cost = self._fallback_cost(it, ot)
                return text, it, ot, cost
            except urllib.error.HTTPError as e:
                last_err = e
                body_text = e.read().decode("utf-8", errors="replace")
                if e.code == 429 or e.code >= 500:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise LLMError(f"Anthropic API error {e.code}: {body_text}") from e
            except urllib.error.URLError as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))
        raise LLMError(f"Anthropic API call failed after {retries} retries: {last_err}")

    def _call_gemini(self, system, user, max_tokens, temperature, retries):
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent?{urllib.parse.urlencode({'key': self.api_key})}"

        def build_req(tokens):
            payload = json.dumps({
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": tokens, "temperature": temperature},
            }).encode("utf-8")
            return urllib.request.Request(
                url, data=payload, headers={"content-type": "application/json"}, method="POST"
            )

        last_err = None
        cur_max_tokens = max_tokens
        length_retries_left = 2  # same reasoning as the OpenRouter path: cheap, no sleep needed
        attempt = 0
        while True:
            req = build_req(cur_max_tokens)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                candidate = (body.get("candidates") or [{}])[0]
                parts = (candidate.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts)
                usage = body.get("usageMetadata", {}) or {}
                it = usage.get("promptTokenCount", 0)
                ot = usage.get("candidatesTokenCount", 0)
                cost = self._fallback_cost(it, ot)
                if candidate.get("finishReason") == "MAX_TOKENS" and length_retries_left > 0:
                    cur_max_tokens *= 2
                    length_retries_left -= 1
                    continue
                return text, it, ot, cost
            except urllib.error.HTTPError as e:
                last_err = e
                body_text = e.read().decode("utf-8", errors="replace")
                if (e.code == 429 or e.code >= 500) and attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 20))
                    attempt += 1
                    continue
                raise LLMError(f"Gemini API error {e.code}: {body_text}") from e
            except urllib.error.URLError as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 20))
                    attempt += 1
                    continue
                raise LLMError(f"Gemini API call failed after {retries} retries: {last_err}") from e

    def call(self, system: str, user: str, max_tokens: int = 1024, stage: str = "unnamed",
              temperature: float = 0.0, retries: int = 4) -> str:
        if not self.api_key:
            raise LLMError(
                f"No API key set for provider '{self.provider}'. Put {self._key_env_hint} in "
                "agent/.env before running the LLM-dependent stages."
            )
        if self.provider == "openrouter":
            text, it, ot, cost = self._call_openrouter(system, user, max_tokens, temperature, retries)
        elif self.provider == "groq":
            text, it, ot, cost = self._call_groq(system, user, max_tokens, temperature, retries)
        elif self.provider == "gemini":
            text, it, ot, cost = self._call_gemini(system, user, max_tokens, temperature, retries)
        else:
            text, it, ot, cost = self._call_anthropic(system, user, max_tokens, temperature, retries)
        self.total_input_tokens += it
        self.total_output_tokens += ot
        if self.log_fn:
            try:
                self.log_fn(stage, self.model, it, ot, cost)
            except Exception as e:
                # cost logging is a nice-to-have; a transient DB hiccup here must never take down
                # a real, already-successful LLM call and crash the whole batch because of it.
                print(f"[llm_client] cost logging failed (non-fatal): {e}")
        return text

    def call_json(self, system: str, user: str, max_tokens: int = 1024, stage: str = "unnamed",
                   retries: int = 4) -> dict:
        """Call the model and parse a JSON object out of the reply (tolerates code fences)."""
        text = self.call(system, user, max_tokens=max_tokens, stage=stage, retries=retries)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        start_arr = text.find("[")
        if start == -1 or (start_arr != -1 and start_arr < start):
            start = start_arr
        end = max(text.rfind("}"), text.rfind("]"))
        if start == -1 or end == -1:
            raise LLMError(f"Could not find JSON in model reply: {text[:300]}")
        snippet = text[start:end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as e:
            raise LLMError(f"Bad JSON from model ({e}): {snippet[:300]}") from e
        if not isinstance(parsed, dict):
            # Every prompt in this codebase asks for a JSON *object* ({"entries": [...]},
            # {"quote": "..."}, etc). A model that wraps it in a bare list (seen from Gemini on a
            # repair prompt) must be treated as a parse failure, not crash the caller's .get().
            raise LLMError(f"Expected a JSON object, got {type(parsed).__name__}: {snippet[:300]}")
        return parsed
