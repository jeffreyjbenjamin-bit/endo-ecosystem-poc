"""Reusable AI service for Evidence Gap features, backed by Anthropic Claude.

Every call runs in the app's Python server process (Streamlit renders pages on the
server), so ANTHROPIC_API_KEY is read from the server environment and never sent
to the browser.

Features call it by name:

    from src.services import ai_service as ai

    if ai.provider_for("newsletter") == "anthropic":
        result = ai.generate("newsletter", system=SYSTEM, prompt=PROMPT)
        text = result.text

    for chunk in ai.stream_text("chat", system=SYSTEM, messages=history):
        ...

Configuration (environment variables; no code changes needed):

    ANTHROPIC_API_KEY             API key (required for Claude; secret)
    ANTHROPIC_MODEL               model for every feature   (default claude-sonnet-5-5)
    ANTHROPIC_MODEL_<FEATURE>     per-feature model, e.g. ANTHROPIC_MODEL_CHAT
    ANTHROPIC_EFFORT_<FEATURE>    low | medium | high | xhigh | max
    ANTHROPIC_MAX_RETRIES         SDK retries for 429 / 5xx / network errors (default 4)
    ANTHROPIC_TIMEOUT             seconds per request (default 600)
    ANTHROPIC_FALLBACKS           "default" (server-side refusal fallback) or "off"
    ANTHROPIC_PRICE_<MODEL>       "input,output[,cache_write,cache_read]" USD per 1M tokens,
                                  e.g. ANTHROPIC_PRICE_CLAUDE_SONNET_5_5=2,10,2.5,0.2
    NEWSLETTER_AI_PROVIDER        anthropic (default) | azure  - provider for the
                                  newsletter features (newsletter, chat)
    TRENDS_AI_PROVIDER            anthropic (default) | azure  - provider for the
                                  Home page AI trend analysis

Every call is logged (Python logging, logger "evidence_gap.ai") with input, output
and cache tokens and an estimated cost, and recorded in ai_usage.sqlite in the app
state directory (see usage_summary()).
"""

from __future__ import annotations

import logging
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import anthropic

from src.common.paths import state_path

log = logging.getLogger("evidence_gap.ai")
if not log.handlers:  # Streamlit doesn't configure app loggers; stdout reaches Azure
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[ai %(asctime)s] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

DEFAULT_MODEL = "claude-sonnet-5-5"
USAGE_DB = state_path("ai_usage.sqlite")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Per-feature defaults. Newsletter writing gets more effort; chat answers quickly.
FEATURES: Dict[str, Dict[str, Any]] = {
    "newsletter": {
        "label": "Newsletter generation",
        "effort": "high",
        "max_tokens": 32000,
    },
    "chat": {
        "label": "Ask AI (newsletter editor)",
        "effort": "medium",
        "max_tokens": 16000,
    },
    "trends": {
        "label": "AI trend analysis (Home)",
        "effort": "high",
        "max_tokens": 32000,
    },
}

# Which setting picks the provider for each feature.
_PROVIDER_ENV = {
    "newsletter": "NEWSLETTER_AI_PROVIDER",
    "chat": "NEWSLETTER_AI_PROVIDER",
    "trends": "TRENDS_AI_PROVIDER",
}
_DEFAULT_FEATURE = {"label": "AI", "effort": "medium", "max_tokens": 16000}

# Anthropic first-party prices, USD per 1M tokens:
# (input, output, cache write 5-minute, cache read). Override with ANTHROPIC_PRICE_<MODEL>.
PRICES: Dict[str, Tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-fable-5-1": (10.00, 50.00, 12.50, 0.25),
}

MODEL_NAMES = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-opus-5": "Claude Opus 5",
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
}


# ── Errors ────────────────────────────────────────────────────────────────────


class AIServiceError(Exception):
    """A Claude call failed. str(error) is a message that is safe to show users."""

    def __init__(self, message: str, kind: str = "error", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


def _translate(exc: Exception, model: str, retries: int) -> AIServiceError:
    """Map SDK exceptions (most specific first) to user-facing errors."""
    if isinstance(exc, AIServiceError):
        return exc
    if isinstance(exc, anthropic.AuthenticationError):
        return AIServiceError(
            "Claude rejected the API key. Check ANTHROPIC_API_KEY in the server settings.",
            "auth",
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return AIServiceError(
            f"This API key isn't allowed to use {model}. Check the key's workspace "
            "and model access in the Anthropic Console.",
            "permission",
        )
    if isinstance(exc, anthropic.NotFoundError):
        return AIServiceError(
            f"Model '{model}' was not found. Check the ANTHROPIC_MODEL settings.",
            "not_found",
        )
    if isinstance(exc, anthropic.RateLimitError):
        return AIServiceError(
            f"Claude is rate-limiting requests right now (retried {retries} times). "
            "Please try again in a minute.",
            "rate_limit",
            retryable=True,
        )
    if isinstance(exc, anthropic.BadRequestError):
        return AIServiceError(
            f"Claude couldn't process this request: {exc.message}", "bad_request"
        )
    if isinstance(exc, anthropic.InternalServerError):  # 5xx, including 529 overloaded
        return AIServiceError(
            "Claude is temporarily unavailable or overloaded. Please try again shortly.",
            "server",
            retryable=True,
        )
    if isinstance(exc, anthropic.APIStatusError):
        if (
            getattr(exc, "status_code", 0) == 402
            or getattr(exc, "type", "") == "billing_error"
        ):
            return AIServiceError(
                "The Anthropic account has a billing problem (credits or payment). "
                "Check Plans & Billing in the Anthropic Console.",
                "billing",
            )
        return AIServiceError(
            f"Claude returned an error ({exc.status_code}): {exc.message}", "api"
        )
    if isinstance(exc, anthropic.APITimeoutError):
        return AIServiceError(
            "The request to Claude timed out. Please try again.",
            "timeout",
            retryable=True,
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return AIServiceError(
            "Couldn't reach Claude (network error). Please try again.",
            "network",
            retryable=True,
        )
    return AIServiceError(f"Unexpected error calling Claude: {exc}", "unexpected")


# ── Configuration ─────────────────────────────────────────────────────────────


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def is_configured() -> bool:
    return bool(_env("ANTHROPIC_API_KEY"))


def model_for(feature: str) -> str:
    return (
        _env(f"ANTHROPIC_MODEL_{feature.upper()}")
        or _env("ANTHROPIC_MODEL")
        or DEFAULT_MODEL
    )


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def effort_for(feature: str) -> str:
    effort = _env(f"ANTHROPIC_EFFORT_{feature.upper()}").lower()
    if effort in ("low", "medium", "high", "xhigh", "max"):
        return effort
    return FEATURES.get(feature, _DEFAULT_FEATURE)["effort"]


def _provider_choice(feature: str) -> str:
    env = _PROVIDER_ENV.get(feature, f"{feature.upper()}_AI_PROVIDER")
    return _env(env, "anthropic").lower()


def provider_for(feature: str) -> str:
    """'anthropic' or 'azure' for a feature.

    NEWSLETTER_AI_PROVIDER / TRENDS_AI_PROVIDER = azure switch a feature back to
    Azure OpenAI; without an ANTHROPIC_API_KEY every feature stays on Azure OpenAI.
    """
    if _provider_choice(feature) == "azure" or not is_configured():
        return "azure"
    return "anthropic"


def provider_status(feature: str) -> str:
    """One-line description for the UI."""
    if provider_for(feature) == "anthropic":
        return model_name(model_for(feature))
    if not is_configured() and _provider_choice(feature) != "azure":
        return "Azure OpenAI GPT-5.1 (Claude not set up: ANTHROPIC_API_KEY missing)"
    return "Azure OpenAI GPT-5.1"


def prices_for(model: str) -> Tuple[float, float, float, float]:
    raw = _env("ANTHROPIC_PRICE_" + model.upper().replace("-", "_").replace(".", "_"))
    if raw:
        try:
            parts = [float(x) for x in raw.split(",")]
            inp, out = parts[0], parts[1]
            cw = parts[2] if len(parts) > 2 else inp * 1.25
            cr = parts[3] if len(parts) > 3 else inp * 0.1
            return (inp, out, cw, cr)
        except (ValueError, IndexError):
            log.warning("Ignoring invalid price override for %s: %r", model, raw)
    for known, price in PRICES.items():  # exact id, else a dated/prefixed variant
        if model == known or model.endswith(known):
            return price
    return PRICES[DEFAULT_MODEL]


def estimate_cost(model: str, usage: Dict[str, int]) -> float:
    inp, out, cw, cr = prices_for(model)
    return (
        usage.get("input_tokens", 0) * inp
        + usage.get("output_tokens", 0) * out
        + usage.get("cache_write_tokens", 0) * cw
        + usage.get("cache_read_tokens", 0) * cr
    ) / 1_000_000


def _fallbacks_enabled(model: str) -> bool:
    # Haiku has no server-side fallback; "off" disables it everywhere.
    return (
        _env("ANTHROPIC_FALLBACKS", "default").lower() != "off" and "haiku" not in model
    )


# ── Client ────────────────────────────────────────────────────────────────────

_client_lock = threading.Lock()
_client: Optional[anthropic.Anthropic] = None
_client_key: Optional[Tuple[str, int, float]] = None


def max_retries() -> int:
    try:
        return max(0, int(_env("ANTHROPIC_MAX_RETRIES", "4")))
    except ValueError:
        return 4


def _get_client() -> anthropic.Anthropic:
    """One shared client (thread-safe); rebuilt if the key or settings change."""
    global _client, _client_key
    key = _env("ANTHROPIC_API_KEY")
    if not key:
        raise AIServiceError(
            "Claude is not set up: ANTHROPIC_API_KEY is missing on the server.",
            "config",
        )
    try:
        timeout = float(_env("ANTHROPIC_TIMEOUT", "600"))
    except ValueError:
        timeout = 600.0
    cfg = (key, max_retries(), timeout)
    with _client_lock:
        if _client is None or _client_key != cfg:
            # The SDK retries 408/409/429/5xx and connection errors with
            # exponential backoff, honoring the API's retry-after header.
            _client = anthropic.Anthropic(
                api_key=key, max_retries=cfg[1], timeout=timeout
            )
            _client_key = cfg
        return _client


# ── Usage logging ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_usage (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT NOT NULL,
    feature            TEXT NOT NULL,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    served_model       TEXT,
    status             TEXT NOT NULL,
    stop_reason        TEXT,
    input_tokens       INTEGER DEFAULT 0,
    output_tokens      INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cache_read_tokens  INTEGER DEFAULT 0,
    cost_usd           REAL DEFAULT 0,
    latency_ms         INTEGER,
    request_id         TEXT,
    error              TEXT
);
"""


def _record(row: Dict[str, Any], db_path: Optional[str] = None) -> None:
    try:
        path = db_path or USAGE_DB
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.executescript(_SCHEMA)
            cols = list(row)
            conn.execute(
                f"INSERT INTO ai_usage ({', '.join(cols)}) VALUES "
                f"({', '.join('?' for _ in cols)})",
                [row[c] for c in cols],
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # logging must never break a feature
        log.warning("Could not record AI usage: %s", exc)


def _usage_dict(usage: Any) -> Dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_write_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    }


def usage_summary(
    days: int = 30, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Per feature and model over the last `days`: calls, tokens, estimated cost."""
    path = db_path or USAGE_DB
    if not Path(path).exists():
        return []
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            "SELECT feature, COALESCE(served_model, model), COUNT(*), "
            "SUM(status != 'ok'), SUM(input_tokens), SUM(output_tokens), "
            "SUM(cache_write_tokens), SUM(cache_read_tokens), SUM(cost_usd) "
            "FROM ai_usage WHERE ts >= ? GROUP BY 1, 2 ORDER BY 1, 2",
            (since,),
        ).fetchall()
    finally:
        conn.close()
    keys = [
        "feature",
        "model",
        "calls",
        "errors",
        "input_tokens",
        "output_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "cost_usd",
    ]
    return [dict(zip(keys, r)) for r in rows]


# ── Calls ─────────────────────────────────────────────────────────────────────


@dataclass
class AIResult:
    text: str
    model: str
    stop_reason: Optional[str]
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: int = 0
    request_id: Optional[str] = None

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


def _messages_from(prompt: Optional[str], messages: Optional[List[Dict[str, str]]]):
    if messages:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
    if prompt:
        return [{"role": "user", "content": prompt}]
    raise ValueError("generate()/stream_text() need a prompt or messages")


def _request(
    feature: str,
    system: str,
    messages,
    max_tokens: Optional[int],
    json_schema: Optional[Dict[str, Any]] = None,
):
    model = model_for(feature)
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens
        or FEATURES.get(feature, _DEFAULT_FEATURE)["max_tokens"],
        # The system prompt (instructions + document) is the stable, large part of
        # every request: cache it so follow-up chat turns read it at ~10% cost.
        "system": [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": messages,
        # Adaptive thinking is on by default on current models; effort sets depth.
        "output_config": {"effort": effort_for(feature)},
    }
    if json_schema:
        # Structured output: the answer is guaranteed to be JSON matching the schema.
        kwargs["output_config"]["format"] = {
            "type": "json_schema",
            "schema": json_schema,
        }
    return model, kwargs


def _open_stream(kwargs: Dict[str, Any]):
    client = _get_client()
    if _fallbacks_enabled(kwargs["model"]):
        # Server-side refusal fallback: a request declined by a safety classifier
        # is re-run on Anthropic's recommended fallback model within the same call.
        return client.beta.messages.stream(
            betas=[_FALLBACK_BETA], fallbacks="default", **kwargs
        )
    return client.messages.stream(**kwargs)


def _finish(feature, model, kwargs, final, t0, status="ok", error=None) -> AIResult:
    usage = _usage_dict(getattr(final, "usage", None))
    served = getattr(final, "model", None) or model
    cost = estimate_cost(served, usage)
    latency = int((time.time() - t0) * 1000)
    stop = getattr(final, "stop_reason", None)
    request_id = getattr(final, "_request_id", None)
    if stop == "refusal":
        status = "refusal"
    elif stop == "max_tokens":
        status = "truncated"
    log.info(
        "feature=%s model=%s served=%s status=%s stop=%s in=%d out=%d cache_write=%d "
        "cache_read=%d cost=$%.4f latency_ms=%d request_id=%s",
        feature,
        model,
        served,
        status,
        stop,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_write_tokens"],
        usage["cache_read_tokens"],
        cost,
        latency,
        request_id,
    )
    _record(
        {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "feature": feature,
            "provider": "anthropic",
            "model": model,
            "served_model": served,
            "status": status,
            "stop_reason": stop,
            **usage,
            "cost_usd": round(cost, 6),
            "latency_ms": latency,
            "request_id": request_id,
            "error": error,
        }
    )
    text = "".join(
        getattr(b, "text", "")
        for b in getattr(final, "content", []) or []
        if getattr(b, "type", "") == "text"
    )
    return AIResult(text, served, stop, usage, cost, latency, request_id)


def _record_failure(feature: str, model: str, t0: float, err: AIServiceError) -> None:
    log.warning(
        "feature=%s model=%s status=error kind=%s error=%s",
        feature,
        model,
        err.kind,
        err,
    )
    _record(
        {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "feature": feature,
            "provider": "anthropic",
            "model": model,
            "status": "error",
            "latency_ms": int((time.time() - t0) * 1000),
            "error": f"{err.kind}: {err}"[:500],
        }
    )


def _refusal_error(final) -> AIServiceError:
    details = getattr(final, "stop_details", None)
    category = getattr(details, "category", None) if details else None
    return AIServiceError(
        "Claude declined this request"
        + (f" (safety category: {category})" if category else "")
        + ". Try rephrasing it.",
        "refusal",
    )


def _backoff(attempt: int) -> None:
    time.sleep(min(2**attempt + random.uniform(0, 1), 20))


def generate(
    feature: str,
    system: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: Optional[int] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> AIResult:
    """Run one request to completion and return the text.

    With json_schema, the text is JSON that matches the schema (structured output).

    Streams under the hood so long outputs never hit HTTP timeouts. The SDK
    retries rate limits and server/network errors; a failure that happens after
    the stream has started is retried here as well. Raises AIServiceError.
    """
    msgs = _messages_from(prompt, messages)
    model, kwargs = _request(feature, system, msgs, max_tokens, json_schema)
    attempts = 1 + min(max_retries(), 2)
    for attempt in range(attempts):
        t0 = time.time()
        try:
            with _open_stream(kwargs) as stream:
                final = stream.get_final_message()
            result = _finish(feature, model, kwargs, final, t0)
            if result.stop_reason == "refusal":
                raise _refusal_error(final)
            return result
        except Exception as exc:
            err = _translate(exc, model, max_retries())
            if err.kind != "refusal":
                _record_failure(feature, model, t0, err)
            if err.retryable and attempt < attempts - 1:
                log.info(
                    "feature=%s retrying after %s (attempt %d)",
                    feature,
                    err.kind,
                    attempt + 2,
                )
                _backoff(attempt)
                continue
            raise err from exc
    raise AIServiceError("Claude request failed.", "unexpected")  # pragma: no cover


def stream_text(
    feature: str,
    system: str,
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Iterator[str]:
    """Yield the answer's text as it is written (for chat UIs).

    Failures before any text arrives are retried; after text has been shown the
    error is raised (AIServiceError). A refusal raises; an answer cut off by
    max_tokens ends with a short note.
    """
    msgs = _messages_from(prompt, messages)
    model, kwargs = _request(feature, system, msgs, max_tokens)
    attempts = 1 + min(max_retries(), 2)
    for attempt in range(attempts):
        t0 = time.time()
        started = False
        try:
            with _open_stream(kwargs) as stream:
                for text in stream.text_stream:
                    started = True
                    yield text
                final = stream.get_final_message()
            result = _finish(feature, model, kwargs, final, t0)
            if result.stop_reason == "refusal":
                raise _refusal_error(final)
            if result.truncated:
                yield "\n\n_(The answer was cut off at the length limit.)_"
            return
        except Exception as exc:
            err = _translate(exc, model, max_retries())
            if err.kind != "refusal":
                _record_failure(feature, model, t0, err)
            if err.retryable and not started and attempt < attempts - 1:
                _backoff(attempt)
                continue
            raise err from exc
