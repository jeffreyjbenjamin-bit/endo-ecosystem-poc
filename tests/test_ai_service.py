"""Shared AI service: config, cost, retries, errors, usage logging (no network)."""

import json
import sqlite3
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src.services import ai_service as ai


def _final(text="Hello", stop="end_turn", model="claude-sonnet-5-5", **usage):
    u = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    u.update(usage)
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(**u),
        model=model,
        stop_reason=stop,
        stop_details=(
            SimpleNamespace(category="general_harms") if stop == "refusal" else None
        ),
        _request_id="req_123",
    )


class FakeStream:
    def __init__(self, final, chunks=None, error=None):
        self._final, self._error = final, error
        self.text_stream = iter(
            chunks if chunks is not None else [final.content[-1].text]
        )

    def __enter__(self):
        if self._error:
            raise self._error
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._final


class FakeClient:
    """Records request kwargs; returns queued streams (or raises queued errors)."""

    def __init__(self, *streams):
        self.queue, self.calls = list(streams), []
        self.messages = SimpleNamespace(stream=self._stream("stable"))
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=self._stream("beta"))
        )

    def _stream(self, kind):
        def stream(**kwargs):
            self.calls.append((kind, kwargs))
            return self.queue.pop(0)

        return stream


def _http_error(cls, status):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=req), body=None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("ANTHROPIC_") or k == "NEWSLETTER_AI_PROVIDER":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(ai, "USAGE_DB", str(tmp_path / "ai_usage.sqlite"))
    monkeypatch.setattr(ai, "_backoff", lambda attempt: None)
    return monkeypatch


def _use(env, client):
    env.setattr(ai, "_get_client", lambda: client)


def test_config_defaults_and_overrides(env):
    assert ai.model_for("newsletter") == "claude-sonnet-5-5"
    env.setenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    env.setenv("ANTHROPIC_MODEL_CHAT", "claude-haiku-5-5")
    assert ai.model_for("newsletter") == "claude-opus-5-5"
    assert ai.model_for("chat") == "claude-haiku-5-5"
    assert ai.effort_for("newsletter") == "high" and ai.effort_for("chat") == "medium"
    env.setenv("ANTHROPIC_EFFORT_CHAT", "LOW")
    assert ai.effort_for("chat") == "low"
    assert ai.provider_for("chat") == "anthropic"
    env.setenv("NEWSLETTER_AI_PROVIDER", "azure")
    assert ai.provider_for("chat") == "azure"
    env.delenv("NEWSLETTER_AI_PROVIDER")
    env.delenv("ANTHROPIC_API_KEY")
    assert ai.provider_for("newsletter") == "azure"  # no key: stay on GPT
    assert "ANTHROPIC_API_KEY missing" in ai.provider_status("newsletter")


def test_cost_estimate_and_price_override(env):
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_write_tokens": 0,
        "cache_read_tokens": 1_000_000,
    }
    # Sonnet 5.5: $2 in + $1 out (100k at $10/M) + $0.20 cache reads
    assert ai.estimate_cost("claude-sonnet-5-5", usage) == pytest.approx(3.20)
    env.setenv("ANTHROPIC_PRICE_CLAUDE_SONNET_5_5", "3,15")
    assert ai.estimate_cost("claude-sonnet-5-5", usage) == pytest.approx(3 + 1.5 + 0.3)


def test_generate_logs_usage_and_uses_fallback_beta(env):
    client = FakeClient(FakeStream(_final(cache_read_input_tokens=2000)))
    _use(env, client)
    r = ai.generate("newsletter", system="SYS", prompt="Write it")
    assert r.text == "Hello" and r.request_id == "req_123"
    assert r.cost_usd == pytest.approx((1000 * 2 + 500 * 10 + 2000 * 0.2) / 1e6)
    kind, kw = client.calls[0]
    assert kind == "beta" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["model"] == "claude-sonnet-5-5" and kw["output_config"] == {
        "effort": "high"
    }
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"] == [{"role": "user", "content": "Write it"}]
    assert "temperature" not in kw and "thinking" not in kw
    row = (
        sqlite3.connect(ai.USAGE_DB)
        .execute(
            "SELECT feature, status, input_tokens, output_tokens, cache_read_tokens, request_id "
            "FROM ai_usage"
        )
        .fetchone()
    )
    assert row == ("newsletter", "ok", 1000, 500, 2000, "req_123")
    summary = ai.usage_summary()
    assert summary[0]["calls"] == 1 and summary[0]["cost_usd"] == pytest.approx(
        r.cost_usd
    )


def test_fallbacks_can_be_turned_off(env):
    env.setenv("ANTHROPIC_FALLBACKS", "off")
    client = FakeClient(FakeStream(_final()))
    _use(env, client)
    ai.generate("chat", system="S", prompt="q")
    assert client.calls[0][0] == "stable" and "fallbacks" not in client.calls[0][1]


def test_retries_retryable_errors_then_succeeds(env):
    client = FakeClient(
        FakeStream(_final(), error=_http_error(anthropic.InternalServerError, 529)),
        FakeStream(_final(text="Second try")),
    )
    _use(env, client)
    assert ai.generate("newsletter", system="S", prompt="p").text == "Second try"
    statuses = [
        r[0]
        for r in sqlite3.connect(ai.USAGE_DB).execute(
            "SELECT status FROM ai_usage ORDER BY id"
        )
    ]
    assert statuses == ["error", "ok"]


def test_rate_limit_gives_friendly_error_after_retries(env):
    errs = [
        FakeStream(_final(), error=_http_error(anthropic.RateLimitError, 429))
        for _ in range(3)
    ]
    _use(env, FakeClient(*errs))
    with pytest.raises(ai.AIServiceError) as e:
        ai.generate("newsletter", system="S", prompt="p")
    assert e.value.kind == "rate_limit" and "try again" in str(e.value)


@pytest.mark.parametrize(
    "cls,status,kind",
    [
        (anthropic.AuthenticationError, 401, "auth"),
        (anthropic.NotFoundError, 404, "not_found"),
        (anthropic.BadRequestError, 400, "bad_request"),
    ],
)
def test_non_retryable_errors_fail_fast(env, cls, status, kind):
    client = FakeClient(FakeStream(_final(), error=_http_error(cls, status)))
    _use(env, client)
    with pytest.raises(ai.AIServiceError) as e:
        ai.generate("chat", system="S", prompt="p")
    assert e.value.kind == kind and len(client.calls) == 1


def test_missing_key_is_a_clear_error(env):
    env.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ai.AIServiceError) as e:
        ai.generate("chat", system="S", prompt="p")
    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_refusal_and_truncation(env):
    _use(env, FakeClient(FakeStream(_final(text="", stop="refusal"))))
    with pytest.raises(ai.AIServiceError) as e:
        ai.generate("newsletter", system="S", prompt="p")
    assert e.value.kind == "refusal" and "general_harms" in str(e.value)

    _use(
        env, FakeClient(FakeStream(_final(text="abc", stop="max_tokens"), ["a", "bc"]))
    )
    out = list(
        ai.stream_text(
            "chat", system="S", messages=[{"role": "user", "content": "q", "edits": []}]
        )
    )
    assert out[:2] == ["a", "bc"] and "cut off" in out[-1]


def test_stream_retries_only_before_text_arrives(env):
    client = FakeClient(
        FakeStream(_final(), error=_http_error(anthropic.InternalServerError, 503)),
        FakeStream(_final(text="ok"), ["o", "k"]),
    )
    _use(env, client)
    assert "".join(ai.stream_text("chat", system="S", prompt="q")) == "ok"
    assert len(client.calls) == 2


def test_trends_provider_and_structured_output(env):
    assert ai.provider_for("trends") == "anthropic"
    env.setenv("TRENDS_AI_PROVIDER", "azure")
    assert ai.provider_for("trends") == "azure"
    assert ai.provider_for("newsletter") == "anthropic"  # independent settings
    env.delenv("TRENDS_AI_PROVIDER")
    client = FakeClient(FakeStream(_final(text='{"week": {}}')))
    _use(env, client)
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    ai.generate("trends", system="S", prompt="p", json_schema=schema)
    oc = client.calls[0][1]["output_config"]
    assert oc == {"effort": "high", "format": {"type": "json_schema", "schema": schema}}


# ── Tool use (search) ─────────────────────────────────────────────────────────


class EventStream(FakeStream):
    """A stream that yields events (text deltas and finished content blocks)."""

    def __init__(self, final, events=None, error=None):
        super().__init__(final, chunks=[], error=error)
        self._events = events or []

    def __iter__(self):
        return iter(self._events)


def _ev_text(t):
    return SimpleNamespace(type="text", text=t)


def _ev_block(block):
    return SimpleNamespace(type="content_block_stop", content_block=block)


def _tool_turn(tool_id="toolu_1", query="dienogest", stop="tool_use"):
    block = SimpleNamespace(
        type="tool_use", id=tool_id, name="search_library", input={"query": query}
    )
    final = _final(text="", stop=stop)
    final.content = [SimpleNamespace(type="thinking", thinking=""), block]
    return final, block


def test_tool_loop_runs_library_tool_and_web_search(env):
    final1, block = _tool_turn()
    ws = SimpleNamespace(
        type="server_tool_use", name="web_search", input={"query": "endometriosis 2026"}
    )
    wsr = SimpleNamespace(type="web_search_tool_result", content=[1, 2, 3])
    final2 = _final(text="Here are 2 articles", web_searches=0)
    final2.usage.server_tool_use = SimpleNamespace(web_search_requests=1)
    client = FakeClient(
        EventStream(final1, [_ev_block(block)]),
        EventStream(
            final2,
            [
                _ev_block(ws),
                _ev_block(wsr),
                _ev_text("Here are "),
                _ev_text("2 articles"),
            ],
        ),
    )
    _use(env, client)
    calls = []

    def handler(data):
        calls.append(data)
        return json.dumps({"results": [{"title": "A"}, {"title": "B"}]}), False

    out = list(
        ai.stream_with_tools(
            "chat",
            system="S",
            messages=[{"role": "user", "content": "find more"}],
            tools=[
                {
                    "name": "search_library",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
                ai.WEB_SEARCH_TOOL,
            ],
            handlers={"search_library": handler},
        )
    )
    text = "".join(o for o in out if isinstance(o, str))
    labels = [o.label for o in out if not isinstance(o, str)]
    assert text == "Here are 2 articles"
    assert calls == [{"query": "dienogest"}]
    assert labels == [
        "Searching the library: dienogest",
        "2 library results",
        "Searching the web: endometriosis 2026",
        "3 web results",
    ]
    # second request carries the full first turn plus the tool result
    msgs = client.calls[1][1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["content"] is final1.content
    assert msgs[-1]["content"][0]["tool_use_id"] == "toolu_1"
    # client tools stream their input; the server tool is left as declared
    tools = client.calls[0][1]["tools"]
    assert (
        tools[0]["eager_input_streaming"] is True
        and "eager_input_streaming" not in tools[1]
    )
    # web search counted in cost
    rows = (
        sqlite3.connect(ai.USAGE_DB)
        .execute("SELECT web_searches FROM ai_usage ORDER BY id")
        .fetchall()
    )
    assert rows[-1] == (1,)


def test_invalid_tool_input_is_reported_not_run(env):
    final1, block = _tool_turn()
    block.input = {"q": "wrong field"}
    client = FakeClient(
        EventStream(final1, [_ev_block(block)]),
        EventStream(_final(text="ok"), [_ev_text("ok")]),
    )
    _use(env, client)
    ran = []
    list(
        ai.stream_with_tools(
            "chat",
            "S",
            [{"role": "user", "content": "q"}],
            tools=[
                {
                    "name": "search_library",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
            handlers={"search_library": lambda d: ran.append(d) or ("x", False)},
        )
    )
    assert ran == []
    result = client.calls[1][1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "INVALID_INPUT" in result["content"]


def test_pause_turn_resumes_and_web_search_can_be_unavailable(env):
    paused = _final(text="", stop="pause_turn")
    client = FakeClient(
        EventStream(_final(), error=_http_error(anthropic.BadRequestError, 400)),
        EventStream(paused, []),
        EventStream(_final(text="done"), [_ev_text("done")]),
    )
    # the 400 message must mention web search to trigger the fallback
    client.queue[0]._error = anthropic.BadRequestError(
        "web_search tool is not enabled for this organization",
        response=httpx2.Response(400, request=httpx2.Request("POST", "https://x")),
        body=None,
    )
    _use(env, client)
    out = list(
        ai.stream_with_tools(
            "chat",
            "S",
            [{"role": "user", "content": "q"}],
            tools=[ai.WEB_SEARCH_TOOL],
            handlers={},
        )
    )
    notes = [o.label for o in out if not isinstance(o, str)]
    assert "Web search isn't enabled" in notes[0]
    assert "".join(o for o in out if isinstance(o, str)) == "done"
    assert all(not c[1]["tools"] for c in client.calls[1:])  # retried without it
    assert client.calls[2][1]["messages"][-1]["content"] is paused.content
