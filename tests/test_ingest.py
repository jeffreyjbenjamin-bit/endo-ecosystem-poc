from __future__ import annotations

from typing import Any, Dict
import ingest


class _DummyResp:
    def __init__(
        self, *, ok: bool, status_code: int, json_obj: Dict[str, Any] | None = None
    ):
        self.ok = ok
        self.status_code = status_code
        self._json_obj = json_obj or {}
        self.text = "fallback text"
        self.headers = {"Content-Type": "application/json"}

    def json(self) -> Dict[str, Any]:
        return self._json_obj

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")


def test_ping_reports_ok(monkeypatch):
    def _fake_get(*args, **kwargs):
        return _DummyResp(ok=True, status_code=200)

    monkeypatch.setattr("requests.get", _fake_get)
    result = ingest.ping("https://example.test")
    assert result["ok"] is True
    assert result["status_code"] == 200


def test_http_get_json_returns_json(monkeypatch):
    def _fake_get(*args, **kwargs):
        return _DummyResp(ok=True, status_code=200, json_obj={"hello": "world"})

    monkeypatch.setattr("requests.get", _fake_get)
    data = ingest.http_get_json("https://example.test/any")
    assert data == {"hello": "world"}


def test_http_get_json_raises_on_non_2xx(monkeypatch):
    def _fake_get(*args, **kwargs):
        return _DummyResp(ok=False, status_code=500)

    monkeypatch.setattr("requests.get", _fake_get)
    try:
        ingest.http_get_json("https://example.test/fail")
        assert False, "Expected an exception on non-2xx"
    except Exception as exc:
        assert "HTTP 500" in str(exc)
