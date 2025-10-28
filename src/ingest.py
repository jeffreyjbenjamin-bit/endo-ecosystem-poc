from __future__ import annotations

from typing import Any, Dict
import requests


def http_get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Basic GET helper that prefers JSON and falls back to text.
    Raises for non-2xx responses.
    """
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "EndoEcosystemPoC/0.1 (+github)"},
    )
    resp.raise_for_status()
    try:
        return resp.json()  # type: ignore[no-any-return]
    except ValueError:
        return {
            "_raw": resp.text,
            "_content_type": resp.headers.get("Content-Type", ""),
            "_status_code": resp.status_code,
        }


def ping(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Lightweight health check: returns ok + status_code only.
    Does NOT raise for non-2xx — just reports.
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "EndoEcosystemPoC/0.1 (+github)"},
        )
        return {"ok": resp.ok, "status_code": resp.status_code}
    except requests.RequestException as exc:  # network or timeout, etc.
        return {"ok": False, "error": str(exc)}
