"""
search_provider.py — pluggable search backend abstraction.

Providers
---------
GoogleCSEProvider   — Google Custom Search Engine (current default)
BingGroundingProvider — Azure AI Foundry Grounding with Bing (ready to activate)

Factory
-------
get_provider(name)  — reads SEARCH_PROVIDER env var; defaults to google_cse.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Load .env relative to repo root so this module works regardless of CWD
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env", override=False)


# ── Error categories ──────────────────────────────────────────


class SearchError(Exception):
    """Base search error with a machine-readable category."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


# ── Diagnostic result ─────────────────────────────────────────


@dataclass
class ProviderDiagnostic:
    provider: str
    ok: bool
    active: bool  # credentials present
    key_hint: str  # last-4 of key, never full value
    cx_hint: str  # last-4 of CX (Google only)
    results_found: int
    error_category: Optional[str]  # None when ok
    error_detail: str


# ── Shared result schema ──────────────────────────────────────


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_name: str
    published_date: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ── Abstract base ─────────────────────────────────────────────


class SearchProvider(ABC):
    """All search backends implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def search(
        self, query: str, num: int = 10, **kwargs: Any
    ) -> List[SearchResult]: ...

    def available(self) -> bool:
        """Return True if the required credentials are present."""
        return True


# ── Google CSE ────────────────────────────────────────────────


class GoogleCSEProvider(SearchProvider):
    """
    Google Custom Search Engine.

    Required env vars:
        GOOGLE_CSE_KEY  — API key
        GOOGLE_CSE_CX   — Search engine ID
    """

    name = "google_cse"
    _ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
    ) -> None:
        self._key = api_key or os.getenv("GOOGLE_CSE_KEY", "")
        self._cx = cx or os.getenv("GOOGLE_CSE_CX", "")

    def available(self) -> bool:
        return bool(self._key and self._cx)

    def search(
        self,
        query: str,
        num: int = 10,
        date_restrict: Optional[str] = None,  # e.g. "d7", "m3"
        **kwargs: Any,
    ) -> List[SearchResult]:
        if not self.available():
            raise SearchError(
                "MISSING_CREDENTIALS",
                "GOOGLE_CSE_KEY and/or GOOGLE_CSE_CX are not set.",
            )
        params: Dict[str, Any] = {
            "key": self._key,
            "cx": self._cx,
            "q": query,
            "num": min(max(int(num), 1), 10),
        }
        if date_restrict:
            params["dateRestrict"] = date_restrict

        try:
            r = requests.get(self._ENDPOINT, params=params, timeout=30)
        except requests.exceptions.ConnectionError as exc:
            raise SearchError(
                "NETWORK_ERROR", f"Could not reach Google CSE: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise SearchError("NETWORK_ERROR", "Google CSE request timed out.") from exc

        self._raise_for_status(r)
        data = r.json()

        results: List[SearchResult] = []
        for item in data.get("items", []) or []:
            url = item.get("link", "")
            pub_date = _extract_date_from_metatags(item)
            results.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("snippet") or "").strip().replace("\n", " "),
                    source_name=_hostname(url),
                    published_date=pub_date,
                    raw=item,
                )
            )
        return results

    def _raise_for_status(self, r: requests.Response) -> None:
        """Translate HTTP errors into typed SearchError categories."""
        if r.status_code == 200:
            return
        try:
            body = r.json()
            msg = (body.get("error", {}) or {}).get("message", "")
        except Exception:
            msg = r.text[:200]

        if r.status_code == 400:
            if "cx" in msg.lower() or "engine" in msg.lower():
                raise SearchError(
                    "INVALID_CX", f"Search Engine ID (CX) rejected: {msg}"
                )
            raise SearchError("BAD_REQUEST", f"Google CSE returned 400: {msg}")
        if r.status_code == 403:
            if (
                "quota" in msg.lower()
                or "limit" in msg.lower()
                or "exceeded" in msg.lower()
            ):
                raise SearchError("QUOTA_EXCEEDED", f"Google CSE quota exceeded: {msg}")
            raise SearchError(
                "INVALID_KEY", f"Google CSE API key rejected (403): {msg}"
            )
        if r.status_code == 429:
            raise SearchError(
                "QUOTA_EXCEEDED", f"Google CSE rate limit hit (429): {msg}"
            )
        raise SearchError("HTTP_ERROR", f"Google CSE returned {r.status_code}: {msg}")

    def diagnose(self, test_query: str = "endometriosis") -> ProviderDiagnostic:
        """
        Run a single-result probe and return a structured diagnostic.
        Never surfaces actual key/CX values — only masked hints.
        """
        key_hint = _mask(self._key)
        cx_hint = _mask(self._cx)

        if not self._key:
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=False,
                key_hint="(not set)",
                cx_hint=cx_hint,
                results_found=0,
                error_category="MISSING_KEY",
                error_detail="GOOGLE_CSE_KEY is not set in environment.",
            )
        if not self._cx:
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=False,
                key_hint=key_hint,
                cx_hint="(not set)",
                results_found=0,
                error_category="MISSING_CX",
                error_detail="GOOGLE_CSE_CX is not set in environment.",
            )

        try:
            results = self.search(test_query, num=1)
        except SearchError as exc:
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=True,
                key_hint=key_hint,
                cx_hint=cx_hint,
                results_found=0,
                error_category=exc.category,
                error_detail=exc.message,
            )
        except Exception as exc:
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=True,
                key_hint=key_hint,
                cx_hint=cx_hint,
                results_found=0,
                error_category="UNKNOWN",
                error_detail=str(exc),
            )

        if not results:
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=True,
                key_hint=key_hint,
                cx_hint=cx_hint,
                results_found=0,
                error_category="NO_RESULTS",
                error_detail=(
                    "The probe query returned no results. "
                    "Check that the Custom Search Engine is configured to search the entire web."
                ),
            )

        return ProviderDiagnostic(
            provider=self.name,
            ok=True,
            active=True,
            key_hint=key_hint,
            cx_hint=cx_hint,
            results_found=len(results),
            error_category=None,
            error_detail="",
        )


# ── Bing Grounding (Azure AI Foundry) ─────────────────────────


class BingGroundingProvider(SearchProvider):
    """
    Azure AI Foundry Grounding with Bing Search.

    Required env vars:
        BING_SEARCH_KEY       — Azure Bing Search resource API key
        BING_SEARCH_ENDPOINT  — defaults to https://api.bing.microsoft.com/v7.0/search

    Supports the `freshness` kwarg: "Day" | "Week" | "Month"
    """

    name = "bing_grounding"
    _DEFAULT_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        self._key = api_key or os.getenv("BING_SEARCH_KEY", "")
        self._endpoint = endpoint or os.getenv(
            "BING_SEARCH_ENDPOINT", self._DEFAULT_ENDPOINT
        )

    def available(self) -> bool:
        return bool(self._key)

    def search(
        self,
        query: str,
        num: int = 10,
        freshness: Optional[str] = None,  # "Day" | "Week" | "Month"
        **kwargs: Any,
    ) -> List[SearchResult]:
        if not self.available():
            raise RuntimeError("Bing Search credentials missing. Set BING_SEARCH_KEY.")
        headers = {"Ocp-Apim-Subscription-Key": self._key}
        params: Dict[str, Any] = {
            "q": query,
            "count": min(int(num), 50),
            "mkt": "en-US",
            "responseFilter": "Webpages",
        }
        if freshness:
            params["freshness"] = freshness

        r = requests.get(self._endpoint, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        results: List[SearchResult] = []
        for item in (data.get("webPages") or {}).get("value", []) or []:
            url = item.get("url", "")
            pub_date = (item.get("dateLastCrawled") or "")[:10] or None
            results.append(
                SearchResult(
                    title=(item.get("name") or "").strip(),
                    url=url,
                    snippet=(item.get("snippet") or "").strip().replace("\n", " "),
                    source_name=_hostname(url),
                    published_date=pub_date,
                    raw=item,
                )
            )
        return results

    def diagnose(self, test_query: str = "endometriosis") -> ProviderDiagnostic:
        key_hint = _mask(self._key)
        if not self._key:
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=False,
                key_hint="(not set)",
                cx_hint="",
                results_found=0,
                error_category="MISSING_KEY",
                error_detail="BING_SEARCH_KEY is not set in environment.",
            )
        try:
            results = self.search(test_query, num=1)
        except Exception as exc:
            cat = exc.category if isinstance(exc, SearchError) else "UNKNOWN"
            return ProviderDiagnostic(
                provider=self.name,
                ok=False,
                active=True,
                key_hint=key_hint,
                cx_hint="",
                results_found=0,
                error_category=cat,
                error_detail=str(exc),
            )
        return ProviderDiagnostic(
            provider=self.name,
            ok=bool(results),
            active=True,
            key_hint=key_hint,
            cx_hint="",
            results_found=len(results),
            error_category=None if results else "NO_RESULTS",
            error_detail="" if results else "Probe returned no results.",
        )


# ── Factory ───────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {
    "google_cse": GoogleCSEProvider,
    "bing_grounding": BingGroundingProvider,
}


def get_provider(name: Optional[str] = None) -> SearchProvider:
    """
    Return an instantiated provider.
    Reads SEARCH_PROVIDER env var; falls back to google_cse.
    """
    key = (name or os.getenv("SEARCH_PROVIDER", "google_cse")).lower()
    cls = _REGISTRY.get(key, GoogleCSEProvider)
    return cls()


def available_providers() -> List[str]:
    """Return names of all providers that have credentials configured."""
    return [name for name, cls in _REGISTRY.items() if cls().available()]


def diagnose_provider(name: Optional[str] = None) -> ProviderDiagnostic:
    """Run a diagnostic probe on the active provider."""
    provider = get_provider(name)
    if hasattr(provider, "diagnose"):
        return provider.diagnose()
    return ProviderDiagnostic(
        provider=provider.name,
        ok=provider.available(),
        active=provider.available(),
        key_hint="",
        cx_hint="",
        results_found=0,
        error_category=None if provider.available() else "MISSING_CREDENTIALS",
        error_detail="",
    )


# ── Helpers ───────────────────────────────────────────────────


def _mask(s: str) -> str:
    """Show only the last 4 characters — never expose the full value."""
    if not s:
        return "(not set)"
    if len(s) <= 4:
        return "****"
    return "…" + s[-4:]


def _hostname(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        return re.sub(r"^www\.", "", h)
    except Exception:
        return ""


def _extract_date_from_metatags(item: Dict[str, Any]) -> Optional[str]:
    pagemap = item.get("pagemap") or {}
    metatags = (pagemap.get("metatags") or [{}])[0]
    for key in ("article:published_time", "og:updated_time", "datePublished", "date"):
        val = metatags.get(key)
        if val and len(str(val)) >= 10:
            return str(val)[:10]
    return None
