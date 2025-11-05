from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import time
import requests

# Docs: https://api.semanticscholar.org/api-docs/
BASE = os.getenv("S2_BASE", "https://api.semanticscholar.org/graph/v1")
SEARCH_URL = f"{BASE}/paper/search"

# Tunables (safe PoC defaults)
S2_LIMIT_DEFAULT = int(os.getenv("S2_LIMIT_DEFAULT", "12"))
S2_PAGE_SIZE = min(int(os.getenv("S2_PAGE_SIZE", "6")), 100)
S2_SLEEP_SEC = float(os.getenv("S2_SLEEP_SEC", "0.5"))
S2_MAX_RETRIES = int(os.getenv("S2_MAX_RETRIES", "8"))
S2_BACKOFF_BASE = float(os.getenv("S2_BACKOFF_BASE", "2.0"))
S2_MAX_WAIT_SEC = float(os.getenv("S2_MAX_WAIT_SEC", "60.0"))  # cap per-attempt wait
S2_FIELDS = os.getenv(
    "S2_FIELDS", "title,abstract,year,authors.name,externalIds,venue,url"
)


DEFAULT_HEADERS = {
    "User-Agent": "endo-ecosystem-poc/0.1 (+https://github.com/jeffreyjbenjamin-bit)"
}


def _parse_retry_after(resp: requests.Response, fallback: float) -> float:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return fallback
    try:
        # seconds per RFC
        wait = float(ra)
        return min(wait, S2_MAX_WAIT_SEC)
    except ValueError:
        # if it's a date string, just use the fallback
        return fallback


def _get_with_backoff(url: str, params: Dict[str, Any]) -> requests.Response:
    delay = S2_SLEEP_SEC
    last = None
    for attempt in range(1, S2_MAX_RETRIES + 1):
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=60)
        last = r
        if r.status_code < 400:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = _parse_retry_after(r, delay)
            time.sleep(wait)
            # exponential backoff, capped by S2_MAX_WAIT_SEC
            delay = min(delay * S2_BACKOFF_BASE, S2_MAX_WAIT_SEC)
            continue
        r.raise_for_status()
    # exhausted
    assert last is not None
    last.raise_for_status()
    return last


def search(query: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Query Semantic Scholar and return merged payload.
    Retries politely on 429/5xx with exponential backoff.
    """
    total_limit = limit or S2_LIMIT_DEFAULT
    page_size = min(max(1, S2_PAGE_SIZE), 100)
    fetched = 0
    offset = 0

    merged: Dict[str, Any] = {"total": 0, "data": []}

    while fetched < total_limit:
        take = min(page_size, total_limit - fetched)
        params = {
            "query": query,
            "limit": take,
            "offset": offset,
            "fields": S2_FIELDS,
        }
        # polite pacing between requests
        time.sleep(S2_SLEEP_SEC)
        r = _get_with_backoff(SEARCH_URL, params)
        payload = r.json() or {}
        data = payload.get("data", []) or []
        if offset == 0:
            merged["total"] = payload.get("total", 0)
        if not data:
            break
        merged["data"].extend(data)
        batch = len(data)
        fetched += batch
        offset += batch

    return merged


def to_docs(
    payload: Dict[str, Any], disease_tag: str = "endometriosis"
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in payload.get("data", []) or []:
        url = p.get("url")
        ext = p.get("externalIds") or {}
        doi = ext.get("DOI") or ext.get("doi")

        # authors.name (list of {name: ...})
        authors_raw = p.get("authors") or []
        authors = [
            {"name": a.get("name")}
            for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        ] or None

        # simple venue string
        venue = p.get("venue") or None

        docs.append(
            {
                "source": "semantic_scholar",
                "source_id": doi or url or p.get("paperId"),
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "url": url,
                "published_date": str(p.get("year")) if p.get("year") else None,
                "authors": authors,
                "journal_or_venue": venue,
                "doi": doi,
                "disease": [disease_tag],
                "topics": None,
                "trial_info": None,
                "geos": None,
                "mesh_terms": None,
                "license": None,
                "quality_score": 0.0,
                "hash_sha256": "",
                "raw_blob_uri": None,
                "lang": "en",
            }
        )
    return docs
