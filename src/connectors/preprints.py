from __future__ import annotations

from typing import Iterator, Dict, Any, Optional, List
import feedparser
import requests
from time import struct_time

# Site-wide recent feeds (we'll keyword-filter in code if desired)
BIORXIV_FEEDS: List[str] = ["https://connect.biorxiv.org/biorxiv_xml.php?subject=all"]
MEDRXIV_FEEDS: List[str] = ["https://connect.biorxiv.org/biorxiv_xml.php?subject=all"]

# PoC keyword filter: set to None/empty to disable filtering
KEYWORDS: Optional[set[str]] = None  # e.g., {"endometriosis", "adenomyosis"}


def _date_from_struct(t: Optional[struct_time]) -> Optional[str]:
    if not t:
        return None
    try:
        return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    except (ValueError, TypeError):
        return None


def _matches_keywords(title: Optional[str], summary: Optional[str]) -> bool:
    if not KEYWORDS:  # filtering disabled for PoC
        return True
    text = f"{title or ''} {summary or ''}".lower()
    return any(k in text for k in KEYWORDS)


def _entries_from_feed(url: str) -> List[Dict[str, Any]]:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
    except Exception:
        return []
    entries = parsed.get("entries") or getattr(parsed, "entries", []) or []
    return [e for e in entries if isinstance(e, dict)]


def _normalize_entry(
    e: Dict[str, Any],
    source: str,
    venue: str,
) -> Dict[str, Any]:
    title = e.get("title") or ""
    summary = e.get("summary") or e.get("description") or ""

    authors_raw = e.get("authors") or []
    authors_norm = None
    if isinstance(authors_raw, list):
        cleaned = [
            {"name": a.get("name")}
            for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        ]
        authors_norm = cleaned or None

    return {
        "source": source,  # "biorxiv" or "medrxiv"
        "source_id": e.get("id") or e.get("link"),
        "title": title or None,
        "abstract": summary or None,
        "url": e.get("link"),
        "published_date": _date_from_struct(
            e.get("published_parsed") or e.get("updated_parsed")
        ),
        "authors": authors_norm,
        "journal_or_venue": venue,
        "doi": None,  # many entries include DOI in the page; PoC skips scraping
        "disease": ["endometriosis"],  # PoC tag; refine later
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


def biorxiv_items() -> Iterator[Dict[str, Any]]:
    for url in BIORXIV_FEEDS:
        for e in _entries_from_feed(url):
            if not _matches_keywords(
                e.get("title"), e.get("summary") or e.get("description")
            ):
                continue
            yield _normalize_entry(e, source="biorxiv", venue="bioRxiv")


def medrxiv_items() -> Iterator[Dict[str, Any]]:
    for url in MEDRXIV_FEEDS:
        for e in _entries_from_feed(url):
            if not _matches_keywords(
                e.get("title"), e.get("summary") or e.get("description")
            ):
                continue
            yield _normalize_entry(e, source="medrxiv", venue="medRxiv")
