from __future__ import annotations

from typing import Iterator, Dict, Any, Optional, List
import feedparser
from time import struct_time

# EMA “Press releases” and “Medicines highlights” Atom feeds export
EMA_FEEDS: List[str] = [
    "https://www.ema.europa.eu/en/news-events/press-releases?export=xml",
    "https://www.ema.europa.eu/en/news-events/medicines-highlights?export=xml",
]

# FDA Press Announcements RSS
FDA_FEEDS: List[str] = [
    "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-announcements/rss.xml",
]

# PoC: keyword filter (set to None/empty to disable)
KEYWORDS: Optional[set[str]] = None  # {"endometriosis", "adenomyosis"}


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
    # Ensure we always return a list, never None
    feed = feedparser.parse(url) or {}
    entries = feed.get("entries") or []
    # Some feedparser versions expose .entries attribute as well
    if not entries and hasattr(feed, "entries"):
        entries = getattr(feed, "entries") or []
    # Guarantee list[dict]
    return [e for e in entries if isinstance(e, dict)]


def _normalize_entry(
    e: Dict[str, Any],
    source: str,
    venue: str,
    geos: List[str],
) -> Dict[str, Any]:
    title = e.get("title") or ""
    summary = e.get("summary") or e.get("description") or ""

    # authors can be missing or None; make it a list
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
        "source": source,
        "source_id": e.get("id") or e.get("link"),
        "title": title or None,
        "abstract": summary or None,
        "url": e.get("link"),
        "published_date": _date_from_struct(
            e.get("published_parsed") or e.get("updated_parsed")
        ),
        "authors": authors_norm,
        "journal_or_venue": venue,
        "doi": None,
        "disease": ["endometriosis"],  # PoC tagging; refine later
        "topics": None,
        "trial_info": None,
        "geos": geos,
        "mesh_terms": None,
        "license": None,
        "quality_score": 0.0,
        "hash_sha256": "",
        "raw_blob_uri": None,
        "lang": "en",
    }


def ema_items() -> Iterator[Dict[str, Any]]:
    """Yield normalized EMA items (no optional iterables)."""
    for url in EMA_FEEDS:
        for e in _entries_from_feed(url):
            if not _matches_keywords(
                e.get("title"), e.get("summary") or e.get("description")
            ):
                continue
            yield _normalize_entry(e, source="ema", venue="EMA", geos=["EU"])


def fda_items() -> Iterator[Dict[str, Any]]:
    """Yield normalized FDA items (no optional iterables)."""
    for url in FDA_FEEDS:
        for e in _entries_from_feed(url):
            if not _matches_keywords(
                e.get("title"), e.get("summary") or e.get("description")
            ):
                continue
            yield _normalize_entry(e, source="fda", venue="FDA", geos=["US"])
