from __future__ import annotations

from typing import Iterator, Dict, Any, Optional, List
import feedparser
import requests
from time import struct_time

EMA_FEEDS: List[str] = [
    "https://www.ema.europa.eu/en/news.xml",
    "https://www.ema.europa.eu/en/new-human-medicine-new.xml",
]

FDA_FEEDS: List[str] = [
    "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml",
]

KEYWORDS: Optional[set[str]] = None  # keep disabled for PoC


def _date_from_struct(t: Optional[struct_time]) -> Optional[str]:
    if not t:
        return None
    try:
        return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    except (ValueError, TypeError):
        return None


def _matches_keywords(title: Optional[str], summary: Optional[str]) -> bool:
    if not KEYWORDS:
        return True
    text = f"{title or ''} {summary or ''}".lower()
    return any(k in text for k in KEYWORDS)


def _entries_from_feed(url: str) -> List[Dict[str, Any]]:
    # Fetch with requests (real UA), then parse bytes to avoid server blocks
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)  # bytes, not URL
    except Exception:
        return []
    entries = parsed.get("entries") or getattr(parsed, "entries", []) or []
    return [e for e in entries if isinstance(e, dict)]


def _normalize_entry(
    e: Dict[str, Any],
    source: str,
    venue: str,
    geos: List[str],
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
        "disease": ["endometriosis"],
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
    for url in EMA_FEEDS:
        for e in _entries_from_feed(url):
            if not _matches_keywords(
                e.get("title"), e.get("summary") or e.get("description")
            ):
                continue
            yield _normalize_entry(e, source="ema", venue="EMA", geos=["EU"])


def fda_items() -> Iterator[Dict[str, Any]]:
    for url in FDA_FEEDS:
        for e in _entries_from_feed(url):
            if not _matches_keywords(
                e.get("title"), e.get("summary") or e.get("description")
            ):
                continue
            yield _normalize_entry(e, source="fda", venue="FDA", geos=["US"])
