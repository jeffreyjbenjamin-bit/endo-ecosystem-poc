"""Source citations for articles: who published it, where, when, and the DOI.

Evidence library items carry authors, journal/venue and DOI from ingestion
(citation_from_library). Web results only carry a hostname, so the article page
is read for the citation metadata most publishers embed (Highwire/Google Scholar
"citation_*" tags, Dublin Core, Open Graph); see fetch_citation(). Lookups are
cached in the app state directory, including misses, so each URL is read once.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from src.common.paths import state_path

CACHE_FILE = "page_citations.json"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EvidenceGap/1.0; citation lookup)",
    "Accept": "text/html,application/xhtml+xml",
}
_MAX_BYTES = 600_000
_META_RE = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR_RE = re.compile(r'([a-zA-Z:_.-]+)\s*=\s*("([^"]*)"|\'([^\']*)\')')

# Where an evidence-library item was indexed (not where it was published).
INDEX_NAMES = {
    "pubmed": "PubMed",
    "ctgov": "ClinicalTrials.gov",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "openalex": "OpenAlex",
    "crossref": "Crossref",
    "semantic_scholar": "Semantic Scholar",
    "fda": "FDA",
    "ema": "EMA",
    "nih_reporter": "NIH RePORTER",
    "web_search": "Web",
}


def host(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", urlparse(url).hostname or "")
    except ValueError:
        return ""


def authors_list(raw: Any) -> List[str]:
    """Names from the library's author formats (list of {name} / strings / text)."""
    names: List[str] = []
    if raw is None:
        return names
    items = (
        list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]
    )
    for item in items:
        if isinstance(item, dict):
            item = item.get("name") or item.get("full_name") or ""
        text = str(item or "").strip()
        if not text:
            continue
        # One string holding many authors: "Smith J; Lee K" or "Smith, J., Lee, K."
        if re.search(r"[A-Z]\.,\s", text):
            parts = [p.strip() for p in re.split(r"(?<=\.),\s+", text) if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r";|\s+and\s+", text) if p.strip()]
        for p in parts:
            if p not in names:
                names.append(p)
    return names


def clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return doi if doi.startswith("10.") else ""


def citation_from_library(row: Dict[str, Any]) -> Dict[str, Any]:
    """Citation fields for an evidence-library row."""
    source = str(row.get("source") or "")
    venue = str(row.get("journal_or_venue") or "").strip()
    if not venue and source == "ctgov":
        venue = "ClinicalTrials.gov"
    return {
        "authors": authors_list(row.get("authors")),
        "venue": venue,
        "publisher": "",
        "date": str(row.get("published_date") or "")[:10],
        "doi": clean_doi(row.get("doi")),
        "via": INDEX_NAMES.get(source, source),
        "site": host(str(row.get("url") or "")),
    }


def citation_from_html(html: str, url: str) -> Dict[str, Any]:
    """Citation fields from a page's <meta> tags."""
    meta: Dict[str, List[str]] = {}
    for tag in _META_RE.findall(html):
        attrs = {
            m.group(1).lower(): (m.group(3) if m.group(3) is not None else m.group(4))
            for m in _ATTR_RE.finditer(tag)
        }
        name = (attrs.get("name") or attrs.get("property") or "").lower()
        content = unescape((attrs.get("content") or "").strip())
        if name and content:
            meta.setdefault(name, []).append(content)

    def first(*keys: str) -> str:
        for k in keys:
            if meta.get(k):
                return meta[k][0]
        return ""

    authors: List[str] = []
    for k in ("citation_author", "dc.creator", "dc.contributor", "author"):
        for v in meta.get(k, []):
            for a in authors_list(v):
                if a not in authors:
                    authors.append(a)
        if authors:
            break
    if not authors and meta.get("citation_authors"):
        authors = authors_list(meta["citation_authors"][0])
    doi = clean_doi(first("citation_doi", "dc.identifier", "prism.doi", "doi"))
    return {
        "authors": authors[:12],
        "venue": first(
            "citation_journal_title",
            "citation_conference_title",
            "prism.publicationname",
            "citation_publisher",
            "og:site_name",
            "application-name",
        ),
        "publisher": first("citation_publisher", "dc.publisher"),
        "date": first(
            "citation_publication_date",
            "citation_date",
            "citation_online_date",
            "dc.date",
            "article:published_time",
        )[:10].replace("/", "-"),
        "doi": doi,
        "via": "",
        "site": host(url),
    }


_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
# Publishers that block automated page reads, or whose site name is not on the page.
KNOWN_SITES = {
    "thelancet.com": "The Lancet (Elsevier)",
    "sciencedirect.com": "ScienceDirect (Elsevier)",
    "cell.com": "Cell Press",
    "nature.com": "Nature Portfolio",
    "nejm.org": "The New England Journal of Medicine",
    "jamanetwork.com": "JAMA Network",
    "bmj.com": "The BMJ",
    "academic.oup.com": "Oxford Academic",
    "onlinelibrary.wiley.com": "Wiley Online Library",
    "link.springer.com": "Springer",
    "tandfonline.com": "Taylor & Francis Online",
    "frontiersin.org": "Frontiers",
    "mdpi.com": "MDPI",
    "journals.plos.org": "PLOS",
    "medrxiv.org": "medRxiv (preprint)",
    "biorxiv.org": "bioRxiv (preprint)",
    "fertstert.org": "Fertility and Sterility",
    "pubs.rsna.org": "Radiological Society of North America (RSNA)",
    "ajronline.org": "American Journal of Roentgenology",
    "acog.org": "American College of Obstetricians and Gynecologists",
    "asrm.org": "American Society for Reproductive Medicine",
    "eshre.eu": "European Society of Human Reproduction and Embryology",
    "contemporaryobgyn.net": "Contemporary OB/GYN",
    "medscape.com": "Medscape",
    "healio.com": "Healio",
    "who.int": "World Health Organization",
    "fda.gov": "U.S. Food and Drug Administration",
    "ema.europa.eu": "European Medicines Agency",
    "nih.gov": "U.S. National Institutes of Health",
    "clinicaltrials.gov": "ClinicalTrials.gov",
    "businesswire.com": "Business Wire (company press release)",
    "prnewswire.com": "PR Newswire (company press release)",
    "globenewswire.com": "GlobeNewswire (company press release)",
    "fiercebiotech.com": "Fierce Biotech",
    "biospace.com": "BioSpace",
    "statnews.com": "STAT",
}


def site_name(url_or_host: str) -> str:
    h = host(url_or_host) if "/" in url_or_host else url_or_host
    for known, name in KNOWN_SITES.items():
        if h == known or h.endswith("." + known):
            return name
    return h


def _ncbi_ids(url: str) -> Optional[tuple]:
    """("pubmed", "41957705") or ("pmc", "13087501") for NCBI article URLs."""
    h = host(url)
    path = urlparse(url).path
    if h == "pubmed.ncbi.nlm.nih.gov":
        m = re.match(r"/(\d+)", path)
        return ("pubmed", m.group(1)) if m else None
    if h in ("pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"):
        m = re.search(r"PMC(\d+)", path, re.I)
        return ("pmc", m.group(1)) if m else None
    return None


def fetch_ncbi_citations(urls: List[str], timeout: float = 10.0) -> Dict[str, Any]:
    """Citations for PubMed / PMC URLs from NCBI's summary service (batched)."""
    out: Dict[str, Any] = {}
    by_db: Dict[str, Dict[str, str]] = {}
    for u in urls:
        ids = _ncbi_ids(u)
        if ids:
            by_db.setdefault(ids[0], {})[ids[1]] = u
    for db, id_map in by_db.items():
        try:
            r = requests.get(
                _EUTILS,
                params={
                    "db": db,
                    "id": ",".join(id_map),
                    "retmode": "json",
                    "tool": "evidence_gap",
                },
                timeout=timeout,
            )
            r.raise_for_status()
            result = r.json().get("result", {})
        except Exception:
            continue
        for uid, url in id_map.items():
            rec = result.get(uid) or {}
            if not rec or rec.get("error"):
                out[url] = None
                continue
            doi = ""
            for aid in rec.get("articleids") or []:
                if aid.get("idtype") == "doi":
                    doi = clean_doi(aid.get("value"))
            if not doi:
                m = re.search(r"doi:\s*(10\.\S+)", rec.get("elocationid") or "")
                doi = clean_doi(m.group(1)) if m else ""
            out[url] = {
                "authors": [
                    a.get("name", "") for a in rec.get("authors") or [] if a.get("name")
                ][:12],
                "venue": rec.get("fulljournalname") or rec.get("source") or "",
                "publisher": "",
                "date": (rec.get("sortpubdate") or rec.get("pubdate") or "")[
                    :10
                ].replace("/", "-"),
                "doi": doi,
                "via": "PubMed" if db == "pubmed" else "PubMed Central",
                "site": host(url),
            }
    return out


def fetch_citation(url: str, timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    try:
        with requests.get(
            url, headers=_HEADERS, timeout=timeout, stream=True, allow_redirects=True
        ) as r:
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                return None
            chunks, size = [], 0
            for chunk in r.iter_content(16_384):
                chunks.append(chunk)
                size += len(chunk)
                if size >= _MAX_BYTES or b"</head>" in chunk.lower():
                    break
            html = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
            c = citation_from_html(html, r.url)
            return c if (c["venue"] or c["authors"] or c["doi"]) else None
    except Exception:
        return None


def _load_cache() -> Dict[str, Any]:
    try:
        return json.loads(Path(state_path(CACHE_FILE)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def citations_for(urls: Iterable[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """{url: citation or None} for web article URLs (cached, read in parallel)."""
    urls = [u for u in dict.fromkeys(urls) if u and u.startswith("http")]
    cache = _load_cache()
    missing = [u for u in urls if u not in cache]
    if missing:
        ncbi = [u for u in missing if _ncbi_ids(u)]
        cache.update(fetch_ncbi_citations(ncbi))
        pages = [u for u in missing if u not in cache]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for u, c in zip(pages, pool.map(fetch_citation, pages)):
                cache[u] = c
        try:
            path = Path(state_path(CACHE_FILE))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
    return {u: cache.get(u) for u in urls}


def _short_name(name: str) -> str:
    """ "Tiniakou, Ioanna" / "HAGIHARA, M." -> "Tiniakou I" / "HAGIHARA M"."""
    if name.count(",") == 1:
        last, first = (p.strip() for p in name.split(","))
        initials = "".join(w[0] for w in re.split(r"[\s.\-]+", first) if w)
        return f"{last} {initials}".strip()
    return name.strip()


def _authors_text(authors: List[str]) -> str:
    names = [_short_name(a) for a in authors if a]
    if not names:
        return ""
    if len(names) > 3:
        return ", ".join(names[:3]) + ", et al"
    return ", ".join(names)


_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def title_case(venue: str) -> str:
    """NCBI lists journals in sentence case ("Journal of minimally invasive
    gynecology"); title-case those, leave already-cased names alone."""
    if not venue:
        return venue
    words = venue.split(" ")
    content = [w for w in words[1:] if w.isalpha() and w.lower() not in _SMALL_WORDS]
    # Already title case (most content words capitalized): leave it alone
    if not content or sum(w[0].isupper() for w in content) * 2 > len(content):
        return venue
    return " ".join(
        w if (i and w.lower() in _SMALL_WORDS) else (w[:1].upper() + w[1:])
        for i, w in enumerate(words)
    )


def format_citation(c: Optional[Dict[str, Any]], fallback_site: str = "") -> str:
    """One-line citation in plain text, e.g.

    "Hong D, Xiao Y. Frontiers in Pharmacology (2025). doi:10.3389/fphar.2025.1656217.
     Indexed in PubMed."
    """
    if not c:
        if not fallback_site:
            return ""
        name = site_name(fallback_site)
        return f"Published by {name}" + (
            f" ({fallback_site})" if name != fallback_site else ""
        )
    parts = []
    a = _authors_text(c.get("authors") or [])
    if a:
        parts.append(a.rstrip(".") + ".")
    venue = title_case(c.get("venue") or "")
    pub = c.get("publisher") or ""
    where = venue
    if pub and pub.lower() not in venue.lower():
        where = f"{venue} ({pub})" if venue else pub
    if not where:
        where = c.get("site") or fallback_site
    year = (c.get("date") or "")[:4]
    if where:
        parts.append(f"{where}" + (f", {year}." if year else "."))
    elif year:
        parts.append(f"{year}.")
    if c.get("doi"):
        parts.append(f"doi:{c['doi']}.")
    if c.get("via") and c["via"].lower() not in where.lower():
        parts.append(f"Indexed in {c['via']}.")
    return " ".join(parts)
