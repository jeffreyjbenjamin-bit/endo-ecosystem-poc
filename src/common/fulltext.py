"""Open-access full text from Europe PMC, fetched on demand and cached.

Used for the few articles that matter most (newsletter featured and supporting
articles, and the sources Ask AI checks claims against), never in bulk.

  lookup(article)    -> Europe PMC record: ids, open-access flag, licence, abstract
  full_text(article) -> sections of an open-access article (results, limitations…)
  excerpt(ft, n)     -> the most useful sections within n characters

Only articles Europe PMC marks open access are fetched in full; their licence is
kept with the text (it governs what may be quoted). Everything is cached in the
app state directory (fulltext/), including "not open access" answers.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests

from src.common.citations import clean_doi
from src.common.paths import state_path

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_TIMEOUT = 20
SECTION_ORDER = [
    "results",
    "limitations",
    "conclusions",
    "discussion",
    "methods",
    "introduction",
    "other",
]
_SECTION_RULES = [
    ("limitations", r"limitation|strengths? and weak"),
    ("conclusions", r"conclusion|summary"),
    ("results", r"result|finding|outcome"),
    ("discussion", r"discussion|interpretation|implication"),
    (
        "methods",
        r"method|material|patients|participants|design|statistic|setting|"
        r"data (collection|source)",
    ),
    ("introduction", r"introduction|background|rationale|aim"),
]


# ── Identifiers ───────────────────────────────────────────────────────────────


def article_ids(a: Dict[str, Any]) -> Dict[str, str]:
    """DOI / PMID / PMCID for an article dict (briefing, library, web, saved source)."""
    ids: Dict[str, str] = {}
    cite = a.get("citation") if isinstance(a.get("citation"), dict) else {}
    doi = clean_doi(a.get("doi") or (cite or {}).get("doi"))
    url = a.get("url") or ""
    host = (urlparse(url).hostname or "").lower()
    path = unquote(urlparse(url).path)
    if not doi and "doi.org" in host:
        doi = clean_doi(path.lstrip("/"))
    if not doi:
        m = re.search(r"(10\.\d{4,9}/[^\s?#]+)", path)
        doi = clean_doi(m.group(1).rstrip("/")) if m else ""
    if doi:
        ids["doi"] = doi
    if host == "pubmed.ncbi.nlm.nih.gov":
        m = re.match(r"/(\d+)", path)
        if m:
            ids["pmid"] = m.group(1)
    m = re.search(r"PMC(\d+)", url, re.I)
    if m:
        ids["pmcid"] = f"PMC{m.group(1)}"
    if a.get("pmcid"):
        ids["pmcid"] = str(a["pmcid"])
    if a.get("source") == "pubmed" and str(a.get("source_id") or "").isdigit():
        ids["pmid"] = str(a["source_id"])
    return ids


def _query(ids: Dict[str, str]) -> Optional[str]:
    if ids.get("pmcid"):
        return f"PMCID:{ids['pmcid']}"
    if ids.get("pmid"):
        return f"EXT_ID:{ids['pmid']} AND SRC:MED"
    if ids.get("doi"):
        return f'DOI:"{ids["doi"]}"'
    return None


# ── Cache ─────────────────────────────────────────────────────────────────────


def _cache_dir() -> Path:
    d = Path(state_path("fulltext"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_get(key: str) -> Any:
    p = _cache_dir() / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "__miss__"


def _cache_put(key: str, value: Any) -> None:
    p = _cache_dir() / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass


# ── Europe PMC ────────────────────────────────────────────────────────────────


def _clean_html(text: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _record(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pmid": x.get("pmid") or "",
        "pmcid": x.get("pmcid") or "",
        "doi": clean_doi(x.get("doi")),
        "is_oa": x.get("isOpenAccess") == "Y",
        "in_pmc": x.get("inPMC") == "Y" or x.get("inEPMC") == "Y",
        "license": (x.get("license") or "").upper(),
        "abstract": _clean_html(x.get("abstractText") or ""),
        "title": _clean_html(x.get("title") or ""),
    }


def _search(q: str, page_size: int = 1) -> Optional[List[Dict[str, Any]]]:
    """Europe PMC search results, or None on network trouble (not cached)."""
    try:
        r = requests.get(
            f"{EPMC}/search",
            params={
                "query": q,
                "resultType": "core",
                "format": "json",
                "pageSize": page_size,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return (r.json().get("resultList") or {}).get("result") or []
    except Exception:
        return None


def title_match(a: str, b: str) -> bool:
    """Same article title, allowing punctuation, case and small edits."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if na == nb or (len(na) > 40 and (na.startswith(nb) or nb.startswith(na))):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.92


def lookup(a: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Europe PMC record for an article, or None if it isn't indexed there.

    Uses the article's PMCID, PMID or DOI; when a link carries none of these
    (most publisher pages), searches by the article's exact title instead.
    """
    q = _query(article_ids(a))
    title = (a.get("title") or "").strip()
    if not q:
        # Title fallback: long enough to be specific, and verified on return
        words = _norm_title(title).split()
        if len(words) < 5:
            return None
        key = "lookup:title:" + _norm_title(title)
        cached = _cache_get(key)
        if cached != "__miss__":
            return cached
        quoted = title.replace('"', " ")
        results = _search(f'TITLE:"{quoted}"', page_size=3)
        if results is None:
            return None
        rec = next(
            (_record(x) for x in results if title_match(x.get("title") or "", title)),
            None,
        )
        _cache_put(key, rec)
        return rec
    cached = _cache_get("lookup:" + q)
    if cached != "__miss__":
        return cached
    results = _search(q)
    if results is None:
        return None
    rec = _record(results[0]) if results else None
    _cache_put("lookup:" + q, rec)
    return rec


def availability(a: Dict[str, Any]) -> str:
    """ "full_text" (open-access text we can read), "abstract", or "" (unknown)."""
    if a.get("origin") in ("clinical_trial", "upload") or "clinicaltrials.gov" in (
        a.get("url") or ""
    ):
        return ""
    rec = lookup(a)
    if not rec:
        return ""
    if rec.get("is_oa") and rec.get("pmcid"):
        return "full_text"
    return "abstract" if rec.get("abstract") else ""


def availability_for(
    articles: List[Dict[str, Any]], max_workers: int = 8
) -> Dict[str, str]:
    """{article key: availability} for many articles (parallel, cached)."""
    keyed = [a for a in articles if a.get("key")]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        found = list(pool.map(availability, keyed))
    return {a["key"]: f for a, f in zip(keyed, found)}


def _text(el: ET.Element) -> str:
    return " ".join(" ".join(el.itertext()).split())


def _classify(title: str) -> str:
    t = title.lower()
    for name, pattern in _SECTION_RULES:
        if re.search(pattern, t):
            return name
    return "other"


def parse_jats(xml_text: str) -> Dict[str, str]:
    """Sections of a JATS article: abstract, introduction, methods, results,
    discussion, limitations, conclusions, other."""
    root = ET.fromstring(xml_text)
    sections: Dict[str, List[str]] = {}
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        sections["abstract"] = [_text(abstract)]
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall("./sec"):
            title_el = sec.find("./title")
            name = _classify(_text(title_el) if title_el is not None else "")
            for sub in sec.iter("sec"):
                st = sub.find("./title")
                if st is not None and re.search(r"limitation", _text(st), re.I):
                    sections.setdefault("limitations", []).extend(
                        _text(p) for p in sub.findall("./p")
                    )
            for p in sec.iter("p"):
                txt = _text(p)
                if not txt:
                    continue
                sections.setdefault(name, []).append(txt)
                # Limitation paragraphs often sit inside the discussion
                if name in ("discussion", "conclusions") and re.search(
                    r"\blimitations?\b|\bwas limited by\b|\bshould be interpreted with "
                    r"caution\b",
                    txt,
                    re.I,
                ):
                    sections.setdefault("limitations", []).append(txt)
    return {k: "\n".join(dict.fromkeys(v))[:20000] for k, v in sections.items() if v}


def full_text(a: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Open-access full text for an article, or None (not OA / not found)."""
    rec = lookup(a)
    if not rec or not rec.get("is_oa") or not rec.get("pmcid"):
        return None
    key = "fulltext:" + rec["pmcid"]
    cached = _cache_get(key)
    if cached != "__miss__":
        return cached
    try:
        r = requests.get(f"{EPMC}/{rec['pmcid']}/fullTextXML", timeout=_TIMEOUT)
        if r.status_code == 404:
            _cache_put(key, None)
            return None
        r.raise_for_status()
        sections = parse_jats(r.text)
    except Exception:
        return None
    ft = {
        "pmcid": rec["pmcid"],
        "license": rec.get("license") or "",
        "source": "Europe PMC (open access)",
        "sections": sections,
    }
    if not sections.get("abstract") and rec.get("abstract"):
        ft["sections"]["abstract"] = rec["abstract"]
    _cache_put(key, ft)
    return ft


def excerpt(ft: Optional[Dict[str, Any]], budget: int = 6000) -> str:
    """The most useful parts of a full text within `budget` characters:
    results and limitations first, then conclusions, discussion, methods."""
    if not ft:
        return ""
    secs = ft.get("sections") or {}
    out: List[str] = []
    left = budget
    for name in SECTION_ORDER:
        text = secs.get(name)
        if not text or left < 200:
            continue
        share = min(len(text), max(600, left // 2) if name != "results" else left // 2)
        chunk = text[:share]
        if len(text) > share:
            chunk = chunk.rsplit(" ", 1)[0] + " …"
        out.append(f"[{name.capitalize()}] {chunk}")
        left -= len(chunk)
    return "\n".join(out)


def best_abstract(a: Dict[str, Any]) -> str:
    """An abstract for an article from Europe PMC (covers PubMed and preprints)."""
    rec = lookup(a)
    return (rec or {}).get("abstract") or ""


def enrich_articles(
    articles: List[Dict[str, Any]], budgets: List[int], max_workers: int = 6
) -> int:
    """Add full text (open access) or a better abstract to each article in place.

    budgets[i] is the excerpt size for articles[i]. Sets "full_text_excerpt",
    "full_text_license", "full_text_source" when open-access text is found,
    and replaces a short snippet with the Europe PMC abstract otherwise.
    Returns how many articles got full text.
    """

    def work(item):
        a, budget = item
        if a.get("origin") in ("clinical_trial", "upload"):
            return False
        ft = full_text(a)
        if ft:
            a["full_text_excerpt"] = excerpt(ft, budget)
            a["full_text_license"] = ft.get("license") or ""
            a["full_text_source"] = f"{ft['source']}, {ft['pmcid']}"
            if ft["sections"].get("abstract") and len(a.get("snippet") or "") < 400:
                a["snippet"] = ft["sections"]["abstract"][:1500]
            return True
        if len(a.get("snippet") or "") < 400:
            abstract = best_abstract(a)
            if len(abstract) > len(a.get("snippet") or ""):
                a["snippet"] = abstract[:1500]
        return False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        got = list(pool.map(work, zip(articles, budgets)))
    return sum(1 for g in got if g)
