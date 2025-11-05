from __future__ import annotations

import os
import time
from typing import List, Dict, Any

import requests

# Base E-utilities endpoint (override via .env if desired)
BASE = os.getenv("PUBMED_EUTILS_BASE", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
ESEARCH = f"{BASE}esearch.fcgi"
ESUMMARY = f"{BASE}esummary.fcgi"

# Optional: NCBI API key (increases rate limits if you have one)
NCBI_API_KEY = os.getenv("NCBI_API_KEY")


def _apply_key(params: Dict[str, Any]) -> Dict[str, Any]:
    """Attach NCBI API key (if present) to request parameters."""
    if NCBI_API_KEY:
        params = dict(params)
        params["api_key"] = NCBI_API_KEY
    return params


def search_ids(term: str, retmax: int = 200, sleep_s: float = 0.34) -> List[str]:
    """
    Search PubMed for PMIDs matching `term`. Paginates until retmax is reached.
    Uses GET (esearch) and returns a list of PMIDs as strings.
    """
    ids: List[str] = []
    retstart = 0
    # esearch returns at most 100k, but we'll just loop until retmax or no more ids
    while len(ids) < retmax:
        page_size = min(
            5000, retmax - len(ids)
        )  # esearch can handle large retmax; 5000 is plenty
        params = _apply_key(
            {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retmax": page_size,
                "retstart": retstart,
            }
        )
        r = requests.get(ESEARCH, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        page_ids = data.get("esearchresult", {}).get("idlist", []) or []
        if not page_ids:
            break
        ids.extend(page_ids)
        retstart += len(page_ids)
        # be courteous to NCBI
        time.sleep(sleep_s)
    return ids[:retmax]


def fetch_summaries(
    pmids: List[str], chunk_size: int = 150, sleep_s: float = 0.34
) -> Dict[str, Any]:
    """
    Fetch ESummary records for many PMIDs safely by batching and using POST to avoid 414 (URI too long).
    Returns a merged ESummary-style payload: {"result": {"uids": [...], "<uid>": {...}, ...}}
    """
    merged: Dict[str, Any] = {"result": {"uids": []}}
    if not pmids:
        return merged

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        if not chunk:
            continue
        # Use POST body to avoid long URLs
        data = _apply_key(
            {
                "db": "pubmed",
                "id": ",".join(chunk),
                "retmode": "json",
            }
        )
        r = requests.post(ESUMMARY, data=data, timeout=60)
        r.raise_for_status()
        part = r.json() or {}
        result = part.get("result", {}) or {}
        uids = [u for u in result.get("uids", []) if u]
        # extend master uid list
        merged["result"]["uids"].extend(uids)
        # merge per-uid dicts
        for uid in uids:
            merged["result"][uid] = result.get(uid)
        # be courteous to NCBI
        time.sleep(sleep_s)

    return merged


def to_docs(
    payload: Dict[str, Any], disease: str = "endometriosis"
) -> List[Dict[str, Any]]:
    """
    Convert PubMed ESummary payload to our normalized document schema.
    """
    docs: List[Dict[str, Any]] = []
    result = payload.get("result", {})
    # ESummary puts record dicts at result[uid] and a "uids" array listing them
    for key, rec in result.items():
        if not isinstance(rec, dict) or "uid" not in rec:
            continue
        uid = rec["uid"]
        eloc = rec.get("elocationid")
        doi = (
            eloc.replace("doi:", "").strip()
            if eloc and isinstance(eloc, str) and "doi:" in eloc
            else None
        )

        authors_list = rec.get("authors") or []
        if isinstance(authors_list, list):
            authors_norm = [
                {"name": a.get("name")}
                for a in authors_list
                if isinstance(a, dict) and a.get("name")
            ]
        else:
            authors_norm = None

        docs.append(
            {
                "source": "pubmed",
                "source_id": uid,
                "title": rec.get("title"),
                "abstract": None,  # ESummary doesn't return full abstracts
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "published_date": rec.get("pubdate"),
                "authors": authors_norm or None,
                "journal_or_venue": rec.get("fulljournalname"),
                "doi": doi,
                "disease": [disease],
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
