from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import time
import requests

BASE = "https://api.crossref.org/works"
CR_LIMIT_DEFAULT = int(os.getenv("CR_LIMIT_DEFAULT", "30"))
CR_ROWS_PER_PAGE = min(int(os.getenv("CR_ROWS_PER_PAGE", "20")), 100)
CR_SLEEP_SEC = float(os.getenv("CR_SLEEP_SEC", "0.25"))

UA = {"User-Agent": "endo-ecosystem-poc/0.1 (mailto:you@example.com)"}


def search(query: str, limit: Optional[int] = None) -> Dict[str, Any]:
    total = limit or CR_LIMIT_DEFAULT
    rows = min(max(1, CR_ROWS_PER_PAGE), 100)
    fetched = 0
    offset = 0
    merged: Dict[str, Any] = {"message": {"items": []}}

    while fetched < total:
        take = min(rows, total - fetched)
        params = {
            "query": query,
            "rows": take,
            "offset": offset,
            "select": "DOI,title,URL,issued,author,container-title",
        }
        r = requests.get(BASE, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        payload = r.json() or {}
        items = payload.get("message", {}).get("items", []) or []
        if not items:
            break
        merged["message"]["items"].extend(items)
        batch = len(items)
        fetched += batch
        offset += batch
        time.sleep(CR_SLEEP_SEC)
    return merged


def to_docs(
    payload: Dict[str, Any], disease_tag: str = "endometriosis"
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for it in payload.get("message", {}).get("items", []) or []:
        doi = it.get("DOI")
        url = it.get("URL")
        title_list = it.get("title") or []
        title = title_list[0] if title_list else None
        issued = it.get("issued", {}).get("date-parts", [])
        year = str(issued[0][0]) if issued and issued[0] else None
        cont = it.get("container-title") or []
        venue = cont[0] if cont else None
        authors = None
        a = it.get("author") or []
        if isinstance(a, list):
            authors = [
                {
                    "name": " ".join(
                        [x for x in [au.get("given"), au.get("family")] if x]
                    )
                }
                for au in a
                if isinstance(au, dict)
            ]
            if not authors:
                authors = None
        docs.append(
            {
                "source": "crossref",
                "source_id": doi or url,
                "title": title,
                "abstract": None,
                "url": url,
                "published_date": year,
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
