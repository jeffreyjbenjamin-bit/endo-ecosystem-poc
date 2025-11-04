from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

# Always load .env when this module is imported
load_dotenv()


def _get_allow_hosts() -> set[str]:
    allow_hosts_env = (os.getenv("SEARCH_ALLOW_HOSTS") or "").strip()
    return {h.strip().lower() for h in allow_hosts_env.split(",") if h.strip()}


def _host_allowed(url: Optional[str]) -> bool:
    if not url:
        return False
    allow_hosts = _get_allow_hosts()
    if not allow_hosts:
        return True
    try:
        host = urlparse(url).hostname or ""
        return host.lower() in allow_hosts
    except ValueError:
        return False


# ---------- Google CSE path ----------
def search_google_cse(query: str, num: int = 10) -> Dict[str, Any]:
    key = os.getenv("GOOGLE_CSE_KEY")
    cx = os.getenv("GOOGLE_CSE_CX")
    endpoint = "https://www.googleapis.com/customsearch/v1"
    if not (key and cx):
        raise ValueError("GOOGLE_CSE_KEY/GOOGLE_CSE_CX must be set for Google CSE.")
    params = {"key": key, "cx": cx, "q": query, "num": min(max(num, 1), 10)}
    r = requests.get(endpoint, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def cse_to_docs(
    payload: Dict[str, Any], disease_tag: str = "endometriosis"
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in payload.get("items", []) or []:
        url = item.get("link")
        if not _host_allowed(url):
            continue
        docs.append(
            {
                "source": "web_search",
                "source_id": url,
                "title": item.get("title"),
                "abstract": item.get("snippet"),
                "url": url,
                "published_date": None,
                "authors": None,
                "journal_or_venue": urlparse(url).hostname if url else None,
                "doi": None,
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
