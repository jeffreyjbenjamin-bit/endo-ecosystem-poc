import os
import requests

BASE = os.getenv("NIH_REPORTER_BASE", "https://api.reporter.nih.gov/v2/projects/search")


def search(query="endometriosis", limit=100):
    """
    NIH RePORTER uses POST with a JSON body. We'll do a simple textSearch.
    limit: total results to request (API supports paging; PoC pulls one page up to 'limit').
    """
    body = {
        "criteria": {"textSearch": query},
        "offset": 0,
        "limit": limit,
        "sortField": "project_start_date",
        "sortOrder": "desc",
    }
    r = requests.post(BASE, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def to_docs(payload):
    """
    Map NIH RePORTER fields to our unified document schema.
    """
    docs = []
    for p in payload.get("results", []):
        pid = p.get("projectNumber")
        title = p.get("projectTitle")
        abs_ = p.get("abstractText")
        start = p.get("projectStartDate")  # e.g., '2024-07-01'
        country = p.get("orgCountry")
        pi = p.get("contactPiName")
        org = p.get("orgName")

        # Reporter doesn't have a canonical page per project id, but we can link to a search.
        url = (
            f"https://reporter.nih.gov/search?q={pid}"
            if pid
            else "https://reporter.nih.gov/"
        )

        authors = [{"name": pi}] if pi else None
        geos = [country] if country else None

        docs.append(
            {
                "source": "nih_reporter",
                "source_id": pid,
                "title": title,
                "abstract": abs_,
                "url": url,
                "published_date": start,  # using project start as a proxy
                "authors": authors,
                "journal_or_venue": org or "NIH RePORTER",
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
        )
    return docs
