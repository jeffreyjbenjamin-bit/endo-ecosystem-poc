import os
import requests
from datetime import datetime

BASE = os.getenv("NIH_REPORTER_BASE", "https://api.reporter.nih.gov/v2/projects/search")


def _recent_fiscal_years(years_back: int) -> list[int]:
    now = datetime.now().year
    # e.g., years_back=3 => [2025, 2024, 2023] (assuming now=2025)
    return list(range(now, now - max(1, years_back), -1))


def search(
    query: str = "endometriosis", limit: int = 100, years_back: int | None = None
) -> dict:
    """
    Returns a single merged payload (like before) but internally pages through
    NIH RePORTER using a fiscal-year filter and per-page caps from .env.
    """
    years_back = years_back or int(os.getenv("NIH_REPORTER_YEARS_BACK", "3"))
    per_page = min(int(os.getenv("NIH_REPORTER_PER_PAGE", "200")), 500)
    max_pages = max(1, int(os.getenv("NIH_REPORTER_MAX_PAGES", "3")))

    fiscal_years = _recent_fiscal_years(years_back)

    merged = {
        "results": [],
        "criteria": {"textSearch": query, "fiscal_years": fiscal_years},
    }
    fetched = 0
    offset = 0
    pages = 0

    while pages < max_pages and fetched < limit:
        page_limit = min(per_page, limit - fetched)
        body = {
            "criteria": {"textSearch": query, "fiscal_years": fiscal_years},
            "offset": offset,
            "limit": page_limit,
            "sortField": "project_start_date",  # newest first
            "sortOrder": "desc",
        }
        r = requests.post(BASE, json=body, timeout=60)
        r.raise_for_status()
        payload = r.json()

        results = payload.get("results", []) or []
        if not results:
            break

        merged["results"].extend(results)
        batch = len(results)
        fetched += batch
        offset += batch
        pages += 1

        # Small courtesy delay; uncomment if you hit rate limits
        # time.sleep(0.25)

    return merged


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
