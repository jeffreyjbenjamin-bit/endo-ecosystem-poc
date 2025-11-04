import requests

BASE = "https://api.openalex.org/works"


def search(query="endometriosis", per_page=50):
    r = requests.get(BASE, params={"search": query, "per_page": per_page}, timeout=60)
    r.raise_for_status()
    return r.json()


def to_docs(payload):
    docs = []
    for w in payload.get("results", []):
        # Try to pick a usable URL
        primary = (w.get("primary_location") or {}).get("source") or {}
        alt0 = (w.get("alternate_host_venues") or [{}])[0] or {}
        url = (
            primary.get("hosted_url")
            or alt0.get("url")
            or w.get("primary_location", {}).get("landing_page_url")
        )

        # Publication date (OpenAlex often has year only)
        pubyear = w.get("publication_year")
        pubdate = f"{pubyear}-01-01" if pubyear else None

        authors = [
            {"name": a.get("author", {}).get("display_name")}
            for a in (w.get("authorships") or [])
        ]

        docs.append(
            {
                "source": "openalex",
                "source_id": w.get("id"),
                "title": w.get("title"),
                "abstract": None,  # OpenAlex abstracts are inverted index; skip for PoC
                "url": url,
                "published_date": pubdate,
                "authors": authors or None,
                "journal_or_venue": (w.get("primary_location") or {})
                .get("source", {})
                .get("display_name"),
                "doi": w.get("doi"),
                "disease": ["endometriosis"],
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
