import os
import requests

BASE = os.getenv("CTGOV_BASE", "https://clinicaltrials.gov/api/v2/studies")


def search(condition="endometriosis", pagesize=200):
    """Return JSON payload of trials for the given condition."""
    r = requests.get(
        BASE, params={"query.term": condition, "pageSize": pagesize}, timeout=60
    )
    r.raise_for_status()
    return r.json()


def to_docs(payload):
    docs = []
    for s in payload.get("studies", []):
        sec = s.get("protocolSection", {})
        ident = sec.get("identificationModule", {})
        desc = sec.get("descriptionModule", {})
        stat = sec.get("statusModule", {})
        cond = sec.get("conditionsModule", {})
        design = sec.get("designModule", {})
        locs = sec.get("contactsLocationsModule", {}).get("locations", []) or []
        countries = sorted({loc.get("country") for loc in locs if loc.get("country")})
        nct_id = ident.get("nctId")

        docs.append(
            {
                "source": "ctgov",
                "source_id": nct_id,
                "title": ident.get("officialTitle") or ident.get("briefTitle"),
                "abstract": desc.get("briefSummary"),
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "published_date": (stat.get("lastUpdatePostDateStruct") or {}).get(
                    "date"
                ),
                "authors": None,
                "journal_or_venue": None,
                "doi": None,
                "disease": cond.get("conditions") or ["endometriosis"],
                "topics": None,
                "trial_info": {
                    "status": stat.get("overallStatus"),
                    "phase": stat.get("phase"),
                    "conditions": cond.get("conditions"),
                    "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
                },
                "geos": countries,
                "mesh_terms": None,
                "license": None,
                "quality_score": 0.0,
                "hash_sha256": "",
                "raw_blob_uri": None,
                "lang": "en",
            }
        )
    return docs
