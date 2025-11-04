import os
import requests

BASE = os.getenv("PUBMED_EUTILS_BASE", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")


def search_ids(term: str, retmax=200):
    q = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}
    r = requests.get(BASE + "esearch.fcgi", params=q, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_summaries(ids: list[str]):
    if not ids:
        return {}
    q = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    r = requests.get(BASE + "esummary.fcgi", params=q, timeout=60)
    r.raise_for_status()
    return r.json()


def to_docs(payload, disease="endometriosis"):
    docs = []
    result = payload.get("result", {})
    for _, rec in result.items():
        if not isinstance(rec, dict) or "uid" not in rec:
            continue
        uid = rec["uid"]
        eloc = rec.get("elocationid")
        doi = eloc.replace("doi:", "").strip() if eloc and "doi:" in eloc else None
        docs.append(
            {
                "source": "pubmed",
                "source_id": uid,
                "title": rec.get("title"),
                "abstract": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "published_date": rec.get("pubdate"),
                "authors": [{"name": a.get("name")} for a in rec.get("authors", [])],
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
