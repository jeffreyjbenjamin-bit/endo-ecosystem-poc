from dataclasses import dataclass
from hashlib import sha256
from datetime import datetime
import json


@dataclass
class Document:
    source: str
    source_id: str | None
    title: str | None
    abstract: str | None
    url: str | None
    published_date: str | None  # ISO8601 text
    authors: list | None
    journal_or_venue: str | None
    doi: str | None
    disease: list | None
    topics: list | None
    trial_info: dict | None
    geos: list | None
    mesh_terms: list | None
    license: str | None
    quality_score: float
    hash_sha256: str
    lang: str | None
    raw_blob_uri: str | None


def _quality_score(doc: "Document") -> float:
    score = 0.0
    if doc.source in {"pubmed", "openalex", "semantic_scholar", "nih_reporter"}:
        score += 0.4
    if doc.doi:
        score += 0.1
    if doc.trial_info:
        score += 0.2
    # Add recency if published in last 12 months
    try:
        if doc.published_date:
            year_str = str(doc.published_date)[:4]
            year = int(year_str)
            if (datetime.now().year - year) <= 1:
                score += 0.1
    except (ValueError, TypeError):
        # Ignore malformed or missing dates
        pass
    return max(0.0, min(1.0, score))


def finalize(doc: dict) -> dict:
    body = json.dumps(
        {k: v for k, v in doc.items() if k not in ("hash_sha256", "quality_score")},
        sort_keys=True,
        ensure_ascii=False,
    )
    doc["hash_sha256"] = sha256(body.encode()).hexdigest()
    # compute score using the dataclass (with the hash filled in)
    from_obj = Document(
        **{
            **doc,
            "quality_score": doc.get("quality_score", 0.0),
            "hash_sha256": doc["hash_sha256"],
        }
    )
    doc["quality_score"] = _quality_score(from_obj)
    return doc
