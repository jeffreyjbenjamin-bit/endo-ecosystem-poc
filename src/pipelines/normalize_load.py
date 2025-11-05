import glob
import json
import hashlib
import pandas as pd
from datetime import datetime
from src.connectors import (
    pubmed,
    ctgov,
    openalex,
    nih_reporter,
    web_search,
    semantic_scholar,
    crossref,
)
from src.common.storage import sqlite_conn, upsert_document
from src.common.normalize import finalize


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------


def make_uid(record: dict) -> str:
    """
    Create a stable SHA256 hash using available identifiers.
    """
    base = (
        (
            (record.get("doi") or "")
            + "|"
            + (record.get("url") or "")
            + "|"
            + (record.get("normalized_title") or record.get("title") or "")
        )
        .strip()
        .lower()
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def dedupe_records(records: list[dict]) -> list[dict]:
    """
    Remove duplicates based on the 'uid' key.
    """
    seen = set()
    deduped = []
    for r in records:
        uid = r.get("uid")
        if uid and uid not in seen:
            deduped.append(r)
            seen.add(uid)
    return deduped


def _iter_local_raw():
    """
    Yield (kind, path) tuples for all raw JSON files.
    """
    for path in glob.glob("./raw/pubmed/**/*.json", recursive=True):
        yield ("pubmed", path)
    for path in glob.glob("./raw/ctgov/**/*.json", recursive=True):
        yield ("ctgov", path)
    for path in glob.glob("./raw/openalex/**/*.json", recursive=True):
        yield ("openalex", path)
    for path in glob.glob("./raw/nih_reporter/**/*.json", recursive=True):
        yield ("nih_reporter", path)
    for path in glob.glob("./raw/ema/**/*.json", recursive=True):
        yield ("ema", path)
    for path in glob.glob("./raw/fda/**/*.json", recursive=True):
        yield ("fda", path)
    for path in glob.glob("./raw/biorxiv/**/*.json", recursive=True):
        yield ("biorxiv", path)
    for path in glob.glob("./raw/medrxiv/**/*.json", recursive=True):
        yield ("medrxiv", path)
    for path in glob.glob("./raw/web_search/**/*.json", recursive=True):
        yield ("web_search", path)
    for path in glob.glob("./raw/semantic_scholar/**/*.json", recursive=True):
        yield ("semantic_scholar", path)
    for path in glob.glob("./raw/crossref/**/*.json", recursive=True):
        yield ("crossref", path)


# --------------------------------------------------------
# Main
# --------------------------------------------------------


def main():
    conn = sqlite_conn()
    all_records = []
    counters = {}

    for kind, path in _iter_local_raw():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        try:
            if kind == "pubmed":
                docs = pubmed.to_docs(payload)
            elif kind == "ctgov":
                docs = ctgov.to_docs(payload)
            elif kind == "openalex":
                docs = openalex.to_docs(payload)
            elif kind == "nih_reporter":
                docs = nih_reporter.to_docs(payload)
            elif kind in {"ema", "fda", "biorxiv", "medrxiv"}:
                docs = payload.get("entries", [])
            elif kind == "web_search":
                docs = web_search.cse_to_docs(payload)
            elif kind == "semantic_scholar":
                docs = semantic_scholar.to_docs(payload)
            elif kind == "crossref":
                docs = crossref.to_docs(payload)
            else:
                docs = []
        except Exception as e:
            print(f"⚠️ Error parsing {path}: {e}")
            continue

        # Finalize, assign UID, and append
        for d in docs:
            doc = finalize(d)
            doc["source"] = kind
            doc["uid"] = make_uid(doc)
            all_records.append(doc)

        counters[kind] = counters.get(kind, 0) + len(docs)

    # Deduplication
    before = len(all_records)
    all_records = dedupe_records(all_records)
    after = len(all_records)
    print(f"✅ Deduped: {before} → {after} unique records")

    # Upsert to SQLite
    for rec in all_records:
        upsert_document(conn, rec)
    conn.commit()

    # Write to Parquet (for analytics + LLM ingestion)
    df = pd.DataFrame(all_records)
    parquet_path = "./data/documents.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"📦 Saved {len(df)} records to {parquet_path}")

    # Simple health log
    with open("./data/ingest.log", "a", encoding="utf-8") as log:
        ts = datetime.utcnow().isoformat()
        log.write(f"{ts}\tProcessed={after}\tSources={dict(counters)}\n")

    print("🎉 Normalization + dedupe complete.")


if __name__ == "__main__":
    main()
