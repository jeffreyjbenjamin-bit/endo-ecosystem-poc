import glob
import json
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


def _iter_local_raw():
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


def main():
    conn = sqlite_conn()

    for kind, path in _iter_local_raw():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if kind == "pubmed":
            for d in pubmed.to_docs(payload):
                upsert_document(conn, finalize(d))
        elif kind == "ctgov":
            for d in ctgov.to_docs(payload):
                upsert_document(conn, finalize(d))
        elif kind == "openalex":
            for d in openalex.to_docs(payload):
                upsert_document(conn, finalize(d))
        elif kind == "nih_reporter":
            for d in nih_reporter.to_docs(payload):
                upsert_document(conn, finalize(d))
        elif kind in {"ema", "fda", "biorxiv", "medrxiv"}:
            for d in payload.get("entries", []):
                upsert_document(conn, finalize(d))
        elif kind == "web_search":
            for d in web_search.cse_to_docs(payload):
                upsert_document(conn, finalize(d))
        elif kind == "semantic_scholar":
            for d in semantic_scholar.to_docs(payload):
                upsert_document(conn, finalize(d))
        elif kind == "crossref":
            for d in crossref.to_docs(payload):
                upsert_document(conn, finalize(d))

    conn.commit()
    print("Normalization complete.")


if __name__ == "__main__":
    main()
