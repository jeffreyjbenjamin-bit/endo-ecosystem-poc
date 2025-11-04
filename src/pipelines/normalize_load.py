import glob
import json
from src.connectors import pubmed, ctgov, openalex, nih_reporter
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
        elif kind in {"ema", "fda"}:
            for d in payload.get("entries", []):
                upsert_document(conn, finalize(d))

    conn.commit()
    print("Normalization complete.")


if __name__ == "__main__":
    main()
