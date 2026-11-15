"""Enrich the evidence library with abstracts, citation counts and open-access links.

Phase 1 of the content-access plan (all free APIs):
  1. PubMed items: abstracts, MeSH terms and publication types from NCBI EFetch
     (the original ingestion used ESummary, which has no abstracts).
  2. Any item with a DOI: citation count, open-access status/link, and a missing
     abstract from OpenAlex (rebuilt from its inverted index).
  3. Re-embed every item whose text changed, so Find Similar and topic search
     match on the abstracts.

Run from the repo root:
    python -m src.pipelines.enrich_library            # enrich + re-embed
    python -m src.pipelines.enrich_library --no-embed # enrich only

Writes data/documents.parquet and updates data/vector_store.sqlite in place.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from src.common.citations import clean_doi
from src.connectors import pubmed

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
OPENALEX_WORKS = "https://api.openalex.org/works"


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and not v.strip()


def abstract_from_inverted_index(inv: Dict[str, List[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    slots: Dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def fetch_openalex_by_doi(
    dois: List[str], batch: int = 50
) -> Dict[str, Dict[str, Any]]:
    """{doi: {cited_by_count, is_oa, oa_url, abstract}} for DOIs (lowercase keys)."""
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(dois), batch):
        chunk = dois[i : i + batch]
        params = {
            "filter": "doi:" + "|".join(chunk),
            "per-page": batch,
            "select": "doi,cited_by_count,open_access,abstract_inverted_index",
        }
        r = requests.get(OPENALEX_WORKS, params=params, timeout=60)
        r.raise_for_status()
        for w in r.json().get("results", []):
            doi = clean_doi(w.get("doi")).lower()
            if not doi:
                continue
            oa = w.get("open_access") or {}
            out[doi] = {
                "cited_by_count": int(w.get("cited_by_count") or 0),
                "is_oa": bool(oa.get("is_oa")),
                "oa_url": oa.get("oa_url") or "",
                "abstract": abstract_from_inverted_index(
                    w.get("abstract_inverted_index")
                ),
            }
        time.sleep(0.15)  # polite pool
    return out


def enrich(df: pd.DataFrame, log=print) -> List[str]:
    """Fill abstracts and metadata in place. Returns uids whose text changed."""
    changed: List[str] = []
    for col, default in (
        ("cited_by_count", None),
        ("is_oa", None),
        ("oa_url", ""),
        ("publication_types", None),
        ("pmcid", ""),
    ):
        if col not in df.columns:
            df[col] = default
    df["cited_by_count"] = df["cited_by_count"].astype("object")
    df["is_oa"] = df["is_oa"].astype("object")

    # 1) PubMed abstracts
    pm = df[(df["source"] == "pubmed")]
    pmids = [str(x) for x in pm["source_id"].tolist() if str(x).strip()]
    log(f"PubMed: fetching full records for {len(pmids)} items…")
    recs = pubmed.fetch_abstracts(pmids) if pmids else {}
    n_abs = 0
    for idx, row in pm.iterrows():
        rec = recs.get(str(row["source_id"]))
        if not rec:
            continue
        if rec["abstract"] and _is_blank(row.get("abstract")):
            df.at[idx, "abstract"] = rec["abstract"]
            changed.append(row["uid"])
            n_abs += 1
        if rec["mesh_terms"]:
            df.at[idx, "mesh_terms"] = rec["mesh_terms"]
        if rec["publication_types"]:
            df.at[idx, "publication_types"] = rec["publication_types"]
        if rec["pmcid"]:
            df.at[idx, "pmcid"] = rec["pmcid"]
    log(f"PubMed: added {n_abs} abstracts.")

    # 2) OpenAlex: citation counts, open access, missing abstracts
    dois = {}
    for idx, row in df.iterrows():
        d = clean_doi(row.get("doi")).lower()
        if d:
            dois.setdefault(d, []).append(idx)
    log(f"OpenAlex: looking up {len(dois)} DOIs…")
    oa = fetch_openalex_by_doi(list(dois)) if dois else {}
    n_oa_abs = 0
    for d, info in oa.items():
        for idx in dois.get(d, []):
            df.at[idx, "cited_by_count"] = info["cited_by_count"]
            df.at[idx, "is_oa"] = info["is_oa"]
            df.at[idx, "oa_url"] = info["oa_url"]
            if info["abstract"] and _is_blank(df.at[idx, "abstract"]):
                df.at[idx, "abstract"] = info["abstract"]
                changed.append(df.at[idx, "uid"])
                n_oa_abs += 1
    log(f"OpenAlex: matched {len(oa)} DOIs, added {n_oa_abs} abstracts.")
    return list(dict.fromkeys(changed))


def reembed(df: pd.DataFrame, uids: List[str], log=print) -> int:
    """Re-embed the changed items into the vector store (same text recipe as before)."""
    import sqlite3

    from src.pipelines import embeddings as emb

    if not uids:
        return 0
    client = emb.make_embed_client()
    rows = df[df["uid"].isin(set(uids))]
    texts = [(r["uid"], emb.build_text(r)) for _, r in rows.iterrows()]
    conn = sqlite3.connect(VECTOR_DB_PATH)
    emb.ensure_tables(conn)
    done = 0
    for chunk in emb.chunked(texts, emb.BATCH_SIZE):
        # Same call (deployment, API version, retries) as the original build
        vecs = [
            np.asarray(v, dtype=np.float32)
            for v in emb.embed_batch(client, [t for _, t in chunk])
        ]
        emb.upsert_vectors(conn, [(u, v) for (u, _), v in zip(chunk, vecs)])
        conn.commit()
        done += len(chunk)
        log(f"  embedded {done}/{len(texts)}")
    conn.close()
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-embed", action="store_true", help="skip re-embedding")
    args = ap.parse_args()
    df = pd.read_parquet(PARQUET_PATH)
    before = int(df["abstract"].fillna("").astype(str).str.strip().ne("").sum())
    changed = enrich(df)
    after = int(df["abstract"].fillna("").astype(str).str.strip().ne("").sum())
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Abstracts: {before} -> {after} of {len(df)} items. Saved {PARQUET_PATH}.")
    if not args.no_embed:
        n = reembed(df, changed)
        print(f"Re-embedded {n} items into {VECTOR_DB_PATH}.")


if __name__ == "__main__":
    main()
