"""
retrieve.py — query your vector store and return top-k matches

- Uses FAISS L2 index if present (./data/vector_store.faiss + .uids.json)
- Falls back to brute-force cosine over SQLite vectors
- Embeds the query with the same Azure OpenAI deployment used in embeddings.py
- Prints a compact result table (rank, score/distance, title, uid, source)
- CLI: python src/pipelines/retrieve.py "biomarkers in endometriosis" 5
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Tuple

from openai import AzureOpenAI

# Optional FAISS
FAISS_PATH = "./data/vector_store.faiss"
FAISS_UIDS = "./data/vector_store.uids.json"
try:
    import faiss

    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")

# Azure OpenAI config (same as embeddings.py)
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://boldwave-openai-dev.openai.azure.com/"
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"
)
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

TEXT_CHAR_LIMIT = int(os.getenv("EMBED_TEXT_CHAR_LIMIT", "8000"))

# ---------- Azure client ----------


def make_client() -> Tuple[AzureOpenAI, str]:
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY not set.")
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    if not AZURE_OPENAI_EMBED_DEPLOYMENT:
        raise RuntimeError("AZURE_OPENAI_EMBED_DEPLOYMENT not set.")
    return client, AZURE_OPENAI_EMBED_DEPLOYMENT


def embed_query(text: str) -> np.ndarray:
    client, deployment = make_client()
    text = (text or "").strip()[:TEXT_CHAR_LIMIT]
    if not text:
        raise ValueError("Empty query text.")
    resp = client.embeddings.create(model=deployment, input=text)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return vec


# ---------- Data helpers ----------


def load_docs_df() -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid' column.")
    # A few convenient fallbacks
    if "title" not in df.columns:
        df["title"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    return df


def open_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector DB not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def load_all_vectors_from_db() -> Tuple[List[str], np.ndarray]:
    conn = open_db()
    cur = conn.cursor()
    cur.execute("SELECT uid, dim, vector FROM vectors")
    rows = cur.fetchall()
    conn.close()
    uids, mat = [], None
    if not rows:
        return uids, np.zeros((0, 0), dtype=np.float32)
    dim = rows[0][1]
    mat = np.empty((len(rows), dim), dtype=np.float32)
    for i, (uid, _dim, blob) in enumerate(rows):
        uids.append(uid)
        mat[i, :] = np.frombuffer(blob, dtype=np.float32)
    return uids, mat


# ---------- FAISS route (L2) ----------


def faiss_search(query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    if not (HAVE_FAISS and os.path.exists(FAISS_PATH) and os.path.exists(FAISS_UIDS)):
        return []
    index = faiss.read_index(FAISS_PATH)
    with open(FAISS_UIDS, "r", encoding="utf-8") as f:
        uids = json.load(f)
    # Reshape to (1, d)
    q = query_vec.reshape(1, -1).astype(np.float32)
    # L2 distances (lower is better)
    distances, idxs = index.search(q, top_k)
    out = []
    for d, i in zip(distances[0], idxs[0]):
        if i == -1:
            continue
        out.append((uids[i], float(d)))
    return out


# ---------- Brute-force cosine route ----------


def cosine_search(
    query_vec: np.ndarray, uids: List[str], mat: np.ndarray, top_k: int
) -> List[Tuple[str, float]]:
    if mat.size == 0:
        return []
    # Normalize for cosine
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = M.dot(q.astype(np.float32))
    # Top-k indices
    k = min(top_k, sims.shape[0])
    idxs = np.argpartition(-sims, k - 1)[:k]
    # Sort by similarity descending
    idxs = idxs[np.argsort(-sims[idxs])]
    return [(uids[i], float(sims[i])) for i in idxs]


# ---------- Presentation ----------


def print_results(matches: List[Tuple[str, float]], df: pd.DataFrame, use_faiss: bool):
    if not matches:
        print("No results.")
        return
    print("\n=== Top results ===")
    print(
        "(score = cosine similarity; distance = L2; higher cosine is better, lower L2 is better)"
    )
    for rank, (uid, score) in enumerate(matches, start=1):
        row = df.loc[df["uid"] == uid]
        title = row["title"].values[0] if not row.empty else ""
        source = row["source"].values[0] if not row.empty else ""
        if use_faiss:
            # FAISS route returns L2 distance
            print(
                f"{rank:>2}. distance={score:.4f} | {title[:100]}  [uid={uid[:8]}… | {source}]"
            )
        else:
            # cosine route returns similarity
            print(
                f"{rank:>2}. score={score:.4f} | {title[:100]}  [uid={uid[:8]}… | {source}]"
            )


# ---------- CLI ----------


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/pipelines/retrieve.py "your query text" [top_k]')
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # 1) Embed the query
    qvec = embed_query(query)

    # 2) Prefer FAISS if available
    use_faiss = HAVE_FAISS and os.path.exists(FAISS_PATH) and os.path.exists(FAISS_UIDS)
    if use_faiss:
        matches = faiss_search(qvec, top_k)
        df = load_docs_df()
        print_results(matches, df, use_faiss=True)
        return

    # 3) Fallback: brute-force cosine over SQLite
    uids, mat = load_all_vectors_from_db()
    matches = cosine_search(qvec, uids, mat, top_k)
    df = load_docs_df()
    print_results(matches, df, use_faiss=False)


if __name__ == "__main__":
    main()
