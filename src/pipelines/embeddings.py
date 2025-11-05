"""
embeddings.py — generate vector embeddings for documents.parquet using Azure OpenAI

- Ensures ./data exists
- Robust batching with retry/backoff
- Skips already-embedded UIDs (idempotent)
- Stores float32 vectors in SQLite (BLOB) with a 'dim' column
- Optional FAISS index + uids sidecar (with Pylance-friendly add)
"""

import os
import json
import time
import sqlite3
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI

# Optional FAISS (fast local ANN)
try:
    import faiss

    USE_FAISS = True
except Exception:
    USE_FAISS = False

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://boldwave-openai-dev.openai.azure.com/"
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # must be set in env
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"
)
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))
TEXT_CHAR_LIMIT = int(os.getenv("EMBED_TEXT_CHAR_LIMIT", "8000"))

FAISS_INDEX_PATH = "./data/vector_store.faiss"
FAISS_UIDS_PATH = "./data/vector_store.uids.json"

# --------------------------------------------------------
# UTIL
# --------------------------------------------------------


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def chunked(iterable, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# --------------------------------------------------------
# CLIENT
# --------------------------------------------------------


def make_client() -> Tuple[AzureOpenAI, str]:
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set in the environment.")
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    deploy = AZURE_OPENAI_EMBED_DEPLOYMENT
    if not deploy:
        raise RuntimeError(
            "AZURE_OPENAI_EMBED_DEPLOYMENT is not set in the environment."
        )
    return client, deploy


# --------------------------------------------------------
# STORAGE (SQLite)
# --------------------------------------------------------


def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            uid    TEXT PRIMARY KEY,
            dim    INTEGER NOT NULL,
            vector BLOB    NOT NULL
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """
    )
    conn.commit()


def get_existing_uids(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    cur.execute("SELECT uid FROM vectors")
    return {row[0] for row in cur.fetchall()}


def upsert_vectors(
    conn: sqlite3.Connection, rows: List[Tuple[str, np.ndarray]]
) -> None:
    cur = conn.cursor()
    for uid, vec in rows:
        arr = np.asarray(vec, dtype=np.float32)
        cur.execute(
            "INSERT OR REPLACE INTO vectors (uid, dim, vector) VALUES (?, ?, ?)",
            (uid, int(arr.shape[0]), arr.tobytes()),
        )
    conn.commit()


# --------------------------------------------------------
# EMBEDDING UTILS
# --------------------------------------------------------


def build_text(row: pd.Series) -> str:
    parts = []
    for key in ("title", "abstract", "summary", "description"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            parts.append(row[key].strip())
    if not parts and "text" in row:
        return str(row["text"])[:TEXT_CHAR_LIMIT]
    return (" ".join(parts))[:TEXT_CHAR_LIMIT]


def embed_batch(client: Any, deployment: str, texts: List[str]) -> List[List[float]]:
    """
    Call Azure OpenAI Embeddings for a batch of texts.
    Explicitly returns a list on success or raises the last error after retries.
    """
    max_retries = 5
    delay = 1.0
    last_err: Exception | None = None

    for _ in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=deployment,  # Azure uses the *deployment name*
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2.0

    assert last_err is not None
    raise last_err


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------


def main():
    # Helpful path debug
    print("cwd =", os.path.abspath(os.getcwd()))
    print("PARQUET_PATH =", os.path.abspath(PARQUET_PATH))
    print("VECTOR_DB_PATH =", os.path.abspath(DB_PATH))

    # Ensure ./data exists for read/write paths
    ensure_parent_dir(PARQUET_PATH)
    ensure_parent_dir(DB_PATH)
    ensure_parent_dir(FAISS_INDEX_PATH)
    ensure_parent_dir(FAISS_UIDS_PATH)

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Parquet not found at {PARQUET_PATH}. Run normalize first to create it."
        )

    print(f"📘 Loading documents from {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet is missing required 'uid' column.")
    print(f"Loaded {len(df)} documents")

    # Prepare texts
    df["__text"] = df.apply(build_text, axis=1)
    uids = df["uid"].tolist()
    texts = df["__text"].tolist()

    client, deployment = make_client()
    print(f"🧠 Generating embeddings using Azure deployment '{deployment}'")

    # Open DB, ensure schema, compute worklist
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)
    existing = get_existing_uids(conn)
    todo_pairs = [
        (u, t) for u, t in zip(uids, texts) if u not in existing and t.strip()
    ]
    skipped = len(uids) - len(todo_pairs)
    if skipped:
        print(f"⏭️  Skipping {skipped} already-embedded or empty-text records")

    total = len(todo_pairs)
    if total == 0:
        print("✅ Nothing to embed — up to date.")
        conn.close()
        return

    all_rows: List[Tuple[str, np.ndarray]] = []

    # Process in batches
    batches = list(chunked(todo_pairs, BATCH_SIZE))
    for batch in tqdm(batches, total=len(batches)):
        batch_uids = [u for u, _ in batch]
        batch_texts = [t for _, t in batch]
        try:
            embs = embed_batch(client, deployment, batch_texts)
        except Exception as e:
            print(f"⚠️  Batch failed for first UID={batch_uids[0][:8]}: {e}")
            continue

        for uid, emb in zip(batch_uids, embs):
            all_rows.append((uid, np.asarray(emb, dtype=np.float32)))

        # Periodic flush to DB
        if len(all_rows) >= 1000:
            upsert_vectors(conn, all_rows)
            all_rows.clear()

    # Flush remaining
    if all_rows:
        upsert_vectors(conn, all_rows)
    conn.close()

    # Report dim and count
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(dim), MAX(dim) FROM vectors")
    count, mind, maxd = cur.fetchone()
    conn.close()

    dim_info = mind if mind == maxd else f"{mind}..{maxd}"
    print(f"✅ Saved/updated embeddings in {DB_PATH} (count={count}, dim={dim_info})")

    # ----------------------------------------------------
    # Optional FAISS index
    # ----------------------------------------------------
    if USE_FAISS:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT uid, dim, vector FROM vectors")
        rows = cur.fetchall()
        conn.close()

        if not rows:
            print("ℹ️  No vectors present; skipping FAISS.")
        else:
            dim = rows[0][1]
            index = faiss.IndexFlatL2(dim)
            matrix = np.empty((len(rows), dim), dtype=np.float32)
            ordered_uids = []
            for i, (uid, _dim, blob) in enumerate(rows):
                vec = np.frombuffer(blob, dtype=np.float32)
                matrix[i, :] = vec
                ordered_uids.append(uid)

            # Ensure float32 + contiguous for FAISS and silence Pylance warning
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            index.add(matrix)  # type: ignore[arg-type]

            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(FAISS_UIDS_PATH, "w", encoding="utf-8") as f:
                json.dump(ordered_uids, f)
            print(
                f"📦 Saved FAISS index to {FAISS_INDEX_PATH} and UID map to {FAISS_UIDS_PATH}"
            )

    print("🎉 Embedding generation complete.")


if __name__ == "__main__":
    main()
