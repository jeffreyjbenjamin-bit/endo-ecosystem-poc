"""
embeddings.py — generate vector embeddings for documents.parquet using Azure OpenAI

Fully updated for the NEW Azure OpenAI / Azure AI Inference API.
"""

# ---- Imports (must be first for Ruff E402 compliance) ----
import os
import time
import sqlite3
from typing import List, Tuple

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Optional FAISS
try:

    USE_FAISS = True
except Exception:
    USE_FAISS = False


load_dotenv()


# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # MUST end with '/'
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"
)

# Embeddings API version (Azure required)
EMBED_API_VERSION = "2023-05-15"

BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))
TEXT_CHAR_LIMIT = int(os.getenv("EMBED_TEXT_CHAR_LIMIT", "8000"))

FAISS_INDEX_PATH = "./data/vector_store.faiss"
FAISS_UIDS_PATH = "./data/vector_store.uids.json"


# --------------------------------------------------------
# PATH HELPERS
# --------------------------------------------------------


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def chunked(iterable, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# --------------------------------------------------------
# CLIENT (CORRECT FOR AZURE)
# --------------------------------------------------------


def make_embed_client() -> OpenAI:
    """
    Uses the new Azure AI Inference API format:
        <endpoint>/openai/deployments/<deployment_name>/
    """
    if not AZURE_OPENAI_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")

    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")

    if not AZURE_OPENAI_ENDPOINT.endswith("/"):
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must end with '/'")

    base = (
        f"{AZURE_OPENAI_ENDPOINT}"
        f"openai/deployments/{AZURE_OPENAI_EMBED_DEPLOYMENT}/"
    )

    print("Embed base_url =", base)

    return OpenAI(
        api_key=AZURE_OPENAI_KEY,
        base_url=base,
    )


# --------------------------------------------------------
# STORAGE
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
    return (" ".join(parts))[:TEXT_CHAR_LIMIT] if parts else ""


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Calls Azure embedding endpoint.
    """
    max_retries = 5
    delay = 1.0
    last = None

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                input=texts,
                model=AZURE_OPENAI_EMBED_DEPLOYMENT,
                extra_query={"api-version": EMBED_API_VERSION},
            )
            return [item.embedding for item in resp.data]

        except Exception as e:
            last = e
            print(f"Retry {attempt+1}/5 after error: {e}")
            time.sleep(delay)
            delay *= 2

    raise last


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------


def main():
    print("=== Embedding Generator ===")
    print("Using endpoint:", AZURE_OPENAI_ENDPOINT)
    print("Using deployment:", AZURE_OPENAI_EMBED_DEPLOYMENT)

    ensure_parent_dir(DB_PATH)
    ensure_parent_dir(FAISS_INDEX_PATH)

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing parquet: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("Parquet missing 'uid' column.")

    df["__text"] = df.apply(build_text, axis=1)
    uids = df["uid"].tolist()
    texts = df["__text"].tolist()

    client = make_embed_client()

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    existing = get_existing_uids(conn)
    todo = [(u, t) for u, t in zip(uids, texts) if u not in existing and t.strip()]

    print(f"Total docs: {len(uids)}")
    print(f"Embedding {len(todo)} new rows...")

    all_rows = []
    batches = list(chunked(todo, BATCH_SIZE))

    for batch in tqdm(batches):
        buids = [u for u, _ in batch]
        btxts = [t for _, t in batch]

        try:
            embs = embed_batch(client, btxts)
        except Exception as e:
            print("Batch failed:", e)
            continue

        for uid, emb in zip(buids, embs):
            all_rows.append((uid, np.asarray(emb, dtype=np.float32)))

        if len(all_rows) > 500:
            upsert_vectors(conn, all_rows)
            all_rows.clear()

    if all_rows:
        upsert_vectors(conn, all_rows)

    conn.close()

    print("✔ Embeddings updated.")


if __name__ == "__main__":
    main()
