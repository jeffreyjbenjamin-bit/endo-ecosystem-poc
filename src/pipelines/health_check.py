"""
health_check.py — quick sanity checks for Endo PoC Phase 2

Validates:
- Env vars for embeddings + chat (split resources supported)
- Azure OpenAI connectivity (1 tiny embed + 1 tiny chat)
- Local artifacts: documents.parquet, vector_store.sqlite (count/dim), FAISS files

Usage:
  python src/pipelines/health_check.py
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from openai import AzureOpenAI

# ---------- Config / Paths ----------

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/vector_store.faiss")
FAISS_UIDS_PATH = os.getenv("FAISS_UIDS_PATH", "./data/vector_store.uids.json")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

# Embeddings (resource A)
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Chat (resource B — can equal A)
CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or os.getenv(
    "AZURE_OPENAI_ENDPOINT", ""
)
CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY") or os.getenv(
    "AZURE_OPENAI_API_KEY", ""
)
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")


def mask(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return s[0] + "…" + s[-1]
    return s[:4] + "…" + s[-3:]


def banner(title: str):
    print("\n" + "=" * 12 + f" {title} " + "=" * 12)


def check_env() -> bool:
    banner("ENVIRONMENT")
    ok = True

    print(f"API_VERSION: {API_VERSION}")
    print(f"EMBED_ENDPOINT: {EMBED_ENDPOINT or '(missing)'}")
    print(f"EMBED_KEY: {mask(EMBED_KEY) or '(missing)'}")
    print(f"EMBED_DEPLOYMENT: {EMBED_DEPLOYMENT or '(missing)'}")

    print(f"CHAT_ENDPOINT: {CHAT_ENDPOINT or '(missing)'}")
    print(f"CHAT_KEY: {mask(CHAT_KEY) or '(missing)'}")
    print(f"CHAT_DEPLOYMENT: {CHAT_DEPLOYMENT or '(missing)'}")

    if not (EMBED_ENDPOINT and EMBED_KEY and EMBED_DEPLOYMENT):
        ok = False
        print(
            "❌ Embedding env incomplete (need AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBED_DEPLOYMENT)."
        )
    if not (CHAT_ENDPOINT and CHAT_KEY and CHAT_DEPLOYMENT):
        ok = False
        print(
            "❌ Chat env incomplete (need AZURE_OPENAI_CHAT_ENDPOINT, AZURE_OPENAI_CHAT_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT)."
        )
    return ok


def make_embed_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=EMBED_KEY, api_version=API_VERSION, azure_endpoint=EMBED_ENDPOINT
    )


def make_chat_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CHAT_KEY, api_version=API_VERSION, azure_endpoint=CHAT_ENDPOINT
    )


def check_embeddings() -> bool:
    banner("EMBEDDINGS PING")
    try:
        client = make_embed_client()
        resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input="ping")
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        print(f"✅ Embed OK — dim={vec.shape[0]}")
        return True
    except Exception as e:
        print(f"❌ Embed failed — {type(e).__name__}: {e}")
        print("   Check endpoint/key/deployment, region quota, and API version.")
        return False


def check_chat() -> bool:
    banner("CHAT PING")
    try:
        client = make_chat_client()
        r = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": "reply with 'ok'"}],
            temperature=0.0,
            max_tokens=5,
        )
        msg = r.choices[0].message.content.strip()
        print(f"✅ Chat OK — sample response: {msg!r}")
        return True
    except Exception as e:
        print(f"❌ Chat failed — {type(e).__name__}: {e}")
        print(
            "   Ensure deployment name matches, endpoint/key are from that resource, and quota exists in region."
        )
        return False


def check_parquet() -> bool:
    banner("LOCAL ARTIFACTS — PARQUET")
    if not os.path.exists(PARQUET_PATH):
        print(f"❌ Missing Parquet at {PARQUET_PATH} — run normalize_load.py first.")
        return False
    try:
        df = pd.read_parquet(PARQUET_PATH).fillna("")
        n = len(df)
        print(f"✅ Found {n} rows in {PARQUET_PATH}")
        if "uid" not in df.columns:
            print("❌ 'uid' column missing — verify normalize/dedupe pipeline.")
            return False
        return True
    except Exception as e:
        print(f"❌ Parquet load failed — {type(e).__name__}: {e}")
        return False


def check_vector_db() -> bool:
    banner("LOCAL ARTIFACTS — VECTOR DB")
    if not os.path.exists(DB_PATH):
        print(f"❌ Missing vector DB at {DB_PATH} — run embeddings.py first.")
        return False
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*), MIN(dim), MAX(dim) FROM vectors")
        count, mind, maxd = cur.fetchone()
        con.close()
        dim_info = mind if mind == maxd else f"{mind}..{maxd}"
        print(f"✅ vectors table present — count={count}, dim={dim_info}")
        return (count or 0) > 0
    except Exception as e:
        print(f"❌ Vector DB check failed — {type(e).__name__}: {e}")
        return False


def check_faiss() -> bool:
    banner("LOCAL ARTIFACTS — FAISS (optional)")
    have_index = os.path.exists(FAISS_INDEX_PATH)
    have_uids = os.path.exists(FAISS_UIDS_PATH)
    if have_index and have_uids:
        print(f"✅ FAISS present: {FAISS_INDEX_PATH} + {FAISS_UIDS_PATH}")
        return True
    else:
        print("ℹ️  FAISS not found (this is optional).")
        return True


def main():
    overall_ok = True

    if not check_env():
        overall_ok = False
    if not check_embeddings():
        overall_ok = False
    if not check_chat():
        overall_ok = False
    if not check_parquet():
        overall_ok = False
    if not check_vector_db():
        overall_ok = False
    if not check_faiss():
        overall_ok = False  # only flips if FAISS check threw an exception (it won't)

    banner("RESULT")
    if overall_ok:
        print("✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("❌ One or more checks failed. See sections above for fixes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
