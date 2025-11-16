"""
rag_ask.py — split-resource RAG with logging:
- Embeddings: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_EMBED_DEPLOYMENT
- Chat:       AZURE_OPENAI_CHAT_ENDPOINT / AZURE_OPENAI_CHAT_DEPLOYMENT
- Logs all runs to ./data/rag_logs.sqlite and ./data/rag_logs.jsonl
"""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from openai import AzureOpenAI

from src.common.rag_log import log_query

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/vector_store.faiss")
FAISS_UIDS_PATH = os.getenv("FAISS_UIDS_PATH", "./data/vector_store.uids.json")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

# Embedding env (resource A)
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Chat env (resource B)
CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or os.getenv(
    "AZURE_OPENAI_ENDPOINT", ""
)
CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY") or os.getenv(
    "AZURE_OPENAI_API_KEY", ""
)
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

TEXT_CHAR_LIMIT = int(os.getenv("RAG_TEXT_CHAR_LIMIT", "8000"))
SNIPPET_CHAR_LIMIT = int(os.getenv("RAG_SNIPPET_CHAR_LIMIT", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))


def make_embed_client() -> AzureOpenAI:
    if not EMBED_ENDPOINT or not EMBED_KEY:
        raise RuntimeError(
            "Embedding env missing: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY."
        )
    return AzureOpenAI(
        api_key=EMBED_KEY, api_version=API_VERSION, azure_endpoint=EMBED_ENDPOINT
    )


def make_chat_client() -> AzureOpenAI:
    if not CHAT_ENDPOINT or not CHAT_KEY:
        raise RuntimeError(
            "Chat env missing: AZURE_OPENAI_CHAT_ENDPOINT / AZURE_OPENAI_CHAT_API_KEY."
        )
    return AzureOpenAI(
        api_key=CHAT_KEY, api_version=API_VERSION, azure_endpoint=CHAT_ENDPOINT
    )


def embed_text(text: str) -> np.ndarray:
    client = make_embed_client()
    text = (text or "").strip()[:TEXT_CHAR_LIMIT]
    if not text:
        raise ValueError("Empty text to embed.")
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def load_docs_df() -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid'.")
    for col in ("title", "abstract", "summary", "description", "url", "source"):
        if col not in df.columns:
            df[col] = ""
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
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = rows[0][1]
    mat = np.empty((len(rows), dim), dtype=np.float32)
    uids = []
    for i, (uid, _dim, blob) in enumerate(rows):
        uids.append(uid)
        mat[i, :] = np.frombuffer(blob, dtype=np.float32)
    return uids, mat


def have_faiss() -> bool:
    try:
        import faiss  # noqa

        return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_UIDS_PATH)
    except Exception:
        return False


def faiss_search(query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    import faiss

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_UIDS_PATH, "r", encoding="utf-8") as f:
        uids = json.load(f)
    q = query_vec.reshape(1, -1).astype(np.float32)
    distances, idxs = index.search(q, top_k)
    out = []
    for d, i in zip(distances[0], idxs[0]):
        if i == -1:
            continue
        out.append((uids[i], float(d)))  # L2 distance
    return out


def cosine_search(
    query_vec: np.ndarray, uids: List[str], mat: np.ndarray, top_k: int
) -> List[Tuple[str, float]]:
    if mat.size == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = M.dot(q.astype(np.float32))
    k = min(top_k, sims.shape[0])
    idxs = np.argpartition(-sims, k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]
    return [(uids[i], float(sims[i])) for i in idxs]


def build_snippet(row: pd.Series) -> str:
    body = row["abstract"] or row["summary"] or row["description"] or ""
    body = str(body).strip().replace("\n", " ")
    title = str(row["title"]).strip()
    return f"Title: {title}\nSource: {row['source']}\nURL: {row['url']}\nSummary: {body[:SNIPPET_CHAR_LIMIT]}"


def assemble_context(
    df: pd.DataFrame, ordered_uids: List[str]
) -> Tuple[str, List[dict]]:
    snippets, metas, total = [], [], 0
    for rank, uid in enumerate(ordered_uids, start=1):
        m = df.loc[df["uid"] == uid]
        if m.empty:
            continue
        r = m.iloc[0]
        snip = build_snippet(r)
        if total + len(snip) > MAX_CONTEXT_CHARS:
            break
        snippets.append(snip)
        metas.append(
            {
                "rank": rank,
                "uid": uid,
                "title": r["title"],
                "url": r["url"],
                "source": r["source"],
            }
        )
        total += len(snip)
    return "\n\n---\n\n".join(snippets), metas


SYSTEM_PROMPT = (
    "You are a helpful research assistant for endometriosis. "
    "Answer with concise, evidence-focused summaries. "
    "Use only the provided context; if uncertain, say so."
)


def chat_answer(question: str, context: str) -> Tuple[str, Dict]:
    client = make_chat_client()
    start = time.time()
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    latency_ms = int((time.time() - start) * 1000)
    # usage is sometimes present; if not, keep empty dict
    usage = getattr(resp, "usage", None)
    usage_dict = (
        {
            "total_tokens": getattr(usage, "total_tokens", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        if usage
        else {}
    )
    answer = resp.choices[0].message.content.strip()
    return answer, {"latency_ms": latency_ms, "usage": usage_dict}


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/pipelines/rag_ask.py "your question" [top_k]')
        sys.exit(1)
    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # 1) Embed query
    qvec = embed_text(question)

    # 2) Retrieve
    if have_faiss():
        matches = faiss_search(qvec, top_k)  # list[(uid, L2)]
        ordered = [uid for uid, _dist in matches]
        used_faiss = True
    else:
        uids, mat = load_all_vectors_from_db()
        matches = cosine_search(qvec, uids, mat, top_k)  # list[(uid, cosine)]
        ordered = [uid for uid, _sim in matches]
        used_faiss = False

    if not ordered:
        print("No matches found. Did you run embeddings.py?")
        sys.exit(1)

    # 3) Build context + meta docs
    df = load_docs_df()
    context, metas = assemble_context(df, ordered)

    # 4) Chat
    answer, perf = chat_answer(question, context)

    # 5) Print result
    print("\n================= ANSWER =================")
    print(answer)
    print("\n================= SOURCES ================")
    for m in metas:
        url_part = f" | {m['url']}" if m.get("url") else ""
        print(
            f"{m['rank']:>2}. {m['title'][:100]}  [uid={m['uid'][:8]} | {m['source']}]{url_part}"
        )
    print(f"\n(retrieval={'FAISS/L2' if used_faiss else 'cosine/SQLite'})")
    if perf.get("latency_ms") is not None:
        print(
            f"(chat latency: {perf['latency_ms']} ms, tokens: {perf.get('usage',{}).get('total_tokens')})"
        )

    # 6) Log to cache
    ts_utc = datetime.now(timezone.utc).isoformat()
    doc_rows = []
    # Attach scores if we have them
    rank_map = {m["uid"]: m["rank"] for m in metas}
    score_map = {}
    if used_faiss:
        score_map.update({uid: float(dist) for uid, dist in matches})
    else:
        score_map.update({uid: float(sim) for uid, sim in matches})

    for m in metas:
        doc_rows.append(
            {
                "rank": rank_map.get(m["uid"], m["rank"]),
                "uid": m["uid"],
                "score": score_map.get(
                    m["uid"]
                ),  # L2 distance (lower better) OR cosine (higher better)
                "title": m["title"],
                "source": m["source"],
                "url": m["url"],
            }
        )

    log_query(
        ts_utc=ts_utc,
        question=question,
        top_k=top_k,
        used_faiss=used_faiss,
        embed_endpoint=EMBED_ENDPOINT,
        embed_deployment=EMBED_DEPLOYMENT,
        chat_endpoint=CHAT_ENDPOINT,
        chat_deployment=CHAT_DEPLOYMENT,
        latency_ms=perf.get("latency_ms", 0),
        usage=perf.get("usage", {}),
        answer=answer,
        docs=doc_rows,
    )


if __name__ == "__main__":
    main()
