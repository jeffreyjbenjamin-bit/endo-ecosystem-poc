import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# --- ensure package path works when run via Streamlit ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.common.rag_log import log_query
except Exception:
    from common.rag_log import log_query

# ---------- Config / Paths ----------
PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
LOG_DB_PATH = "./data/rag_logs.sqlite"
FAISS_INDEX_PATH = "./data/vector_store.faiss"
FAISS_UIDS_PATH = "./data/vector_store.uids.json"
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

# Embeddings (resource A)
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

# Chat (resource B — may differ)
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


# ---------- Clients ----------
def make_embed_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=EMBED_KEY, api_version=API_VERSION, azure_endpoint=EMBED_ENDPOINT
    )


def make_chat_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CHAT_KEY, api_version=API_VERSION, azure_endpoint=CHAT_ENDPOINT
    )


# ---------- Data helpers ----------
@st.cache_data(show_spinner=False)
def load_docs_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid'")
    for col in ("title", "abstract", "summary", "description", "url", "source"):
        if col not in df.columns:
            df[col] = ""
    return df


@st.cache_resource(show_spinner=False)
def load_sqlite_vectors() -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(DB_PATH):
        return [], np.zeros((0, 0), dtype=np.float32)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT uid, dim, vector FROM vectors")
    rows = cur.fetchall()
    con.close()
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

        return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_UIDS_PATH)
    except Exception:
        return False


def embed_text(q: str) -> np.ndarray:
    client = make_embed_client()
    q = (q or "").strip()[:TEXT_CHAR_LIMIT]
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=q)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


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


def faiss_search(query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    import faiss

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_UIDS_PATH, "r", encoding="utf-8") as f:
        uids = json.load(f)
    q = query_vec.reshape(1, -1).astype(np.float32)
    d, idx = index.search(q, top_k)
    out = []
    for dist, i in zip(d[0], idx[0]):
        if i == -1:
            continue
        out.append((uids[i], float(dist)))
    return out


def build_snippet(row: pd.Series) -> str:
    body = row["abstract"] or row["summary"] or row["description"] or ""
    body = str(body).strip().replace("\n", " ")
    title = str(row["title"]).strip()
    return f"Title: {title}\nSource: {row['source']}\nURL: {row['url']}\nSummary: {body[:SNIPPET_CHAR_LIMIT]}"


def assemble_context(df: pd.DataFrame, ordered_uids: List[str]) -> Tuple[str, list]:
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


def chat_answer(question: str, context: str) -> Tuple[str, dict]:
    client = make_chat_client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant for endometriosis. Use only the provided context; if uncertain, say so.",
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    latency = int((time.time() - t0) * 1000)
    usage = getattr(resp, "usage", None)
    usage_dict = {"total_tokens": getattr(usage, "total_tokens", None)} if usage else {}
    return resp.choices[0].message.content.strip(), {
        "latency_ms": latency,
        "usage": usage_dict,
    }


# ---------- UI ----------
st.set_page_config(page_title="Endo PoC — RAG Panel", layout="wide")
st.title("Endo PoC — RAG Panel")

tab1, tab2 = st.tabs(["🔍 Ask", "🕓 History"])

# Sidebar info (always visible)
with st.sidebar:
    st.subheader("Endpoints")
    st.markdown("**Embeddings:**")
    st.code(EMBED_ENDPOINT or "(missing)", language="text")
    st.markdown("**Chat:**")
    st.code(CHAT_ENDPOINT or "(missing)", language="text")

# ---------- TAB 1 — ASK ----------
with tab1:
    top_k = st.slider("Top-K results", 1, 10, 5)
    use_faiss_pref = st.checkbox("Prefer FAISS (if available)", True)
    query = st.text_input(
        "Ask a question",
        placeholder="e.g., What inflammatory biomarkers are linked to endometriosis?",
    )
    go = st.button("Run query", use_container_width=True)

    if go and query.strip():
        try:
            df = load_docs_df()
            uids, mat = load_sqlite_vectors()
            qvec = embed_text(query)
            used_faiss = False
            if use_faiss_pref and have_faiss():
                matches = faiss_search(qvec, top_k)
                ordered = [u for u, _ in matches]
                used_faiss = True
            else:
                matches = cosine_search(qvec, uids, mat, top_k)
                ordered = [u for u, _ in matches]
            if not ordered:
                st.warning("No matches found.")
                st.stop()

            context, metas = assemble_context(df, ordered)
            answer, perf = chat_answer(query, context)

            st.subheader("Answer")
            st.write(answer)
            st.caption(
                f"Latency {perf.get('latency_ms')} ms, Tokens {perf.get('usage',{}).get('total_tokens')}"
            )

            st.divider()
            st.subheader("Sources")
            for m in metas:
                with st.expander(f"{m['rank']}. {m['title'][:100]}"):
                    row = df.loc[df["uid"] == m["uid"]].iloc[0]
                    st.markdown(
                        f"**UID:** `{m['uid']}`  **Source:** {m['source']}  **URL:** {m['url'] or '—'}"
                    )
                    st.write(
                        (row["abstract"] or row["summary"] or row["description"] or "")[
                            :1200
                        ]
                    )

            # ---- LOG ----
            score_map = {u: float(s) for u, s in matches}
            docs = [
                {
                    "rank": m["rank"],
                    "uid": m["uid"],
                    "score": score_map.get(m["uid"]),
                    "title": m["title"],
                    "source": m["source"],
                    "url": m["url"],
                }
                for m in metas
            ]
            ts = datetime.now(timezone.utc).isoformat()
            log_query(
                ts,
                query,
                top_k,
                used_faiss,
                EMBED_ENDPOINT,
                EMBED_DEPLOYMENT,
                CHAT_ENDPOINT,
                CHAT_DEPLOYMENT,
                perf.get("latency_ms", 0),
                perf.get("usage", {}),
                answer,
                docs,
            )
            st.success("Logged to rag_logs.sqlite/jsonl")

        except Exception as e:
            st.error(f"Error: {e}")

# ---------- TAB 2 — HISTORY ----------
with tab2:
    if not os.path.exists(LOG_DB_PATH):
        st.info("No logs yet — run a query in the Ask tab first.")
    else:
        con = sqlite3.connect(LOG_DB_PATH)
        cur = con.cursor()
        cur.execute(
            """
            SELECT id,ts_utc,question,top_k,used_faiss,latency_ms,total_tokens
            FROM queries ORDER BY id DESC LIMIT 50
        """
        )
        rows = cur.fetchall()
        con.close()

        if not rows:
            st.info("Log table empty.")
        else:
            st.subheader("Recent Queries")
            for r in rows:
                qid, ts, q, topk, faiss, lat, toks = r
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.markdown(f"**{qid}. {q[:100]}**")
                    st.caption(f"{ts}")
                with col2:
                    st.caption(f"Top {topk} | {'FAISS' if faiss else 'cosine'}")
                with col3:
                    if st.button("Re-run", key=f"r{qid}"):
                        st.session_state["rerun_query"] = q
                        st.experimental_rerun()

# Allow single-click re-run
if "rerun_query" in st.session_state:
    st.switch_page(
        "src/ui/rag_panel.py"
    )  # reloads and puts query in box (works best when app rerun manually)
