"""
Streamlit RAG Panel for Endo Ecosystem PoC (Cloud Version)

- Assumes precomputed data under: <repo_root>/data/
    - documents.parquet
    - vector_store.sqlite
    - (optional) vector_store.faiss + vector_store.uids.json
- Uses Azure OpenAI:
    - Embeddings: text-embedding-3-large
    - Chat: gpt-4o-mini or gpt-5.1-chat (selectable in sidebar)
- Multi-turn chat with RAG context on latest question
- Simple password gate via APP_PASSWORD (env var or Streamlit secret)
"""

from __future__ import annotations

import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# ============================================================
# Page config (must be first Streamlit call)
# ============================================================
st.set_page_config(page_title="Endo PoC — RAG Panel", layout="wide")

# ============================================================
# Repo root / environment
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env if present (for local dev); Streamlit Cloud should use secrets
env_path = REPO_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Ensure repo root on sys.path so "src" imports work
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import logging helper; if missing, fall back to no-op
try:
    from src.common.rag_log import log_query
except Exception:  # pragma: no cover - safety fallback

    def log_query(
        ts_utc: str,
        question: str,
        top_k: int,
        used_faiss: bool,
        embed_endpoint: str,
        embed_deployment: str,
        chat_endpoint: str,
        chat_deployment: str,
        latency_ms: int,
        usage: Dict[str, Any],
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> None:
        """No-op logger fallback for environments without rag_log."""
        return


# ============================================================
# Simple password gate (APP_PASSWORD from env or secrets)
# ============================================================
APP_PASSWORD = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD", "")

if "pw_ok" not in st.session_state:
    st.session_state["pw_ok"] = False

if APP_PASSWORD:
    if not st.session_state["pw_ok"]:
        st.subheader("🔒 Endo Ecosystem Access")
        pwd = st.text_input("Enter application password", type="password")
        if not pwd:
            st.stop()
        if pwd != APP_PASSWORD:
            st.error("Incorrect password")
            st.stop()
        # Correct password
        st.session_state["pw_ok"] = True
        st.rerun()
else:
    st.info(
        "No application password set. "
        "Set APP_PASSWORD in Streamlit secrets or environment for production."
    )


# ============================================================
# Styling (Blue/Green Palette)
# ============================================================
blue = "#1F6FEB"
green = "#2ECC71"
bg_light = "#F5FAFD"
text_dark = "#0A2540"

st.markdown(
    f"""
    <style>
        .main {{
            background-color: {bg_light};
        }}
        .stButton>button {{
            background-color: {blue};
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: {green};
        }}
        .st-emotion-cache-10trblm {{
            color: {text_dark};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# RAG CONFIG
# ============================================================
PARQUET_PATH = os.getenv("PARQUET_PATH", str(REPO_ROOT / "data" / "documents.parquet"))
DB_PATH = os.getenv("VECTOR_DB_PATH", str(REPO_ROOT / "data" / "vector_store.sqlite"))
LOG_DB_PATH = str(REPO_ROOT / "data" / "rag_logs.sqlite")
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH", str(REPO_ROOT / "data" / "vector_store.faiss")
)
FAISS_UIDS_PATH = os.getenv(
    "FAISS_UIDS_PATH", str(REPO_ROOT / "data" / "vector_store.uids.json")
)

# Azure OpenAI
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or EMBED_ENDPOINT
CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY") or EMBED_KEY

CHAT_DEPLOYMENT_4O = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_4O", "gpt-4o-mini")
CHAT_DEPLOYMENT_5 = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_5", "gpt-5.1-chat")

TEXT_CHAR_LIMIT = int(os.getenv("RAG_TEXT_CHAR_LIMIT", "8000"))
SNIPPET_CHAR_LIMIT = int(os.getenv("RAG_SNIPPET_CHAR_LIMIT", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

# ============================================================
# Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # type: ignore[list-item]

if "chat_deployment" not in st.session_state:
    st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_5


# ============================================================
# Azure OpenAI Clients
# ============================================================
def make_embed_client() -> AzureOpenAI:
    if not EMBED_ENDPOINT or not EMBED_KEY:
        raise RuntimeError("Embedding endpoint or API key not configured.")
    return AzureOpenAI(
        api_key=EMBED_KEY,
        azure_endpoint=EMBED_ENDPOINT,
        api_version="2023-05-15",  # embeddings API version
    )


def make_chat_client_4o() -> AzureOpenAI:
    if not CHAT_ENDPOINT or not CHAT_KEY:
        raise RuntimeError("Chat endpoint or API key not configured.")
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version="2025-01-01-preview",  # gpt-4o-mini
    )


def make_chat_client_5() -> AzureOpenAI:
    if not CHAT_ENDPOINT or not CHAT_KEY:
        raise RuntimeError("Chat endpoint or API key not configured.")
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version="2025-04-01-preview",  # gpt-5.1-chat via responses API
    )


# ============================================================
# Data Loaders
# ============================================================
@st.cache_data(show_spinner=False)
def load_docs_df() -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"documents.parquet not found at {PARQUET_PATH}. "
            "Make sure /data is committed to the repo."
        )
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid' column.")
    for c in ("title", "abstract", "summary", "description", "url", "source"):
        if c not in df.columns:
            df[c] = ""
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
    uids: List[str] = []
    for i, (uid, _dim, blob) in enumerate(rows):
        uids.append(uid)
        mat[i, :] = np.frombuffer(blob, dtype=np.float32)
    return uids, mat


def have_faiss() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_UIDS_PATH)


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
    import faiss  # local import to avoid issues if faiss isn't installed

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_UIDS_PATH, "r", encoding="utf-8") as f:
        uids = json.load(f)
    q = query_vec.reshape(1, -1).astype(np.float32)
    d, idx = index.search(q, top_k)
    out: List[Tuple[str, float]] = []
    for dist, i in zip(d[0], idx[0]):
        if i == -1:
            continue
        out.append((uids[i], float(dist)))
    return out


def build_snippet(row: pd.Series) -> str:
    body = row["abstract"] or row["summary"] or row["description"] or ""
    body = str(body).strip().replace("\n", " ")
    title = str(row["title"]).strip()
    return (
        f"Title: {title}\n"
        f"Source: {row['source']}\n"
        f"URL: {row['url']}\n"
        f"Summary: {body[:SNIPPET_CHAR_LIMIT]}"
    )


def assemble_context(
    df: pd.DataFrame, ordered_uids: List[str]
) -> Tuple[str, List[Dict[str, Any]]]:
    snippets: List[str] = []
    metas: List[Dict[str, Any]] = []
    total = 0
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


# ============================================================
# Chat logic (multi-turn with RAG on latest question)
# ============================================================
def chat_answer(
    question: str,
    context: str,
    history: List[Dict[str, str]],
    chat_deployment: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Multi-turn chat:
      - history: previous user/assistant messages (no context injected)
      - question: current user question
      - context: RAG context built from this latest question
    """
    t0 = time.time()

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant for endometriosis. "
                "Use ONLY the provided context for factual claims; "
                "if the context does not support an answer, say you are unsure."
            ),
        }
    ]

    # Add previous turns
    for m in history:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    # Latest user with explicit context
    messages.append(
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        }
    )

    usage_dict: Dict[str, Any] = {}

    if chat_deployment == CHAT_DEPLOYMENT_5:
        client = make_chat_client_5()
        resp = client.responses.create(
            model=chat_deployment,
            input=messages,
            max_output_tokens=1400,  # doubled vs earlier
        )
        answer = resp.output_text.strip() if hasattr(resp, "output_text") else str(resp)
    else:
        client = make_chat_client_4o()
        resp = client.chat.completions.create(
            model=chat_deployment,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1400,
        )
        answer = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage is not None:
            usage_dict = {"total_tokens": getattr(usage, "total_tokens", None)}

    latency = int((time.time() - t0) * 1000)
    return answer, {"latency_ms": latency, "usage": usage_dict}


# ============================================================
# Layout: title, tabs, sidebar
# ============================================================
st.title("Endo PoC — RAG Panel")

tab_chat, tab_history = st.tabs(["💬 Chat", "🕓 History"])

# Sidebar: model selector + disabled dataset updater + new chat
with st.sidebar:
    st.subheader("Model Settings")
    st.markdown("*(Endpoints hidden for security)*")
    st.markdown("**Embeddings Endpoint:** 🔒 Hidden")
    st.markdown("**Chat Endpoint:** 🔒 Hidden")

    model_choice = st.radio(
        "Model",
        ["GPT-5.1 (default)", "GPT-4o-mini"],
        index=0,
    )

    if model_choice == "GPT-5.1 (default)":
        st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_5
    else:
        st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_4O

    if st.button("🆕 New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.subheader("Dataset Controls")
    st.button(
        "🔄 Update Dataset (local only)",
        disabled=True,
        help="Dataset updates are disabled on the hosted app. "
        "Run the pipelines locally to regenerate embeddings.",
    )

# ============================================================
# Tab 1: Chat
# ============================================================
with tab_chat:
    top_k = st.slider("Top-K results", 1, 10, 5)
    use_faiss_pref = st.checkbox(
        "Prefer FAISS (if available)",
        True,
        help="If enabled and FAISS index exists, use ANN search; "
        "otherwise cosine search over SQLite vectors.",
    )

    st.markdown("---")
    st.subheader("Conversation")

    # Render existing conversation
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        elif m["role"] == "assistant":
            st.markdown(f"**Assistant:** {m['content']}")

    st.markdown("---")

    # Chat input at the bottom
    user_input = st.text_input(
        "Your message",
        placeholder="Ask about endometriosis biomarkers, trials, mechanisms, etc.",
        key="chat_input_field",
    )
    send = st.button("Send", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()
        st.session_state["messages"].append({"role": "user", "content": question})

        try:
            df = load_docs_df()
            uids, mat = load_sqlite_vectors()
            qvec = embed_text(question)

            used_faiss = False
            if use_faiss_pref and have_faiss():
                matches = faiss_search(qvec, top_k)
                ordered = [u for u, _ in matches]
                used_faiss = True
            else:
                matches = cosine_search(qvec, uids, mat, top_k)
                ordered = [u for u, _ in matches]

            if not ordered:
                st.warning("No matches found in the current dataset.")
                st.stop()

            context, metas = assemble_context(df, ordered)

            history = st.session_state["messages"][:-1]

            answer, perf = chat_answer(
                question=question,
                context=context,
                history=history,
                chat_deployment=st.session_state["chat_deployment"],
            )

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )

            st.subheader("Answer")
            st.write(answer)
            st.caption(
                f"Latency {perf.get('latency_ms')} ms, "
                f"Tokens {perf.get('usage', {}).get('total_tokens')}"
            )

            st.divider()
            st.subheader("Sources")
            score_map = {u: float(s) for u, s in matches}
            docs_for_log: List[Dict[str, Any]] = []
            for m in metas:
                with st.expander(f"{m['rank']}. {m['title'][:100]}"):
                    row = df.loc[df["uid"] == m["uid"]].iloc[0]
                    st.markdown(
                        f"**UID:** `{m['uid']}`  "
                        f"**Source:** {m['source']}  "
                        f"**URL:** {m['url'] or '—'}"
                    )
                    st.write(
                        (row["abstract"] or row["summary"] or row["description"] or "")[
                            :1200
                        ]
                    )
                docs_for_log.append(
                    {
                        "rank": m["rank"],
                        "uid": m["uid"],
                        "score": score_map.get(m["uid"]),
                        "title": m["title"],
                        "source": m["source"],
                        "url": m["url"],
                    }
                )

            ts = datetime.now(timezone.utc).isoformat()
            log_query(
                ts,
                question,
                top_k,
                used_faiss,
                EMBED_ENDPOINT,
                EMBED_DEPLOYMENT,
                CHAT_ENDPOINT,
                st.session_state["chat_deployment"],
                perf.get("latency_ms", 0),
                perf.get("usage", {}),
                answer,
                docs_for_log,
            )
            st.success("Logged to rag_logs.sqlite/jsonl")

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:  # pragma: no cover
            st.error(f"Error while answering: {e}")

# ============================================================
# Tab 2: History
# ============================================================
with tab_history:
    if not os.path.exists(LOG_DB_PATH):
        st.info("No logs yet — run a query in the Chat tab first.")
    else:
        con = sqlite3.connect(LOG_DB_PATH)
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, ts_utc, question, top_k, used_faiss, latency_ms, total_tokens
            FROM queries
            ORDER BY id DESC
            LIMIT 50
            """
        )
        rows = cur.fetchall()
        con.close()

        if not rows:
            st.info("Log table empty.")
        else:
            st.subheader("Recent Queries")
            for r in rows:
                qid, ts, q, topk, faiss_flag, lat, toks = r
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.markdown(f"**{qid}. {q[:100]}**")
                    st.caption(f"{ts}")
                with col2:
                    st.caption(f"Top {topk} | {'FAISS' if faiss_flag else 'cosine'}")
                with col3:
                    if st.button("Re-run", key=f"r{qid}"):
                        st.session_state["messages"].append(
                            {"role": "user", "content": q}
                        )
                        st.rerun()
