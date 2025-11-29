"""
Streamlit RAG Panel for Endo Ecosystem PoC
- Azure OpenAI embeddings + GPT-4o-mini / GPT-5.1-chat
- Multi-turn chat with RAG context injection on latest question
- Dataset updater (pull_all → normalize_load → embeddings → health_check)
- Authentication via streamlit-authenticator 0.4.2 (auth.yaml at repo root)
"""

from __future__ import annotations
import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI


st.subheader("📌 DEBUG: Path Check (TEMPORARY)")
st.code(
    f"""
cwd = {os.getcwd()}
file = {__file__}
repo_root = {Path(__file__).resolve().parents[2]}
"""
)

# ============================================================
# Streamlit CONFIG (MUST be first Streamlit call)
# ============================================================
st.set_page_config(page_title="Endo PoC — RAG Panel", layout="wide")

# ============================================================
# Simple Streamlit Password Gate
# ============================================================

load_dotenv()

APP_PASSWORD = st.secrets.get("APP_PASSWORD")

entered_pw = st.text_input("Enter access password", type="password")

if APP_PASSWORD and entered_pw != APP_PASSWORD:
    st.stop()


# ============================================================
# Paths / Environment
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# Ensure project root import path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.common.rag_log import log_query
except Exception:
    from common.rag_log import log_query
# ============================================================
# Authentication disabled (dev mode)
# ============================================================
st.caption("🔓 Authentication disabled — development mode")
name = "Developer"
username = "dev"


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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG CONFIG
# ============================================================
PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
LOG_DB_PATH = "./data/rag_logs.sqlite"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/vector_store.faiss")
FAISS_UIDS_PATH = os.getenv("FAISS_UIDS_PATH", "./data/vector_store.uids.json")

# Azure OpenAI
EMBED_ENDPOINT = st.secrets.get(
    "AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")
)
EMBED_KEY = st.secrets.get(
    "AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", "")
)

EMBED_DEPLOYMENT = st.secrets.get(
    "AZURE_OPENAI_EMBED_DEPLOYMENT",
    os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"),
)

CHAT_ENDPOINT = st.secrets.get("AZURE_OPENAI_CHAT_ENDPOINT", EMBED_ENDPOINT)
CHAT_KEY = st.secrets.get("AZURE_OPENAI_CHAT_API_KEY", EMBED_KEY)

CHAT_DEPLOYMENT_4O = st.secrets.get(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_4O",
    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_4O", "gpt-4o-mini"),
)
CHAT_DEPLOYMENT_5 = st.secrets.get(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_5",
    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_5", "gpt-5.1-chat"),
)

TEXT_CHAR_LIMIT = int(os.getenv("RAG_TEXT_CHAR_LIMIT", "8000"))
SNIPPET_CHAR_LIMIT = int(os.getenv("RAG_SNIPPET_CHAR_LIMIT", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))


# ============================================================
# Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_deployment" not in st.session_state:
    st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_5


# ============================================================
# Azure OpenAI Clients
# ============================================================
def make_embed_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=EMBED_KEY,
        azure_endpoint=EMBED_ENDPOINT,
        api_version="2023-05-15",
    )


def make_chat_client_4o() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version="2025-01-01-preview",
    )


def make_chat_client_5() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version="2025-04-01-preview",
    )


# ============================================================
# Data Loaders
# ============================================================
@st.cache_data(show_spinner=False)
def load_docs_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid'")
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
    uids = []
    for i, (uid, d, blob) in enumerate(rows):
        uids.append(uid)
        mat[i, :] = np.frombuffer(blob, dtype=np.float32)
    return uids, mat


def have_faiss() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_UIDS_PATH)


def embed_text(q: str) -> np.ndarray:
    client = make_embed_client()
    q = q.strip()[:TEXT_CHAR_LIMIT]
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=q)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def cosine_search(query_vec, uids, mat, top_k) -> List[Tuple[str, float]]:
    if mat.size == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = M.dot(q.astype(np.float32))
    k = min(top_k, sims.shape[0])
    idxs = np.argpartition(-sims, k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]
    return [(uids[i], float(sims[i])) for i in idxs]


def faiss_search(query_vec, top_k):
    import faiss

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_UIDS_PATH, "r") as f:
        uids = json.load(f)
    q = query_vec.reshape(1, -1).astype(np.float32)
    d, idx = index.search(q, top_k)
    out = []
    for dist, i in zip(d[0], idx[0]):
        if i != -1:
            out.append((uids[i], float(dist)))
    return out


def build_snippet(row):
    body = row["abstract"] or row["summary"] or row["description"] or ""
    body = str(body).replace("\n", " ")
    return (
        f"Title: {row['title']}\n"
        f"Source: {row['source']}\n"
        f"URL: {row['url']}\n"
        f"Summary: {body[:SNIPPET_CHAR_LIMIT]}"
    )


def assemble_context(df: pd.DataFrame, ordered_uids: List[str]):
    snippets = []
    metas = []
    total = 0
    for rank, uid in enumerate(ordered_uids, start=1):
        m = df.loc[df["uid"] == uid]
        if m.empty:
            continue
        r = m.iloc[0]
        snip = build_snippet(r)
        if total + len(snip) > MAX_CONTEXT_CHARS:
            break
        total += len(snip)
        snippets.append(snip)
        metas.append(
            dict(rank=rank, uid=uid, title=r["title"], url=r["url"], source=r["source"])
        )
    return "\n\n---\n\n".join(snippets), metas


# ============================================================
# Chat logic (multi-turn)
# ============================================================
def chat_answer(question: str, context: str, history, deployment: str):
    t0 = time.time()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant for endometriosis. "
                "Use ONLY the provided context for factual claims; "
                "if unsupported, say you're unsure."
            ),
        }
    ]

    for m in history:
        messages.append(dict(role=m["role"], content=m["content"]))

    messages.append(
        dict(role="user", content=f"Question: {question}\n\nContext:\n{context}")
    )

    if deployment == CHAT_DEPLOYMENT_5:
        client = make_chat_client_5()
        resp = client.responses.create(
            model=deployment,
            input=messages,
            max_output_tokens=1400,
        )
        answer = resp.output_text.strip()
        usage = {}
    else:
        client = make_chat_client_4o()
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1400,
        )
        answer = resp.choices[0].message.content.strip()
        u = getattr(resp, "usage", None)
        usage = {"total_tokens": getattr(u, "total_tokens", None)} if u else {}

    return answer, {
        "latency_ms": int((time.time() - t0) * 1000),
        "usage": usage,
    }


# ============================================================
# Layout
# ============================================================
st.title("Endo PoC — RAG Panel")

tab_chat, tab_history = st.tabs(["💬 Chat", "🕓 History"])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Model Settings")
    st.markdown("**Embeddings Endpoint:** 🔒 Hidden")
    st.markdown("**Chat Endpoint:** 🔒 Hidden")

    model_choice = st.radio("Model", ["GPT-5.1 (default)", "GPT-4o-mini"], index=0)
    st.session_state["chat_deployment"] = (
        CHAT_DEPLOYMENT_5 if model_choice == "GPT-5.1 (default)" else CHAT_DEPLOYMENT_4O
    )

    if st.button("🆕 New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.subheader("Dataset Controls")

    if st.button("🔄 Update Dataset"):
        st.write("### Updating Dataset…")
        progress = st.progress(0)
        output = st.empty()

        def run_step(label: str, cmd: str, pct: int):
            output.write(f"**{label}** running…")
            result = os.popen(cmd).read()
            output.write(f"```\n{result}\n```")
            progress.progress(pct)

        try:
            run_step("Pulling raw data", "python src/pipelines/pull_all.py", 25)
            run_step(
                "Normalizing + Loading",
                "python src/pipelines/normalize_load.py",
                50,
            )
            run_step("Generating Embeddings", "python src/pipelines/embeddings.py", 75)
            run_step("Health Check", "python src/pipelines/health_check.py", 100)
            st.success("Dataset successfully updated!")
        except Exception as e:
            st.error(f"❌ Update failed: {e}")


# ============================================================
# Tab 1 — Chat
# ============================================================
with tab_chat:
    top_k = st.slider("Top-K", 1, 10, 5)
    use_faiss = st.checkbox(
        "Prefer FAISS (if available)",
        True,
        help="Use FAISS ANN search when index is present.",
    )

    st.markdown("---")
    st.subheader("Conversation")

    # Render chat history
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Assistant:** {m['content']}")

    st.markdown("---")

    # Chat input bottom
    question = st.text_input(
        "Your message",
        placeholder="Ask about mechanisms, biomarkers, clinical trials…",
        key="chat_input",
    )

    if st.button("Send", use_container_width=True) and question.strip():
        q = question.strip()
        st.session_state["messages"].append({"role": "user", "content": q})

        try:
            df = load_docs_df()
            uids, mat = load_sqlite_vectors()
            qvec = embed_text(q)

            if use_faiss and have_faiss():
                matches = faiss_search(qvec, top_k)
                ordered = [u for u, _ in matches]
            else:
                matches = cosine_search(qvec, uids, mat, top_k)
                ordered = [u for u, _ in matches]

            if not ordered:
                st.warning("No document matches found.")
                st.stop()

            context, metas = assemble_context(df, ordered)

            history = st.session_state["messages"][:-1]

            answer, perf = chat_answer(
                question=q,
                context=context,
                history=history,
                deployment=st.session_state["chat_deployment"],
            )

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )

            st.subheader("Answer")
            st.write(answer)
            st.caption(
                f"Latency {perf['latency_ms']} ms • Tokens {perf['usage'].get('total_tokens')}"
            )

            st.divider()
            st.subheader("Sources")
            scores = {u: float(s) for u, s in matches}

            for m in metas:
                with st.expander(f"{m['rank']}. {m['title'][:100]}"):
                    row = df.loc[df["uid"] == m["uid"]].iloc[0]
                    st.markdown(
                        f"**UID:** {m['uid']}  |  **Source:** {m['source']}  |  **Link:** {m['url'] or '—'}"
                    )
                    text = (
                        row["abstract"] or row["summary"] or row["description"] or ""
                    )[:1200]
                    st.write(text)

            # -------- LOG QUERY --------
            docs = [
                {
                    "rank": m["rank"],
                    "uid": m["uid"],
                    "score": scores.get(m["uid"]),
                    "title": m["title"],
                    "source": m["source"],
                    "url": m["url"],
                }
                for m in metas
            ]

            ts = datetime.now(timezone.utc).isoformat()
            log_query(
                ts,
                q,
                top_k,
                use_faiss,
                EMBED_ENDPOINT,
                EMBED_DEPLOYMENT,
                CHAT_ENDPOINT,
                st.session_state["chat_deployment"],
                perf["latency_ms"],
                perf["usage"],
                answer,
                docs,
            )

        except Exception as e:
            st.error(f"Error: {e}")


# ============================================================
# Tab 2 — History
# ============================================================
with tab_history:
    if not os.path.exists(LOG_DB_PATH):
        st.info("No logs yet.")
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
            for row in rows:
                qid, ts, q, topk, faiss_flag, lat, toks = row
                col1, col2, col3 = st.columns([5, 2, 1])

                with col1:
                    st.markdown(f"**{qid}. {q[:100]}**")
                    st.caption(ts)

                with col2:
                    st.caption(f"Top {topk} | {'FAISS' if faiss_flag else 'cosine'}")

                with col3:
                    if st.button("Re-run", key=f"rerun_{qid}"):
                        st.session_state["messages"].append(
                            {"role": "user", "content": q}
                        )
                        st.rerun()
