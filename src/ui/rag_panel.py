"""
Streamlit RAG Panel for Endo Ecosystem PoC
- Azure OpenAI embeddings + GPT-4o-mini / GPT-5.1-chat
- Multi-turn chat with RAG context for latest question
- Dataset update button (pull_all → normalize → embed → health_check)
- Lightweight password prompt using Streamlit secrets
- Absolute paths for Streamlit Cloud
"""

from __future__ import annotations
import os
import sys
import time
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# ============================================================
#   PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Endo PoC — RAG Panel", layout="wide")

# ============================================================
#   REPO ROOT / PATH SETUP (WORKS IN STREAMLIT CLOUD)
# ============================================================
file_path = Path(__file__).resolve()
repo_root = file_path.parents[2]  # /mount/src/endo-ecosystem-poc
sys.path.insert(0, str(repo_root))

st.write(f"📁 Repo root resolved to: `{repo_root}`")

# Load .env
load_dotenv(repo_root / ".env")

# ============================================================
#   PASSWORD GATE (Simple)
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 Endo Ecosystem PoC Login")

    pwd = st.text_input("Enter password:", type="password")

    if st.button("Login"):
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()

# If authenticated, continue:
st.success("Authenticated")
st.caption("Use sidebar to update dataset or switch models.")

# ============================================================
#   ABSOLUTE PATHS FOR STREAMLIT CLOUD
# ============================================================
DATA_DIR = repo_root / "data"
PIPELINE_DIR = repo_root / "src" / "pipelines"

PARQUET_PATH = DATA_DIR / "documents.parquet"
DB_PATH = DATA_DIR / "vector_store.sqlite"
LOG_DB_PATH = DATA_DIR / "rag_logs.sqlite"
FAISS_INDEX_PATH = DATA_DIR / "vector_store.faiss"
FAISS_UIDS_PATH = DATA_DIR / "vector_store.uids.json"

# Debug output
st.write("PARQUET_PATH:", PARQUET_PATH)
st.write("DB_PATH:", DB_PATH)
st.write("PIPELINE_DIR:", PIPELINE_DIR)

# ============================================================
#   ENV VAR CONFIG (Azure OpenAI)
# ============================================================
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
#   SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_deployment" not in st.session_state:
    st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_5  # default model


# ============================================================
#   AZURE OPENAI CLIENTS
# ============================================================
def make_embed_client():
    return AzureOpenAI(
        api_key=EMBED_KEY, azure_endpoint=EMBED_ENDPOINT, api_version="2023-05-15"
    )


def make_chat_client_4o():
    return AzureOpenAI(
        api_key=CHAT_KEY, azure_endpoint=CHAT_ENDPOINT, api_version="2025-01-01-preview"
    )


def make_chat_client_5():
    return AzureOpenAI(
        api_key=CHAT_KEY, azure_endpoint=CHAT_ENDPOINT, api_version="2025-04-01-preview"
    )


# ============================================================
#   LOADERS
# ============================================================
@st.cache_data
def load_docs_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid'")
    return df


@st.cache_resource
def load_sqlite_vectors() -> Tuple[List[str], np.ndarray]:
    if not DB_PATH.exists():
        return [], np.zeros((0, 0), dtype=np.float32)
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT uid, dim, vector FROM vectors").fetchall()
    con.close()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)

    dim = rows[0][1]
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    uids = []
    for i, (uid, _dim, blob) in enumerate(rows):
        uids.append(uid)
        mat[i] = np.frombuffer(blob, dtype=np.float32)
    return uids, mat


def have_faiss():
    return FAISS_INDEX_PATH.exists() and FAISS_UIDS_PATH.exists()


# ============================================================
#   EMBEDDING / SEARCH
# ============================================================
def embed_text(text: str) -> np.ndarray:
    client = make_embed_client()
    text = (text or "")[:TEXT_CHAR_LIMIT]
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine_search(qvec, uids, mat, top_k):
    if mat.size == 0:
        return []
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = M.dot(q)
    idx = np.argsort(-sims)[:top_k]
    return [(uids[i], float(sims[i])) for i in idx]


# ============================================================
#   SNIPPETS + CONTEXT
# ============================================================
def build_snippet(row: pd.Series) -> str:
    body = row["abstract"] or row["summary"] or row["description"] or ""
    body = str(body).strip().replace("\n", " ")
    return f"""
Title: {row['title']}
Source: {row['source']}
URL: {row['url']}
Summary: {body[:SNIPPET_CHAR_LIMIT]}
""".strip()


def assemble_context(df, ordered_uids):
    snippets = []
    metas = []
    total = 0
    for rnk, uid in enumerate(ordered_uids, start=1):
        r = df[df["uid"] == uid]
        if r.empty:
            continue
        row = r.iloc[0]
        snip = build_snippet(row)
        if total + len(snip) > MAX_CONTEXT_CHARS:
            break
        snippets.append(snip)
        metas.append(
            {
                "rank": rnk,
                "uid": uid,
                "title": row["title"],
                "url": row["url"],
                "source": row["source"],
            }
        )
        total += len(snip)
    return "\n\n---\n\n".join(snippets), metas


# ============================================================
#   CHAT COMPLETION
# ============================================================
def chat_answer(question, context, history, deployment):
    t0 = time.time()
    messages = [
        {
            "role": "system",
            "content": "You are a biomedical RAG assistant. Use context strictly.",
        }
    ]
    for m in history:
        messages.append(m)
    messages.append({"role": "user", "content": f"{question}\n\nContext:\n{context}"})

    if deployment == CHAT_DEPLOYMENT_5:
        client = make_chat_client_5()
        resp = client.responses.create(
            model=deployment, input=messages, max_output_tokens=1400
        )
        answer = resp.output_text
    else:
        client = make_chat_client_4o()
        resp = client.chat.completions.create(
            model=deployment, messages=messages, max_completion_tokens=1400
        )
        answer = resp.choices[0].message.content

    latency = int((time.time() - t0) * 1000)
    return answer, {"latency_ms": latency}


# ============================================================
#   SIDEBAR (MODEL + DATASET UPDATE)
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.radio("Model", ["GPT-5.1 (default)", "GPT-4o-mini"])
    if model_choice.startswith("GPT-5"):
        st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_5
    else:
        st.session_state["chat_deployment"] = CHAT_DEPLOYMENT_4O

    if st.button("🆕 New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.subheader("🔄 Dataset Controls")

    if st.button("Update Dataset"):
        st.write("Updating…")
        progress = st.progress(0)
        out = st.empty()

        def run_step(name, cmd, pct):
            out.write(f"**{name}** …")
            result = os.popen(cmd).read()
            out.write(f"```\n{result}\n```")
            progress.progress(pct)

        try:
            run_step("Pulling raw data", f"python {PIPELINE_DIR/'pull_all.py'}", 25)
            run_step("Normalizing", f"python {PIPELINE_DIR/'normalize_load.py'}", 50)
            run_step("Embedding", f"python {PIPELINE_DIR/'embeddings.py'}", 75)
            run_step("Health check", f"python {PIPELINE_DIR/'health_check.py'}", 100)
            st.success("Dataset updated!")
        except Exception as e:
            st.error(f"Update failed: {e}")

# ============================================================
#   MAIN APP
# ============================================================
st.title("💬 Endo PoC — RAG Panel")

tab_chat, tab_history = st.tabs(["💬 Chat", "📜 History"])

# ---------------- CHAT TAB ----------------
with tab_chat:
    top_k = st.slider("Top-K Results", 1, 10, 5)

    # Render chat history
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Assistant:** {m['content']}")

    st.markdown("---")
    question = st.text_input("Your Message:")
    send = st.button("Send")

    if send and question.strip():
        question = question.strip()
        st.session_state["messages"].append({"role": "user", "content": question})

        df = load_docs_df()
        uids, mat = load_sqlite_vectors()
        qvec = embed_text(question)
        matches = cosine_search(qvec, uids, mat, top_k)
        ordered = [u for u, _ in matches]
        context, metas = assemble_context(df, ordered)

        history = st.session_state["messages"][:-1]
        answer, perf = chat_answer(
            question, context, history, st.session_state["chat_deployment"]
        )

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.success(f"Response received in {perf['latency_ms']} ms")
        st.rerun()

# ---------------- HISTORY TAB ----------------
with tab_history:
    if not LOG_DB_PATH.exists():
        st.info("No history yet.")
    else:
        con = sqlite3.connect(LOG_DB_PATH)
        rows = con.execute(
            """
            SELECT id, ts_utc, question, top_k, used_faiss, latency_ms
            FROM queries ORDER BY id DESC LIMIT 50
        """
        ).fetchall()
        con.close()

        st.write(rows if rows else "No logs.")
