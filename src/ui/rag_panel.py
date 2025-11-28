import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load .env AFTER all imports (fixes Ruff E402)
load_dotenv()


# ---------- Paths ----------
PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
LOG_DB_PATH = "./data/rag_logs.sqlite"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/vector_store.faiss")
FAISS_UIDS_PATH = os.getenv("FAISS_UIDS_PATH", "./data/vector_store.uids.json")

# Ensure package path works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    pass
except Exception:
    pass


# ---------- Config ----------
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

CHAT_DEPLOYMENT_4O = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_4O", "")
CHAT_DEPLOYMENT_5 = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_5", "")

TEXT_CHAR_LIMIT = int(os.getenv("RAG_TEXT_CHAR_LIMIT", "8000"))
SNIPPET_CHAR_LIMIT = int(os.getenv("RAG_SNIPPET_CHAR_LIMIT", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

print("DEBUG: Using endpoint:", EMBED_ENDPOINT)


# ---------- Clients ----------
def make_embed_client():
    """Embeddings client using correct Azure API version."""
    return AzureOpenAI(
        api_key=EMBED_KEY, azure_endpoint=EMBED_ENDPOINT, api_version="2023-05-15"
    )


def make_chat_client_4o():
    """GPT-4o-mini client using chat/completions."""
    return AzureOpenAI(
        api_key=EMBED_KEY,
        azure_endpoint=EMBED_ENDPOINT,
        api_version="2025-01-01-preview",
    )


def make_chat_client_5():
    """GPT-5.1 client using responses API."""
    return AzureOpenAI(
        api_key=EMBED_KEY,
        azure_endpoint=EMBED_ENDPOINT,
        api_version="2025-04-01-preview",
    )


# ---------- Data helpers ----------
@st.cache_data(show_spinner=False)
def load_docs_df():
    df = pd.read_parquet(PARQUET_PATH).fillna("")
    if "uid" not in df.columns:
        raise RuntimeError("documents.parquet missing 'uid'")

    for col in ("title", "abstract", "summary", "description", "url", "source"):
        if col not in df.columns:
            df[col] = ""

    return df


@st.cache_resource(show_spinner=False)
def load_sqlite_vectors():
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


def have_faiss():
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_UIDS_PATH)


def embed_text(q: str):
    client = make_embed_client()
    q = q.strip()[:TEXT_CHAR_LIMIT]
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=q)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def cosine_search(query_vec, uids, mat, top_k):
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
    with open(FAISS_UIDS_PATH, "r", encoding="utf-8") as f:
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
    return (
        f"Title: {row['title']}\n"
        f"Source: {row['source']}\n"
        f"URL: {row['url']}\n"
        f"Summary: {str(body).strip()[:SNIPPET_CHAR_LIMIT]}"
    )


def assemble_context(df, ordered_uids):
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


# ---------- Chat Logic ----------
def chat_answer(question: str, context: str):

    # GPT-5.1 — uses responses API
    if CHAT_DEPLOYMENT == CHAT_DEPLOYMENT_5:
        client = make_chat_client_5()
        resp = client.responses.create(
            model=CHAT_DEPLOYMENT,
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant for endometriosis.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
            max_output_tokens=700,
        )
        answer = resp.output_text

    # GPT-4o-mini — uses chat/completions
    else:
        client = make_chat_client_4o()
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant for endometriosis.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
            max_completion_tokens=700,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content

    return answer, {}


# ---------- UI ----------
st.set_page_config(page_title="Endo PoC — RAG Panel", layout="wide")

blue = "#1F6FEB"  # Azure blue
green = "#2ECC71"  # Soft biotech green
bg_light = "#F5FAFD"  # Pale blue/green background
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
            color: white;
        }}
        /* Title */
        .st-emotion-cache-10trblm {{
            color: {text_dark};
        }}
        /* Radio buttons */
        .st-emotion-cache-1m1v06i label {{
            color: {text_dark};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Endo PoC — RAG Panel")

tab1, tab2 = st.tabs(["🔍 Ask", "🕓 History"])

with st.sidebar:
    st.subheader("Model Settings")
    st.markdown("*(Endpoints hidden for security)*")

    st.markdown("**Embeddings Endpoint:** 🔒 Hidden")
    st.markdown("**Chat Endpoint:** 🔒 Hidden")

    # Model selector
    model_choice = st.radio("Model", ["GPT-5.1 (default)", "GPT-4o-mini"], index=0)

if model_choice == "GPT-5.1 (default)":
    CHAT_DEPLOYMENT = CHAT_DEPLOYMENT_5
else:
    CHAT_DEPLOYMENT = CHAT_DEPLOYMENT_4O

with st.sidebar:
    st.subheader("Dataset Controls")

    if st.button("🔄 Update Dataset", type="primary"):
        st.write("### Updating Dataset…")
        progress = st.progress(0)
        output = st.empty()

        def run_step(step_name, cmd, pct):
            output.write(f"**{step_name}** running…")
            result = os.popen(cmd).read()
            output.write(f"```\n{result}\n```")
            progress.progress(pct)

        try:
            # 1. Pull
            run_step("Pulling raw data", "python src/pipelines/pull_all.py", 25)

            # 2. Normalize
            run_step(
                "Normalizing + Loading", "python src/pipelines/normalize_load.py", 50
            )

            # 3. Embeddings
            run_step("Generating Embeddings", "python src/pipelines/embeddings.py", 75)

            # 4. Health check
            run_step("Health Check", "python src/pipelines/health_check.py", 100)

            st.success("Dataset successfully updated!")

        except Exception as e:
            st.error(f"❌ Update failed: {e}")


# ---------- TAB 1 ----------
with tab1:
    top_k = st.slider("Top-K results", 1, 10, 5)
    use_faiss_pref = st.checkbox("Prefer FAISS (if available)", True)

    query = st.text_input("Ask a question")
    go = st.button("Run Query", use_container_width=True)

    if go and query.strip():
        try:
            df = load_docs_df()
            uids, mat = load_sqlite_vectors()

            qvec = embed_text(query)

            if use_faiss_pref and have_faiss():
                matches = faiss_search(qvec, top_k)
                ordered = [u for u, _ in matches]
            else:
                matches = cosine_search(qvec, uids, mat, top_k)
                ordered = [u for u, _ in matches]

            if not ordered:
                st.warning("No matches found.")
                st.stop()

            context, metas = assemble_context(df, ordered)
            answer, _ = chat_answer(query, context)

            st.subheader("Answer")
            st.write(answer)

            st.divider()
            st.subheader("Sources")

            for m in metas:
                with st.expander(f"{m['rank']}. {m['title'][:100]}"):
                    row = df.loc[df["uid"] == m["uid"]].iloc[0]
                    st.markdown(
                        f"**UID:** {m['uid']} | **Source:** {m['source']} | **URL:** {m['url']}"
                    )
                    st.write(
                        (row["abstract"] or row["summary"] or row["description"] or "")[
                            :1200
                        ]
                    )

        except Exception as e:
            st.error(f"Error: {e}")


# ---------- TAB 2 ----------
with tab2:
    if not os.path.exists(LOG_DB_PATH):
        st.info("No history yet.")
    else:
        con = sqlite3.connect(LOG_DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT id,ts_utc,question,top_k,used_faiss,latency_ms,total_tokens FROM queries ORDER BY id DESC LIMIT 50"
        )
        rows = cur.fetchall()
        con.close()

        if not rows:
            st.info("Log empty.")
        else:
            st.subheader("Recent Queries")
            for r in rows:
                qid, ts, q, topk, faiss, lat, toks = r
                st.markdown(f"**{qid}.** {q[:100]}")
                st.caption(f"{ts} | Top {topk} | {'FAISS' if faiss else 'cosine'}")
