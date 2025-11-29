import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# ---------- Paths / Constants ----------
PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/documents.parquet")
DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store.sqlite")
LOG_DB_PATH = "./data/rag_logs.sqlite"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/vector_store.faiss")
FAISS_UIDS_PATH = os.getenv("FAISS_UIDS_PATH", "./data/vector_store.uids.json")

# Azure OpenAI
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or EMBED_ENDPOINT
CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY") or EMBED_KEY

# Deployments from .env
CHAT_DEPLOYMENT_4O = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_4O", "gpt-4o-mini")
CHAT_DEPLOYMENT_5 = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_5", "gpt-5.1-chat")

TEXT_CHAR_LIMIT = int(os.getenv("RAG_TEXT_CHAR_LIMIT", "8000"))
SNIPPET_CHAR_LIMIT = int(os.getenv("RAG_SNIPPET_CHAR_LIMIT", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

# ---------- Ensure imports to project root ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.common.rag_log import log_query
except Exception:
    from common.rag_log import log_query

# ---------- Session State ----------
if "messages" not in st.session_state:
    # Simple chat history for UI (no context injected here)
    st.session_state["messages"] = (
        []
    )  # list[{"role": "user"|"assistant", "content": str}]


# ---------- Clients ----------
def make_embed_client() -> AzureOpenAI:
    """Client for embeddings (uses embeddings API)."""
    return AzureOpenAI(
        api_key=EMBED_KEY,
        azure_endpoint=EMBED_ENDPOINT,
        api_version="2023-05-15",  # from Azure portal for embeddings
    )


def make_chat_client_4o() -> AzureOpenAI:
    """Client for GPT-4o-mini (chat/completions)."""
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version="2025-01-01-preview",  # from Azure portal for gpt-4o-mini
    )


def make_chat_client_5() -> AzureOpenAI:
    """Client for GPT-5.1-chat (responses API)."""
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version="2025-04-01-preview",  # from Azure portal for gpt-5.1-chat
    )


# ---------- Data Helpers ----------
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
    uids: List[str] = []
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
    return f"Title: {title}\nSource: {row['source']}\nURL: {row['url']}\nSummary: {body[:SNIPPET_CHAR_LIMIT]}"


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


# ---------- Chat with Multi-turn + RAG-on-latest ----------
def chat_answer(
    question: str,
    context: str,
    history: List[Dict[str, str]],
    chat_deployment: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Multi-turn chat:
    - history: previous user/assistant messages (no context injected)
    - question: latest user question (no context in stored history)
    - context: RAG context built ONLY from this latest question
    """
    t0 = time.time()

    # Build messages for the model
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

    # Add prior turns (conversation memory)
    for m in history:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    # Add the latest user turn with context injected
    messages.append(
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        }
    )

    usage_dict: Dict[str, Any] = {}

    # GPT-5.1-chat via responses API
    if chat_deployment == CHAT_DEPLOYMENT_5:
        client = make_chat_client_5()
        resp = client.responses.create(
            model=chat_deployment,
            input=messages,
            max_output_tokens=1400,  # doubled from 700
        )
        # SDK exposes a convenience property for text
        answer = resp.output_text.strip() if hasattr(resp, "output_text") else str(resp)
        # usage currently not standardized on responses API; leave mostly empty
        usage_dict = {}
    else:
        # GPT-4o-mini via chat/completions
        client = make_chat_client_4o()
        resp = client.chat.completions.create(
            model=chat_deployment,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1400,  # doubled from 700
        )
        answer = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage is not None:
            usage_dict = {"total_tokens": getattr(usage, "total_tokens", None)}

    latency = int((time.time() - t0) * 1000)
    return answer, {"latency_ms": latency, "usage": usage_dict}


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Endo PoC — RAG Panel", layout="wide")

# ---- Custom Blue/Green Theme ----
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Endo PoC — RAG Panel")

tab1, tab2 = st.tabs(["💬 Chat", "🕓 History"])

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Model Settings")
    st.markdown("*(Endpoints hidden for security)*")
    st.markdown("**Embeddings Endpoint:** 🔒 Hidden")
    st.markdown("**Chat Endpoint:** 🔒 Hidden")

    # Model selector
    MODEL_CHOICE = st.radio(
        "Model",
        ["GPT-5.1 (default)", "GPT-4o-mini"],
        index=0,
    )

    if MODEL_CHOICE == "GPT-5.1 (default)":
        CHAT_DEPLOYMENT = CHAT_DEPLOYMENT_5
    else:
        CHAT_DEPLOYMENT = CHAT_DEPLOYMENT_4O

    if st.button("🆕 New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.subheader("Dataset Controls")

    if st.button("🔄 Update Dataset"):
        st.write("### Updating Dataset…")
        progress = st.progress(0)
        output = st.empty()

        def run_step(step_name: str, cmd: str, pct: int) -> None:
            output.write(f"**{step_name}** running…")
            result = os.popen(cmd).read()
            output.write(f"```\n{result}\n```")
            progress.progress(pct)

        try:
            run_step("Pulling raw data", "python src/pipelines/pull_all.py", 25)
            run_step(
                "Normalizing + Loading", "python src/pipelines/normalize_load.py", 50
            )
            run_step("Generating Embeddings", "python src/pipelines/embeddings.py", 75)
            run_step("Health Check", "python src/pipelines/health_check.py", 100)
            st.success("Dataset successfully updated!")
        except Exception as e:
            st.error(f"❌ Update failed: {e}")


# ---------- TAB 1 — CHAT (multi-turn with RAG on latest) ----------
with tab1:
    top_k = st.slider("Top-K results", 1, 10, 5)
    use_faiss_pref = st.checkbox(
        "Prefer FAISS (if available)",
        True,
        help="If enabled and FAISS index exists, use ANN search; otherwise cosine search over SQLite vectors.",
    )

    st.markdown("---")
    st.subheader("Conversation")

    # Render chat history
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Assistant:** {m['content']}")

    # Chat input
    # ---- Chat Input (Always at the Bottom) ----
st.markdown("---")

user_input = st.text_input(
    "Your message",
    placeholder="Ask me something...",
    key="chat_input_field",
)

send = st.button("Send", use_container_width=True)

if send and user_input.strip():
    question = user_input.strip()
    st.session_state["messages"].append({"role": "user", "content": question})

    try:
        # (same logic as before — RAG fetch, context building, chat_answer call)
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

        context, metas = assemble_context(df, ordered)

        history = st.session_state["messages"][:-1]

        answer, perf = chat_answer(
            question=question,
            context=context,
            history=history,
            chat_deployment=CHAT_DEPLOYMENT,
        )

        st.session_state["messages"].append({"role": "assistant", "content": answer})

        st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")

    send = st.button("Send", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()

        try:
            # Append user message to history
            st.session_state["messages"].append({"role": "user", "content": question})

            # RAG: build context **only for this latest question**
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
                st.warning("No matches found.")
                st.stop()

            context, metas = assemble_context(df, ordered)

            # history WITHOUT the new user message as context is injected separately
            history = st.session_state["messages"][:-1]

            answer, perf = chat_answer(
                question=question,
                context=context,
                history=history,
                chat_deployment=CHAT_DEPLOYMENT,
            )

            # Append assistant message to history
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )

            # Display answer
            st.subheader("Answer")
            st.write(answer)
            st.caption(
                f"Latency {perf.get('latency_ms')} ms, Tokens {perf.get('usage', {}).get('total_tokens')}"
            )

            st.divider()
            st.subheader("Sources")
            score_map = {u: float(s) for u, s in matches}
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
                question,
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
