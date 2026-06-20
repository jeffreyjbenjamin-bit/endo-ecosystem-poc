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
import subprocess
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

# Email briefing helper
try:
    from src.common.email_briefing import (
        send_briefing,
        smtp_configured,
        load_smtp_config,
    )
except Exception:

    def send_briefing(to_addr, subject, title, meta, briefing_md):
        raise RuntimeError("email_briefing not available")

    def smtp_configured():
        return False

    def load_smtp_config():
        return None


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
# Styling
# ============================================================
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #1F6FEB;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            border: none;
        }
        .stButton>button:hover { background-color: #2ECC71; }
        div[data-testid="stStatusWidget"] { display: none; }

        /* Article cards */
        .endo-card {
            background: #ffffff;
            border: 1px solid #e4e9f0;
            border-radius: 10px;
            padding: 20px 22px 16px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: box-shadow 0.15s ease;
        }
        .endo-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }

        .endo-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 2px 9px;
            border-radius: 4px;
            margin-right: 8px;
            vertical-align: middle;
        }
        .badge-pubmed           { background:#d6eaf8; color:#1a5276; }
        .badge-openalex         { background:#d5f5e3; color:#1e8449; }
        .badge-ctgov            { background:#fdebd0; color:#935116; }
        .badge-semantic_scholar { background:#e8daef; color:#6c3483; }
        .badge-crossref         { background:#fdfefe; color:#555; border:1px solid #ccc; }
        .badge-web_search       { background:#fdedec; color:#922b21; }
        .badge-nih_reporter     { background:#eaf4fb; color:#1f618d; }
        .badge-biorxiv          { background:#fef9e7; color:#9a7d0a; }
        .badge-medrxiv          { background:#fef5e7; color:#b7770d; }
        .badge-default          { background:#f2f3f4; color:#444; }

        .endo-title { font-size:1.08rem; font-weight:700; color:#0d2137;
                      line-height:1.4; margin:6px 0 4px 0; }
        .endo-title a { color:#0d2137; text-decoration:none; }
        .endo-title a:hover { color:#1a6eb5; text-decoration:underline; }
        .endo-meta    { font-size:0.76rem; color:#7f8c8d; margin-bottom:8px; }
        .endo-excerpt { font-size:0.88rem; color:#2c3e50; line-height:1.6;
                        margin-bottom:10px; display:-webkit-box;
                        -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden; }
        .endo-tags { margin-top:8px; }
        .endo-tag  { display:inline-block; font-size:0.68rem; background:#eaf0fb;
                     color:#2e4482; border-radius:3px; padding:2px 7px;
                     margin-right:5px; margin-bottom:3px; }
        .endo-readmore { font-size:0.78rem; font-weight:600; color:#1a6eb5;
                         text-decoration:none; }
        .endo-readmore:hover { text-decoration:underline; }
        .endo-quality { float:right; font-size:0.70rem; color:#aab; margin-top:2px; }

        /* Briefing panel */
        .brief-panel { background:#0d2137; border-radius:10px;
                       overflow:hidden; box-shadow:-3px 0 18px rgba(0,0,0,0.12); }
        .brief-panel-header { background:#0d2137; color:#ffffff;
                              padding:14px 18px 10px 18px; font-size:0.72rem;
                              font-weight:700; letter-spacing:0.08em; text-transform:uppercase; }
        .brief-panel-run { color:#7fa8cc; font-size:0.68rem; font-weight:400;
                           letter-spacing:0; text-transform:none; margin-top:2px; }
        .brief-panel-body { background:#f8faff; padding:20px 18px 24px 18px;
                            font-size:0.84rem; line-height:1.72; color:#1a2535;
                            border-radius:0 0 10px 10px; }
        .brief-panel-body h2 { font-size:0.82rem; font-weight:700; color:#0d2137;
                               border-bottom:1.5px solid #dde4ef; padding-bottom:3px;
                               margin-top:18px; margin-bottom:8px;
                               text-transform:uppercase; letter-spacing:0.04em; }

        /* Sticky right panel */
        div[data-testid="column"]:last-of-type > div:first-child {
            position: sticky; top: 60px;
        }
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

RAG_BRIEF_DB = str(REPO_ROOT / "data" / "rag_briefs.sqlite")
COMPARE_DB = str(REPO_ROOT / "data" / "comparisons.sqlite")


# ── RAG briefing persistence ──────────────────────────────────


def _ensure_rag_brief_db(db_path: str = RAG_BRIEF_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_briefs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts     TEXT NOT NULL,
            filter     TEXT,
            n_sources  INTEGER,
            latency_ms INTEGER,
            briefing_text TEXT
        )
    """
    )
    conn.commit()
    return conn


def save_rag_brief(
    filter_label: str, n_sources: int, latency_ms: int, briefing_text: str
) -> int:
    conn = _ensure_rag_brief_db()
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO rag_briefs (run_ts, filter, n_sources, latency_ms, briefing_text) VALUES (?,?,?,?,?)",
        (ts, filter_label, n_sources, latency_ms, briefing_text),
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def load_rag_briefs(limit: int = 20) -> List[Dict[str, Any]]:
    if not Path(RAG_BRIEF_DB).exists():
        return []
    conn = sqlite3.connect(RAG_BRIEF_DB)
    cur = conn.execute(
        "SELECT id, run_ts, filter, n_sources, latency_ms, briefing_text "
        "FROM rag_briefs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


# ── Comparison persistence ────────────────────────────────────


def _ensure_compare_db(db_path: str = COMPARE_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comparisons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts          TEXT NOT NULL,
            ai_search_run_id  INTEGER,
            rag_brief_run_id  INTEGER,
            scores_json     TEXT,
            overlap_json    TEXT,
            recommendation  TEXT,
            full_eval_text  TEXT
        )
    """
    )
    conn.commit()
    return conn


def save_comparison(
    ai_run_id: int,
    rag_run_id: int,
    scores: Dict[str, Any],
    overlap: Dict[str, Any],
    recommendation: str,
    full_eval: str,
) -> int:
    conn = _ensure_compare_db()
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO comparisons
           (run_ts, ai_search_run_id, rag_brief_run_id, scores_json, overlap_json, recommendation, full_eval_text)
           VALUES (?,?,?,?,?,?,?)""",
        (
            ts,
            ai_run_id,
            rag_run_id,
            json.dumps(scores),
            json.dumps(overlap),
            recommendation,
            full_eval,
        ),
    )
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


# ── Newsletter persistence ────────────────────────────────────

NEWSLETTER_DB = str(REPO_ROOT / "data" / "newsletters.sqlite")


def _ensure_newsletter_db(db_path: str = NEWSLETTER_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS newsletters (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts   TEXT NOT NULL,
            updated_ts   TEXT NOT NULL,
            title        TEXT,
            coverage_period TEXT,
            content_md   TEXT,
            sources_json TEXT,
            config_json  TEXT
        )
    """
    )
    conn.commit()
    return conn


def save_newsletter(
    title: str, coverage_period: str, content_md: str, sources: List[Dict], config: Dict
) -> int:
    conn = _ensure_newsletter_db()
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO newsletters
           (created_ts, updated_ts, title, coverage_period, content_md, sources_json, config_json)
           VALUES (?,?,?,?,?,?,?)""",
        (
            ts,
            ts,
            title,
            coverage_period,
            content_md,
            json.dumps(sources),
            json.dumps(config),
        ),
    )
    conn.commit()
    nid = cur.lastrowid
    conn.close()
    return nid


def update_newsletter(nid: int, content_md: str) -> None:
    conn = _ensure_newsletter_db()
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE newsletters SET content_md=?, updated_ts=? WHERE id=?",
        (content_md, ts, nid),
    )
    conn.commit()
    conn.close()


def load_newsletters(limit: int = 20) -> List[Dict[str, Any]]:
    if not Path(NEWSLETTER_DB).exists():
        return []
    conn = sqlite3.connect(NEWSLETTER_DB)
    cur = conn.execute(
        "SELECT id, created_ts, updated_ts, title, coverage_period, content_md "
        "FROM newsletters ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def _extract_highlights(
    briefing_text: str, source_type: str, run_id: int, run_ts: str, filter_label: str
) -> List[Dict[str, Any]]:
    """Parse a briefing into selectable highlight items (bullets + paragraphs)."""
    items: List[Dict[str, Any]] = []
    current_section = "General"
    idx = 0
    for raw_line in briefing_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current_section = line[3:].strip()
        elif line.startswith("# "):
            current_section = line[2:].strip()
        elif line.startswith(("- ", "* ", "• ")):
            content = line[2:].strip()
            if len(content) > 15:
                items.append(
                    {
                        "id": f"{source_type}_{run_id}_{idx}",
                        "source_type": source_type,
                        "run_id": run_id,
                        "run_ts": run_ts[:16].replace("T", " "),
                        "filter_label": filter_label,
                        "section": current_section,
                        "content": content,
                        "item_type": "bullet",
                    }
                )
                idx += 1
        elif (
            line
            and not line.startswith(("#", "---", "["))
            and len(line) > 40
            and current_section not in ("References",)
        ):
            items.append(
                {
                    "id": f"{source_type}_{run_id}_{idx}",
                    "source_type": source_type,
                    "run_id": run_id,
                    "run_ts": run_ts[:16].replace("T", " "),
                    "filter_label": filter_label,
                    "section": current_section,
                    "content": line,
                    "item_type": "paragraph",
                }
            )
            idx += 1
    return items


def _extract_references(briefing_text: str) -> List[str]:
    """Pull numbered reference lines from the ## References section."""
    refs: List[str] = []
    in_refs = False
    for line in briefing_text.splitlines():
        if line.startswith("## References"):
            in_refs = True
            continue
        if in_refs:
            if line.startswith("## "):
                break
            stripped = line.strip()
            if stripped:
                refs.append(stripped)
    return refs


# ── Newsletter generation ─────────────────────────────────────

_NL_SYSTEM = """\
You are a professional medical intelligence editor producing a newsletter for life sciences professionals focused on endometriosis.

Your output must be clean, publication-ready Markdown. Follow the structure exactly as provided in the user message.

EDITORIAL RULES:
- Write in confident, precise, third-person editorial prose.
- Every factual claim must cite at least one source using [N] inline.
- Do not invent new findings. Only synthesize what is in the provided highlights.
- The Lead Story should be the single most significant finding.
- Group Key Developments by category — only include categories with selected findings.
- Why It Matters must explain clinical or strategic significance, not just restate findings.
- Open Questions must be specific and investigable, not generic.
- Watchlist items must be concrete: named trials, drugs, companies, or regulatory actions.
- Evidence Gaps must be honest about weak sourcing, missing categories, or uncertain claims.
- Preserve all [N] citation numbers and URLs exactly as they appear in the source highlights.
- Do not use em dashes excessively. Do not use filler phrases."""

_NL_REVISE_SYSTEM = """\
You are editing a professional endometriosis intelligence newsletter.
Apply the revision instruction precisely. Preserve all [N] source citations and URLs.
Do not add new factual claims not grounded in the existing text.
Return only the revised content with no preamble or explanation."""


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
st.title("Evidence Gap")

tab_search, tab_rag, tab_compare, tab_newsletter, tab_qa = st.tabs(
    [
        "🔍 AI Article Search & Summary",
        "📚 RAG Article Presentation",
        "⚖️ Output Comparison",
        "📰 Newsletter Builder",
        "💬 Q & A",
    ]
)

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

    if st.button("🆕 New Q&A Session"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.subheader("Dataset Controls")

    if st.button("🔄 Update Dataset", use_container_width=True):
        _pipeline_steps = [
            (
                "Pulling from data sources",
                [sys.executable, "-m", "src.pipelines.pull_all"],
            ),
            (
                "Normalizing & loading documents",
                [sys.executable, "-m", "src.pipelines.normalize_load"],
            ),
            (
                "Building vector embeddings",
                [sys.executable, "-m", "src.pipelines.embeddings"],
            ),
        ]

        with st.status("Updating dataset…", expanded=True) as _upd_status:
            _had_error = False
            _n_steps = len(_pipeline_steps)
            _pbar = st.progress(0, text="Starting…")
            _log = st.empty()

            for _i, (_step_label, _cmd) in enumerate(_pipeline_steps):
                _pbar.progress(
                    _i / _n_steps,
                    text=f"Step {_i + 1}/{_n_steps}: {_step_label}…",
                )
                _proc = subprocess.run(
                    _cmd,
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                if _proc.returncode == 0:
                    _out_lines = (_proc.stdout or "").strip().splitlines()
                    _log.caption("\n".join(_out_lines[-5:]) if _out_lines else "Done")
                else:
                    _pbar.progress(
                        (_i + 1) / _n_steps,
                        text=f"Failed at: {_step_label}",
                    )
                    _err_text = (
                        _proc.stderr or _proc.stdout or "Unknown error"
                    ).strip()
                    st.code(_err_text[-1000:], language=None)
                    _had_error = True
                    break

            if _had_error:
                _upd_status.update(
                    label="Update failed — see details above",
                    state="error",
                    expanded=True,
                )
            else:
                _pbar.progress(1.0, text="Complete")
                _log.empty()
                _upd_status.update(
                    label="Dataset updated successfully",
                    state="complete",
                    expanded=False,
                )
                st.cache_data.clear()
                st.cache_resource.clear()

    st.markdown("---")
    st.subheader("Email Settings")

    _smtp_cfg = load_smtp_config()
    if _smtp_cfg is None:
        st.warning("SMTP not configured.")
        st.caption(
            "Add `EMAIL_SMTP_HOST`, `EMAIL_FROM`, and `EMAIL_PASSWORD` to `.env`."
        )
    else:
        st.markdown(
            f"**Host:** `{_smtp_cfg.host}:{_smtp_cfg.port}`  \n"
            f"**From:** `{_smtp_cfg.from_addr}`  \n"
            f"**Password:** `{'*' * max(0, len(_smtp_cfg.password) - 4)}"
            f"{_smtp_cfg.password[-4:] if len(_smtp_cfg.password) >= 4 else '****'}`"
        )
        _test_to = st.text_input(
            "Test recipient",
            value=_smtp_cfg.default_to,
            key="smtp_test_to",
            placeholder="you@example.com",
        )
        if st.button("Send Test Email", use_container_width=True):
            if not _test_to.strip():
                st.error("Enter a recipient address.")
            else:
                with st.spinner("Sending…"):
                    try:
                        send_briefing(
                            to_addr=_test_to.strip(),
                            subject="Evidence Gap — SMTP Test",
                            title="SMTP Connection Test",
                            meta=f"Sent from Evidence Gap · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                            briefing_md=(
                                "This is a test email confirming that your SMTP "
                                "configuration is working correctly.\n\n"
                                "**Host:** " + _smtp_cfg.host + "  \n"
                                "**Port:** " + str(_smtp_cfg.port) + "  \n"
                                "**From:** " + _smtp_cfg.from_addr
                            ),
                        )
                        st.success(f"Test email sent to {_test_to.strip()}")
                    except Exception as _smtp_exc:
                        st.error(f"Failed: {_smtp_exc}")

# ============================================================
# Tab: Output Comparison
# ============================================================
with tab_compare:
    st.markdown(
        """
        <div style="margin-bottom:18px">
          <span style="font-size:22px;font-weight:700;color:#e8f4fd">Output Comparison</span><br>
          <span style="font-size:13px;color:#7fa8cc">
            Evaluate and compare the most recent AI Search briefing vs. the most recent
            RAG briefing using a structured quality rubric.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from src.pipelines.ai_search import (
            load_recent_runs as _cmp_load_ai_runs,
            AI_SEARCH_DB as _CMP_AI_DB,
            CHAT_DEPLOYMENT_5 as _CMP_DEPLOY,
        )

        _cmp_imports_ok = True
    except Exception as _cmp_imp_err:
        st.error(f"Could not load ai_search pipeline: {_cmp_imp_err}")
        _cmp_imports_ok = False

    if _cmp_imports_ok:
        _ai_runs = _cmp_load_ai_runs(_CMP_AI_DB, limit=20)
        _rag_runs = load_rag_briefs(limit=20)

        if not _ai_runs:
            st.info(
                "No AI Search runs found. Run a search on the AI Article Search & Summary tab first."
            )
        elif not _rag_runs:
            st.info(
                "No RAG briefings found. Generate a briefing on the RAG Article Presentation tab first."
            )
        else:
            # ── Run selectors ─────────────────────────────────
            _sel_l, _sel_r = st.columns(2, gap="large")

            with _sel_l:
                st.markdown("**AI Search & Summary run**")
                _ai_run_labels = {
                    r[
                        "id"
                    ]: f"Run #{r['id']} · {r['run_ts'][:16].replace('T',' ')} UTC · {r['topic']}"
                    for r in _ai_runs
                }
                _sel_ai_id = st.selectbox(
                    "Select run",
                    options=list(_ai_run_labels.keys()),
                    format_func=lambda x: _ai_run_labels[x],
                    key="cmp_ai_run_id",
                    label_visibility="collapsed",
                )
                _sel_ai = next(r for r in _ai_runs if r["id"] == _sel_ai_id)
                st.caption(
                    f"{_sel_ai['n_scored']} sources · {_sel_ai['model_used']} · "
                    f"{_sel_ai['latency_ms']:,} ms"
                )

            with _sel_r:
                st.markdown("**RAG Article Presentation run**")
                _rag_run_labels = {
                    r[
                        "id"
                    ]: f"Run #{r['id']} · {r['run_ts'][:16].replace('T',' ')} UTC · Filter: {r['filter']}"
                    for r in _rag_runs
                }
                _sel_rag_id = st.selectbox(
                    "Select run",
                    options=list(_rag_run_labels.keys()),
                    format_func=lambda x: _rag_run_labels[x],
                    key="cmp_rag_run_id",
                    label_visibility="collapsed",
                )
                _sel_rag = next(r for r in _rag_runs if r["id"] == _sel_rag_id)
                st.caption(
                    f"{_sel_rag['n_sources']} sources · {_sel_rag['latency_ms']:,} ms"
                )

            st.markdown("---")

            if st.button(
                "⚖️ Compare Outputs", type="primary", use_container_width=False
            ):
                _ai_text = (_sel_ai.get("briefing_text") or "").strip()
                _rag_text = (_sel_rag.get("briefing_text") or "").strip()

                if not _ai_text or not _rag_text:
                    st.error("One or both selected runs have no briefing text.")
                else:
                    _cmp_pbar = st.progress(0, text="Preparing evaluation…")

                    # Build evaluation prompt
                    _eval_prompt = f"""You are evaluating two intelligence briefing outputs about Endometriosis.

BRIEFING A — AI Search & Summary (live external web search grounding):
Topic: {_sel_ai.get('topic', 'endometriosis')}
Date range: last {_sel_ai.get('date_range_days', '?')} days
Sources used: {_sel_ai.get('n_scored', '?')}
---
{_ai_text[:4000]}

BRIEFING B — RAG Search & Summary (existing uploaded/internal dataset):
Filter: {_sel_rag.get('filter', 'All articles')}
Sources used: {_sel_rag.get('n_sources', '?')}
---
{_rag_text[:4000]}

Your task is to compare the quality, usefulness, evidence strength, and professional readiness of both outputs.

SCORING SCALE — apply strictly, do not over-score:
  5 = Excellent
  4 = Strong
  3 = Adequate
  2 = Weak
  1 = Poor

EVALUATION RULES:
- Do not reward longer output unless it is more useful.
- Penalize unsupported claims.
- Penalize missing source links or vague sourcing.
- Penalize stale information when the task requires recent developments.
- Reward clear "why it matters" analysis.
- Reward explicit open questions and watchlist items.
- Reward clear separation of research, trials, regulatory, funding, pharma/biotech, diagnostics, and market activity.
- Reward professional intelligence-style formatting.
- Be fair to the intended purpose of each method:
    AI Search should be better at recent external developments.
    RAG should be better at interpreting the known corpus.
- Do not assume either method is better by default.

RUBRIC DIMENSIONS (score each 1–5):
1. relevance — Directly addresses recent endometriosis-related developments; stays on topic.
2. recency — Identifies timely, current developments; distinguishes breaking from established findings.
3. source_quality — Sources are credible, primary, appropriate for life sciences intelligence. Hierarchy: peer-reviewed > regulatory filings > clinical registries > press releases > blogs. Penalize unattributed claims.
4. coverage_breadth — Covers: Research, Diagnostics, Biomarkers, Clinical Trials, Regulatory, Funding, Pharma/Biotech, Market/News. Score 5 only if 6+ categories are meaningfully represented.
5. evidence_strength — Distinguishes RCTs/systematic reviews vs. Phase I/II vs. preprints vs. press releases vs. speculative claims. Applies epistemic labels rather than treating all findings equally.
6. analytical_value — Explains WHY findings matter, not just WHAT they are. Contextualizes within the broader endometriosis landscape. Penalize purely descriptive outputs.
7. actionability — Identifies open questions, watchlist assets, pipeline items, or concrete follow-up for a life sciences intelligence professional. Generic recommendations score no higher than 2.
8. clarity_and_format — Resembles a professional intelligence briefing (Endpoints/FirstWord style). Scannable, structured, with clear headers, appropriate length, no filler.
9. novelty — Surfaces non-obvious information meaningfully differentiated from a standard PubMed search. Penalize outputs that only summarize widely known findings.
10. confidence_and_limitations — Explicitly identifies evidence gaps, uncertainty, missing categories, or weak sourcing. Avoids overstatement. An output with uniform confidence across all claims scores no higher than 2.

COVERAGE MATRIX — for each of the 8 categories assess:
  "yes" = meaningfully covered
  "partial" = mentioned but thin
  "no" = absent
Categories: Research, Diagnostics, Biomarkers, Clinical Trials, Regulatory, Funding, Pharma/Biotech, Market/News

Respond in this exact JSON format with no prose or markdown fences:
{{
  "executive_comparison": {{
    "winner": "A|B|TIE",
    "winner_label": "AI Search & Summary|RAG Article Presentation|Tie",
    "explanation": "3-5 sentence explanation of the overall result.",
    "best_use_case_A": "1-2 sentences on where AI Search performs best.",
    "best_use_case_B": "1-2 sentences on where RAG performs best."
  }},
  "rubric": [
    {{"dimension": "Relevance",              "key": "relevance",                 "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Recency",                "key": "recency",                   "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Source Quality",         "key": "source_quality",            "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Coverage Breadth",       "key": "coverage_breadth",          "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Evidence Strength",      "key": "evidence_strength",         "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Analytical Value",       "key": "analytical_value",          "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Actionability",          "key": "actionability",             "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Clarity & Format",       "key": "clarity_and_format",        "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Novelty",                "key": "novelty",                   "score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}},
    {{"dimension": "Confidence / Limitations","key": "confidence_and_limitations","score_A": 0, "score_B": 0, "stronger": "A|B|=", "rationale": "..."}}
  ],
  "coverage_matrix": {{
    "Research":       {{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Diagnostics":    {{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Biomarkers":     {{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Clinical Trials":{{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Regulatory":     {{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Funding":        {{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Pharma/Biotech": {{"A": "yes|partial|no", "B": "yes|partial|no"}},
    "Market/News":    {{"A": "yes|partial|no", "B": "yes|partial|no"}}
  }},
  "overlap_and_gaps": {{
    "shared_findings": ["finding1", "finding2"],
    "only_in_A": ["finding1", "finding2"],
    "only_in_B": ["finding1", "finding2"],
    "missing_topics": ["topic1", "topic2"],
    "weak_or_unsupported": ["claim1", "claim2"]
  }},
  "evidence_and_source_quality": {{
    "A": "Assessment of A's citation quality, source types, dates, and traceability.",
    "B": "Assessment of B's citation quality, source types, dates, and traceability.",
    "comparison": "Direct comparison of the two approaches to sourcing."
  }},
  "recommendation": {{
    "rely_on": "A|B|COMBINED",
    "rationale": "Explanation of whether to use A, B, or combine both, and why."
  }},
  "improvements": {{
    "A": ["improvement1", "improvement2", "improvement3"],
    "B": ["improvement1", "improvement2", "improvement3"],
    "workflow": ["improvement1", "improvement2"]
  }}
}}"""

                    _cmp_pbar.progress(30, text="Sending to evaluator model…")

                    try:
                        _cmp_client = make_chat_client_5()
                        _cmp_response = _cmp_client.chat.completions.create(
                            model=_CMP_DEPLOY,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a senior life sciences intelligence analyst specializing in women's health and endometriosis. "
                                        "You evaluate intelligence briefings with the rigor of a medical editor at Endpoints News or FirstWord Pharma. "
                                        "You do not inflate scores, you do not reward length over substance, and you do not assume either method is superior. "
                                        "Respond with valid JSON only — no prose, no markdown fences, no explanation outside the JSON structure."
                                    ),
                                },
                                {"role": "user", "content": _eval_prompt},
                            ],
                            max_completion_tokens=3500,
                        )
                        _raw_eval = _cmp_response.choices[0].message.content or ""
                        _cmp_pbar.progress(70, text="Parsing evaluation…")

                        # Strip markdown code fences if present
                        import re as _re

                        _json_str = _re.sub(
                            r"^```(?:json)?\s*|\s*```$",
                            "",
                            _raw_eval.strip(),
                            flags=_re.MULTILINE,
                        )
                        _eval = json.loads(_json_str)

                        _cmp_pbar.progress(90, text="Saving…")
                        _rec_obj = _eval.get("recommendation", {})
                        _cmp_id = save_comparison(
                            ai_run_id=_sel_ai_id,
                            rag_run_id=_sel_rag_id,
                            scores={
                                r["key"]: {"A": r.get("score_A"), "B": r.get("score_B")}
                                for r in _eval.get("rubric", [])
                            },
                            overlap=_eval.get("overlap_and_gaps", {}),
                            recommendation=(
                                _rec_obj.get("rely_on", "")
                                if isinstance(_rec_obj, dict)
                                else str(_rec_obj)
                            ),
                            full_eval=_raw_eval,
                        )
                        _cmp_pbar.progress(100, text="Done")
                        _cmp_pbar.empty()
                        st.session_state["last_comparison"] = {
                            "eval": _eval,
                            "cmp_id": _cmp_id,
                            "ai": _sel_ai,
                            "rag": _sel_rag,
                        }

                    except json.JSONDecodeError as _je:
                        _cmp_pbar.empty()
                        st.error(
                            "Evaluator returned malformed JSON. Raw response shown below."
                        )
                        st.code(_raw_eval, language="text")
                    except Exception as _ce:
                        _cmp_pbar.empty()
                        st.error(f"Evaluation failed: {_ce}")

            # ── Display comparison results ─────────────────────
            _cmp_result = st.session_state.get("last_comparison")
            if _cmp_result:
                _eval = _cmp_result["eval"]
                _exec = _eval.get("executive_comparison", {})
                _rubric = _eval.get("rubric", [])
                _cov = _eval.get("coverage_matrix", {})
                _gaps = _eval.get("overlap_and_gaps", {})
                _esq = _eval.get("evidence_and_source_quality", {})
                _rec = _eval.get("recommendation", {})
                _impr = _eval.get("improvements", {})

                # ── 1. Executive Comparison ────────────────────
                _winner = _exec.get("winner", "")
                _winner_label = _exec.get("winner_label", _winner)
                _banner_color = {"A": "#1a3d5c", "B": "#1a3d2a", "TIE": "#3a3a1a"}.get(
                    _winner, "#2a2a2a"
                )
                _total_a = sum(r.get("score_A", 0) for r in _rubric)
                _total_b = sum(r.get("score_B", 0) for r in _rubric)

                st.markdown(
                    f"""
                    <div style="background:{_banner_color};border:1px solid #2a5a80;border-radius:8px;
                                padding:20px 24px;margin:12px 0 24px 0;">
                      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.10em;
                                  color:#7fa8cc;margin-bottom:6px;">1 &nbsp;·&nbsp; Executive Comparison</div>
                      <div style="font-size:22px;font-weight:700;color:#e8f4fd;margin-bottom:8px">
                        Overall Winner: {_winner_label}
                      </div>
                      <div style="font-size:13px;color:#c0d8ec;line-height:1.65;margin-bottom:12px">
                        {_exec.get('explanation', '')}
                      </div>
                      <div style="display:flex;gap:32px;margin-top:10px;font-size:13px">
                        <div><span style="color:#7fa8cc;font-size:11px;text-transform:uppercase">
                          Total score</span><br>
                          <strong style="color:#e8f4fd">A (AI Search):</strong>
                          <span style="color:#a8d0f0"> {_total_a}/50</span>
                          &nbsp;&nbsp;
                          <strong style="color:#e8f4fd">B (RAG Corpus):</strong>
                          <span style="color:#a8d0f0"> {_total_b}/50</span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                _ex_l, _ex_r = st.columns(2, gap="large")
                with _ex_l:
                    st.markdown(
                        "<div style='font-size:11px;text-transform:uppercase;letter-spacing:.07em;"
                        "color:#7fa8cc;margin-bottom:4px'>Best use case — AI Search & Summary</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(_exec.get("best_use_case_A", "—"))
                with _ex_r:
                    st.markdown(
                        "<div style='font-size:11px;text-transform:uppercase;letter-spacing:.07em;"
                        "color:#7fa8cc;margin-bottom:4px'>Best use case — RAG Article Presentation</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(_exec.get("best_use_case_B", "—"))

                st.markdown("---")

                # ── 2. Quality Rubric Table ────────────────────
                st.markdown(
                    "#### 2 &nbsp;·&nbsp; Quality Rubric Table"
                    "<span style='font-size:12px;color:#7fa8cc;margin-left:12px'>"
                    "1 = Poor &nbsp; 2 = Weak &nbsp; 3 = Adequate &nbsp; 4 = Strong &nbsp; 5 = Excellent"
                    "</span>",
                    unsafe_allow_html=True,
                )
                _SCORE_LABELS = {
                    1: "1 — Poor",
                    2: "2 — Weak",
                    3: "3 — Adequate",
                    4: "4 — Strong",
                    5: "5 — Excellent",
                }
                _rubric_rows = []
                for _r in _rubric:
                    _va = _r.get("score_A")
                    _vb = _r.get("score_B")
                    _rubric_rows.append(
                        {
                            "Dimension": _r.get("dimension", ""),
                            "AI Search (A)": _SCORE_LABELS.get(
                                _va, str(_va) if _va is not None else "—"
                            ),
                            "RAG Corpus (B)": _SCORE_LABELS.get(
                                _vb, str(_vb) if _vb is not None else "—"
                            ),
                            "Stronger": _r.get("stronger", ""),
                            "Rationale": _r.get("rationale", ""),
                        }
                    )
                _rubric_rows.append(
                    {
                        "Dimension": "TOTAL",
                        "AI Search (A)": f"{_total_a} / 50",
                        "RAG Corpus (B)": f"{_total_b} / 50",
                        "Stronger": (
                            "A"
                            if _total_a > _total_b
                            else ("B" if _total_b > _total_a else "=")
                        ),
                        "Rationale": "",
                    }
                )
                st.dataframe(
                    pd.DataFrame(_rubric_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("---")

                # ── 3. Coverage Matrix ─────────────────────────
                st.markdown(
                    "#### 3 &nbsp;·&nbsp; Coverage Matrix", unsafe_allow_html=True
                )
                _COV_ICON = {"yes": "✓", "partial": "~", "no": "✗"}
                _COV_COLOR = {"yes": "#2ecc71", "partial": "#f39c12", "no": "#e74c3c"}
                _cov_rows = []
                for _cat, _vals in _cov.items():
                    _a_val = (_vals.get("A") or "no").lower()
                    _b_val = (_vals.get("B") or "no").lower()
                    _cov_rows.append(
                        {
                            "Category": _cat,
                            "AI Search (A)": _COV_ICON.get(_a_val, _a_val),
                            "RAG Corpus (B)": _COV_ICON.get(_b_val, _b_val),
                        }
                    )
                if _cov_rows:
                    st.dataframe(
                        pd.DataFrame(_cov_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.caption("✓ = covered &nbsp; ~ = partial &nbsp; ✗ = absent")

                st.markdown("---")

                # ── 4. Overlap & Gap Analysis ──────────────────
                st.markdown(
                    "#### 4 &nbsp;·&nbsp; Overlap and Gap Analysis",
                    unsafe_allow_html=True,
                )
                _g1, _g2, _g3 = st.columns(3, gap="medium")
                with _g1:
                    st.markdown("**Shared findings**")
                    for _t in _gaps.get("shared_findings") or []:
                        st.markdown(f"- {_t}")
                with _g2:
                    st.markdown("**Only in AI Search (A)**")
                    for _t in _gaps.get("only_in_A") or []:
                        st.markdown(f"- {_t}")
                with _g3:
                    st.markdown("**Only in RAG Corpus (B)**")
                    for _t in _gaps.get("only_in_B") or []:
                        st.markdown(f"- {_t}")

                _g4, _g5 = st.columns(2, gap="medium")
                with _g4:
                    _missing = _gaps.get("missing_topics") or []
                    if _missing:
                        st.markdown("**Missing topics (neither method covered)**")
                        for _t in _missing:
                            st.markdown(f"- {_t}")
                with _g5:
                    _weak = _gaps.get("weak_or_unsupported") or []
                    if _weak:
                        st.warning("**Weak or unsupported findings**")
                        for _t in _weak:
                            st.markdown(f"- {_t}")

                st.markdown("---")

                # ── 5. Evidence & Source Quality Assessment ────
                st.markdown(
                    "#### 5 &nbsp;·&nbsp; Evidence and Source Quality Assessment",
                    unsafe_allow_html=True,
                )
                _eq_l, _eq_r = st.columns(2, gap="large")
                with _eq_l:
                    st.markdown("**AI Search & Summary (A)**")
                    st.markdown(_esq.get("A", "—"))
                with _eq_r:
                    st.markdown("**RAG Article Presentation (B)**")
                    st.markdown(_esq.get("B", "—"))
                if _esq.get("comparison"):
                    st.markdown(f"**Direct comparison:** {_esq['comparison']}")

                st.markdown("---")

                # ── 6. Recommendation ──────────────────────────
                st.markdown(
                    "#### 6 &nbsp;·&nbsp; Recommendation", unsafe_allow_html=True
                )
                _rely = _rec.get("rely_on", "")
                _rely_label = {
                    "A": "Rely on AI Search & Summary",
                    "B": "Rely on RAG Article Presentation",
                    "COMBINED": "Use a combined briefing from both methods",
                }.get(_rely, _rely)
                st.markdown(
                    f"""
                    <div style="background:#1e2d3d;border-left:4px solid #1a7fb5;border-radius:4px;
                                padding:14px 18px;margin:8px 0 16px 0;">
                      <div style="font-weight:700;color:#e8f4fd;margin-bottom:4px">{_rely_label}</div>
                      <div style="font-size:13px;color:#b0cce0">{_rec.get('rationale', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # ── 7. Improvement Suggestions ─────────────────
                st.markdown(
                    "#### 7 &nbsp;·&nbsp; Improvement Suggestions",
                    unsafe_allow_html=True,
                )
                _im_l, _im_m, _im_r = st.columns(3, gap="medium")
                with _im_l:
                    st.markdown("**AI Search & Summary**")
                    for _i in _impr.get("A") or []:
                        st.markdown(f"- {_i}")
                with _im_m:
                    st.markdown("**RAG Article Presentation**")
                    for _i in _impr.get("B") or []:
                        st.markdown(f"- {_i}")
                with _im_r:
                    st.markdown("**Comparison workflow**")
                    for _i in _impr.get("workflow") or []:
                        st.markdown(f"- {_i}")

                st.markdown("---")

                # ── Side-by-side briefing text ─────────────────
                st.markdown("#### Full Briefing Text")
                _txt_l, _txt_r = st.columns(2, gap="large")
                with _txt_l:
                    st.markdown(
                        f"<div style='font-size:12px;font-weight:700;color:#7fa8cc;"
                        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>"
                        f"AI Search & Summary — Run #{_cmp_result['ai']['id']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(_cmp_result["ai"].get("briefing_text", ""))
                with _txt_r:
                    st.markdown(
                        f"<div style='font-size:12px;font-weight:700;color:#7fa8cc;"
                        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>"
                        f"RAG Article Presentation — Run #{_cmp_result['rag']['id']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(_cmp_result["rag"].get("briefing_text", ""))


# ============================================================
# Tab: Newsletter Builder
# ============================================================
with tab_newsletter:

    # ── Session state init ────────────────────────────────────
    for _k, _v in [
        ("nl_draft", ""),
        ("nl_saved_id", None),
        ("nl_selected_ids", set()),
        ("nl_undo_stack", []),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Import pipeline ───────────────────────────────────────
    try:
        from src.pipelines.ai_search import (
            load_recent_runs as _nl_load_ai,
            AI_SEARCH_DB as _NL_AI_DB,
            CHAT_DEPLOYMENT_5 as _NL_DEPLOY,
            make_chat_client as _nl_make_client,
        )

        _nl_ok = True
    except Exception as _nl_err:
        st.error(f"Could not load pipeline: {_nl_err}")
        _nl_ok = False

    if _nl_ok:

        # ── Load saved runs ───────────────────────────────────
        _nl_ai_runs = _nl_load_ai(_NL_AI_DB, limit=30)
        _nl_rag_runs = load_rag_briefs(limit=30)

        # ── Header ───────────────────────────────────────────
        st.markdown(
            "<div style='font-size:22px;font-weight:700;margin-bottom:4px'>"
            "Newsletter Builder</div>"
            "<div style='font-size:13px;color:#7f8c8d;margin-bottom:20px'>"
            "Select highlights from saved runs, configure your newsletter, "
            "generate a draft, and edit it directly.</div>",
            unsafe_allow_html=True,
        )

        # ── Controls ─────────────────────────────────────────
        with st.expander("Newsletter Settings", expanded=True):
            _nl_c1, _nl_c2, _nl_c3 = st.columns([2, 2, 2], gap="medium")
            with _nl_c1:
                _nl_title = st.text_input(
                    "Newsletter title",
                    value="Endometriosis Intelligence Briefing",
                    key="nl_title_input",
                )
                _nl_period = st.text_input(
                    "Coverage period",
                    placeholder="e.g. June 2025",
                    key="nl_period_input",
                )
            with _nl_c2:
                _nl_source_sel = st.radio(
                    "Source",
                    ["Both", "AI Search only", "RAG only"],
                    horizontal=True,
                    key="nl_source_sel",
                )
                _nl_date_range = st.slider(
                    "Max run age (days)",
                    7,
                    365,
                    90,
                    key="nl_date_range",
                )
            with _nl_c3:
                _ALL_CATS = [
                    "Research",
                    "Diagnostics",
                    "Biomarkers",
                    "Clinical Trials",
                    "Regulatory",
                    "Funding",
                    "Pharma/Biotech",
                    "Market/News",
                    "General",
                    "Intelligence Summary",
                    "Key Developments",
                    "Clinical & Regulatory Highlights",
                    "Emerging Research Themes",
                    "What to Watch",
                ]
                _nl_cat_filter = st.multiselect(
                    "Section / category filter",
                    options=_ALL_CATS,
                    default=[],
                    key="nl_cat_filter",
                    placeholder="All sections (leave blank)",
                )
                _nl_custom_note = st.text_area(
                    "Editorial note (optional)",
                    placeholder="Add a note for the AI editor, e.g. tone, audience, focus…",
                    height=70,
                    key="nl_custom_note",
                )

        st.markdown("---")

        # ── Build highlight pool ──────────────────────────────
        _nl_all_items: List[Dict[str, Any]] = []
        _cutoff_dt = datetime.now(timezone.utc).timestamp() - _nl_date_range * 86400

        if _nl_source_sel in ("Both", "AI Search only"):
            for _r in _nl_ai_runs:
                try:
                    _rts = datetime.fromisoformat(
                        _r["run_ts"].replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    _rts = 0
                if _rts < _cutoff_dt:
                    continue
                _bt = _r.get("briefing_text") or ""
                if _bt:
                    _nl_all_items.extend(
                        _extract_highlights(
                            _bt,
                            "AI Search",
                            _r["id"],
                            _r["run_ts"],
                            _r.get("topic", "endometriosis"),
                        )
                    )

        if _nl_source_sel in ("Both", "RAG only"):
            for _r in _nl_rag_runs:
                try:
                    _rts = datetime.fromisoformat(
                        _r["run_ts"].replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    _rts = 0
                if _rts < _cutoff_dt:
                    continue
                _bt = _r.get("briefing_text") or ""
                if _bt:
                    _nl_all_items.extend(
                        _extract_highlights(
                            _bt,
                            "RAG",
                            _r["id"],
                            _r["run_ts"],
                            _r.get("filter", "All articles"),
                        )
                    )

        # Apply category filter
        if _nl_cat_filter:
            _nl_all_items = [
                it
                for it in _nl_all_items
                if any(f.lower() in it["section"].lower() for f in _nl_cat_filter)
            ]

        # ── Highlight picker ──────────────────────────────────
        st.markdown(
            f"#### Select Highlights &nbsp;"
            f"<span style='font-size:13px;color:#7f8c8d'>"
            f"{len(_nl_all_items)} items available — check the ones to include in your newsletter</span>",
            unsafe_allow_html=True,
        )

        if not _nl_all_items:
            st.info(
                "No highlights found. Generate an AI Search briefing or RAG briefing first, "
                "or adjust the date range / source filter above."
            )
        else:
            _sel_col, _act_col = st.columns([6, 1])
            with _act_col:
                if st.button("Select all", use_container_width=True, key="nl_sel_all"):
                    st.session_state["nl_selected_ids"] = {
                        it["id"] for it in _nl_all_items
                    }
                    st.rerun()
                if st.button("Clear all", use_container_width=True, key="nl_clr_all"):
                    st.session_state["nl_selected_ids"] = set()
                    st.rerun()

            # Group by section
            _nl_by_section: Dict[str, List[Dict]] = {}
            for _it in _nl_all_items:
                _nl_by_section.setdefault(_it["section"], []).append(_it)

            for _sec, _sec_items in _nl_by_section.items():
                with st.expander(f"{_sec} ({len(_sec_items)} items)", expanded=True):
                    _gc1, _gc2 = st.columns(2, gap="medium")
                    for _ci, _it in enumerate(_sec_items):
                        with _gc1 if _ci % 2 == 0 else _gc2:
                            _checked = _it["id"] in st.session_state["nl_selected_ids"]
                            _src_color = (
                                "#1a5276"
                                if _it["source_type"] == "AI Search"
                                else "#1e8449"
                            )
                            _src_bg = (
                                "#d6eaf8"
                                if _it["source_type"] == "AI Search"
                                else "#d5f5e3"
                            )
                            _preview = _it["content"][:180] + (
                                "…" if len(_it["content"]) > 180 else ""
                            )
                            _new_checked = st.checkbox(
                                f"{'✓ ' if _checked else ''}{_preview}",
                                value=_checked,
                                key=f"nl_chk_{_it['id']}",
                                help=f"{_it['source_type']} · Run {_it['run_id']} · {_it['run_ts']}",
                            )
                            if _new_checked != _checked:
                                if _new_checked:
                                    st.session_state["nl_selected_ids"].add(_it["id"])
                                else:
                                    st.session_state["nl_selected_ids"].discard(
                                        _it["id"]
                                    )

            _n_sel = len(st.session_state["nl_selected_ids"])
            st.markdown(
                f"<div style='margin:8px 0 4px 0;font-size:13px;color:#7f8c8d'>"
                f"<strong style='color:#0a2540'>{_n_sel}</strong> highlights selected</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Generate newsletter button ────────────────────────
        _nl_gen_btn = st.button(
            "📰 Generate Newsletter",
            type="primary",
            disabled=len(st.session_state.get("nl_selected_ids", set())) == 0,
            key="nl_generate",
        )

        if _nl_gen_btn:
            _sel_ids = st.session_state.get("nl_selected_ids", set())
            _sel_items = [it for it in _nl_all_items if it["id"] in _sel_ids]

            # Collect references from source briefings
            _ref_pool: List[str] = []
            _seen_run_ids: set = set()
            for _r in _nl_ai_runs + _nl_rag_runs:
                _bt = _r.get("briefing_text") or ""
                _rid = f"{_r.get('topic','rag')}_{_r['id']}"
                if _bt and _rid not in _seen_run_ids:
                    _ref_pool.extend(_extract_references(_bt))
                    _seen_run_ids.add(_rid)

            # Build the generation prompt
            _sel_by_section: Dict[str, List[str]] = {}
            for _it in _sel_items:
                _sel_by_section.setdefault(_it["section"], []).append(
                    f"[{_it['source_type']} · {_it['filter_label']}] {_it['content']}"
                )

            _highlights_block = "\n\n".join(
                f"### {sec}\n" + "\n".join(f"- {c}" for c in items)
                for sec, items in _sel_by_section.items()
            )

            _refs_block = (
                "\n".join(_ref_pool[:40]) if _ref_pool else "(No references extracted)"
            )

            _nl_user_msg = f"""Produce a professional endometriosis intelligence newsletter using ONLY the selected highlights below.

NEWSLETTER CONFIG:
- Title: {_nl_title}
- Coverage period: {_nl_period or 'Recent'}
- Published: {datetime.now().strftime('%B %d, %Y')}
{('- Editorial note: ' + _nl_custom_note) if _nl_custom_note else ''}

SELECTED HIGHLIGHTS:
{_highlights_block}

AVAILABLE REFERENCES (use [N] inline to cite; include cited ones in the References section):
{_refs_block}

OUTPUT STRUCTURE — use this exact Markdown structure:

# {_nl_title}
**Endometriosis Intelligence | {_nl_period or 'Recent Coverage'}**
*Published {datetime.now().strftime('%B %d, %Y')}*

---

## Executive Summary
[2-3 sentence overview of the most important developments in the selected highlights.]

---

## Lead Story
[The single most significant finding from the highlights, written as a focused 2-3 paragraph editorial. Include inline citations.]

---

## Key Developments
[Group under sub-headers only for categories that have selected findings. Each item: **bolded headline.** 1-2 sentences with [N] citation.]

---

## Why It Matters
[1-2 paragraphs explaining the clinical, scientific, or strategic significance of these findings collectively.]

---

## Open Questions
[3-5 specific, investigable questions raised by the selected findings.]

---

## Watchlist
[3-5 concrete items: named trials, drugs, companies, regulatory actions to monitor.]

---

## Evidence Gaps & Notes
[Honest assessment: what is weakly sourced, missing, uncertain, or contradicted across the selected findings.]

---

## Sources
[List every [N] cited inline, formatted as: [N] Title. Source. Date. URL]
"""

            _nl_pbar = st.progress(0, text="Generating newsletter…")
            try:
                _nl_client = _nl_make_client(_NL_DEPLOY)
                _nl_pbar.progress(20, text="Sending to model…")

                try:
                    _nl_resp = _nl_client.responses.create(
                        model=_NL_DEPLOY,
                        input=[
                            {"role": "system", "content": _NL_SYSTEM},
                            {"role": "user", "content": _nl_user_msg},
                        ],
                        max_output_tokens=4000,
                    )
                    _nl_text = (_nl_resp.output_text or "").strip()
                except Exception:
                    _nl_resp = _nl_client.chat.completions.create(
                        model=_NL_DEPLOY,
                        messages=[
                            {"role": "system", "content": _NL_SYSTEM},
                            {"role": "user", "content": _nl_user_msg},
                        ],
                        max_completion_tokens=4000,
                    )
                    _nl_text = (_nl_resp.choices[0].message.content or "").strip()

                _nl_pbar.progress(90, text="Saving draft…")
                _nl_sources = [
                    {
                        "content": it["content"],
                        "source_type": it["source_type"],
                        "section": it["section"],
                    }
                    for it in _sel_items
                ]
                _nl_sid = save_newsletter(
                    title=_nl_title,
                    coverage_period=_nl_period or "Recent",
                    content_md=_nl_text,
                    sources=_nl_sources,
                    config={
                        "custom_note": _nl_custom_note,
                        "source_sel": _nl_source_sel,
                    },
                )
                st.session_state["nl_draft"] = _nl_text
                st.session_state["nl_saved_id"] = _nl_sid
                st.session_state["nl_undo_stack"] = []
                _nl_pbar.progress(100, text="Done")
                _nl_pbar.empty()
                st.rerun()

            except Exception as _nl_gen_err:
                _nl_pbar.empty()
                st.error(f"Generation failed: {_nl_gen_err}")

        # ── Editor + Preview ──────────────────────────────────
        if st.session_state.get("nl_draft"):
            st.markdown("---")
            st.markdown("#### Draft Newsletter")

            _ed_col, _prev_col = st.columns([1, 1], gap="large")

            with _ed_col:
                st.markdown(
                    "<div style='font-size:11px;text-transform:uppercase;"
                    "letter-spacing:.07em;color:#7f8c8d;margin-bottom:4px'>Edit</div>",
                    unsafe_allow_html=True,
                )
                _edited = st.text_area(
                    "newsletter_editor",
                    value=st.session_state["nl_draft"],
                    height=900,
                    label_visibility="collapsed",
                    key="nl_editor_area",
                )
                if _edited != st.session_state["nl_draft"]:
                    st.session_state["nl_undo_stack"].append(
                        st.session_state["nl_draft"]
                    )
                    st.session_state["nl_draft"] = _edited
                    if st.session_state.get("nl_saved_id"):
                        update_newsletter(st.session_state["nl_saved_id"], _edited)

            with _prev_col:
                st.markdown(
                    "<div style='font-size:11px;text-transform:uppercase;"
                    "letter-spacing:.07em;color:#7f8c8d;margin-bottom:4px'>Preview</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='background:#fff;border:1px solid #e4e9f0;border-radius:8px;"
                    "padding:24px 28px;font-size:0.88rem;line-height:1.7;"
                    "max-height:900px;overflow-y:auto'>",
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["nl_draft"])
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            # ── AI Revision panel ─────────────────────────────
            st.markdown("#### AI Revision")

            _rev_scope_options = ["Full draft"] + [
                line[3:].strip()
                for line in st.session_state["nl_draft"].splitlines()
                if line.startswith("## ")
            ]

            _rv1, _rv2 = st.columns([3, 1], gap="medium")
            with _rv1:
                _rev_instruction = st.text_area(
                    "Revision instruction",
                    placeholder=(
                        "e.g. Make the Lead Story more concise. "
                        "Strengthen the Why It Matters section. "
                        "Add a more cautious tone to the Evidence Gaps section."
                    ),
                    height=90,
                    key="nl_rev_instruction",
                )
            with _rv2:
                _rev_scope = st.selectbox(
                    "Scope",
                    options=_rev_scope_options,
                    key="nl_rev_scope",
                )
                _rev_btn = st.button(
                    "Revise",
                    type="primary",
                    use_container_width=True,
                    key="nl_revise_btn",
                    disabled=not _rev_instruction.strip(),
                )

            if _rev_btn and _rev_instruction.strip():
                _current_draft = st.session_state["nl_draft"]
                if _rev_scope == "Full draft":
                    _rev_target = _current_draft
                    _rev_user_msg = (
                        f"REVISION INSTRUCTION:\n{_rev_instruction}\n\n"
                        f"SCOPE: Revise the full newsletter draft below.\n\n"
                        f"CURRENT DRAFT:\n{_current_draft}"
                    )
                    _replace_full = True
                else:
                    # Extract just the target section
                    _sec_lines: List[str] = []
                    _in_sec = False
                    for _sl in _current_draft.splitlines():
                        if _sl.startswith("## ") and _sl[3:].strip() == _rev_scope:
                            _in_sec = True
                            _sec_lines.append(_sl)
                        elif _in_sec:
                            if _sl.startswith("## ") and _sl[3:].strip() != _rev_scope:
                                break
                            _sec_lines.append(_sl)
                    _rev_target = "\n".join(_sec_lines)
                    _rev_user_msg = (
                        f"REVISION INSTRUCTION:\n{_rev_instruction}\n\n"
                        f"SCOPE: Revise only the '{_rev_scope}' section below. "
                        f"Return only the revised section text.\n\n"
                        f"SECTION TO REVISE:\n{_rev_target}"
                    )
                    _replace_full = False

                with st.spinner("Revising…"):
                    try:
                        _rv_client = _nl_make_client(_NL_DEPLOY)
                        try:
                            _rv_resp = _rv_client.responses.create(
                                model=_NL_DEPLOY,
                                input=[
                                    {"role": "system", "content": _NL_REVISE_SYSTEM},
                                    {"role": "user", "content": _rev_user_msg},
                                ],
                                max_output_tokens=3000,
                            )
                            _rv_text = (_rv_resp.output_text or "").strip()
                        except Exception:
                            _rv_resp = _rv_client.chat.completions.create(
                                model=_NL_DEPLOY,
                                messages=[
                                    {"role": "system", "content": _NL_REVISE_SYSTEM},
                                    {"role": "user", "content": _rev_user_msg},
                                ],
                                max_completion_tokens=3000,
                            )
                            _rv_text = (
                                _rv_resp.choices[0].message.content or ""
                            ).strip()

                        st.session_state["nl_undo_stack"].append(_current_draft)
                        if _replace_full:
                            st.session_state["nl_draft"] = _rv_text
                        else:
                            st.session_state["nl_draft"] = _current_draft.replace(
                                _rev_target, _rv_text, 1
                            )
                        if st.session_state.get("nl_saved_id"):
                            update_newsletter(
                                st.session_state["nl_saved_id"],
                                st.session_state["nl_draft"],
                            )
                        st.rerun()
                    except Exception as _rv_err:
                        st.error(f"Revision failed: {_rv_err}")

            # Undo
            if st.session_state.get("nl_undo_stack"):
                if st.button("↩ Undo last revision", key="nl_undo"):
                    st.session_state["nl_draft"] = st.session_state[
                        "nl_undo_stack"
                    ].pop()
                    if st.session_state.get("nl_saved_id"):
                        update_newsletter(
                            st.session_state["nl_saved_id"],
                            st.session_state["nl_draft"],
                        )
                    st.rerun()

            st.markdown("---")

            # ── Save / Export ─────────────────────────────────
            st.markdown("#### Save & Export")
            _exp1, _exp2, _exp3, _exp4 = st.columns(4, gap="small")

            with _exp1:
                if st.button(
                    "💾 Save draft", use_container_width=True, key="nl_save_btn"
                ):
                    if st.session_state.get("nl_saved_id"):
                        update_newsletter(
                            st.session_state["nl_saved_id"],
                            st.session_state["nl_draft"],
                        )
                        st.success("Saved.")
                    else:
                        _sid = save_newsletter(
                            title=_nl_title,
                            coverage_period=_nl_period or "Recent",
                            content_md=st.session_state["nl_draft"],
                            sources=[],
                            config={},
                        )
                        st.session_state["nl_saved_id"] = _sid
                        st.success(f"Saved as Newsletter #{_sid}.")

            with _exp2:
                st.download_button(
                    "⬇ Export .md",
                    data=st.session_state["nl_draft"].encode("utf-8"),
                    file_name=f"newsletter_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="nl_dl_md",
                )

            with _exp3:
                # Convert markdown to basic HTML for export
                import re as _re_nl

                _html_body = st.session_state["nl_draft"]
                _html_body = _re_nl.sub(
                    r"^# (.+)$", r"<h1>\1</h1>", _html_body, flags=_re_nl.MULTILINE
                )
                _html_body = _re_nl.sub(
                    r"^## (.+)$", r"<h2>\1</h2>", _html_body, flags=_re_nl.MULTILINE
                )
                _html_body = _re_nl.sub(
                    r"^### (.+)$", r"<h3>\1</h3>", _html_body, flags=_re_nl.MULTILINE
                )
                _html_body = _re_nl.sub(
                    r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _html_body
                )
                _html_body = _re_nl.sub(r"\*(.+?)\*", r"<em>\1</em>", _html_body)
                _html_body = _re_nl.sub(
                    r"^- (.+)$", r"<li>\1</li>", _html_body, flags=_re_nl.MULTILINE
                )
                _html_body = _html_body.replace("\n\n", "<br><br>")
                _html_export = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    "<style>body{font-family:Georgia,serif;max-width:800px;margin:40px auto;"
                    "padding:0 24px;color:#1a1a2e;line-height:1.7}"
                    "h1{font-size:26px;border-bottom:2px solid #1a7fb5;padding-bottom:8px}"
                    "h2{font-size:18px;color:#0d3349;margin-top:32px}"
                    "h3{font-size:15px;color:#1a5276}"
                    "li{margin-bottom:6px}</style></head>"
                    f"<body>{_html_body}</body></html>"
                )
                st.download_button(
                    "⬇ Export .html",
                    data=_html_export.encode("utf-8"),
                    file_name=f"newsletter_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="nl_dl_html",
                )

            with _exp4:
                st.download_button(
                    "⬇ Export .txt",
                    data=st.session_state["nl_draft"].encode("utf-8"),
                    file_name=f"newsletter_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="nl_dl_txt",
                )

            # Copy-ready text area
            with st.expander("Copy to clipboard", expanded=False):
                st.text_area(
                    "Select all (Ctrl+A) and copy",
                    value=st.session_state["nl_draft"],
                    height=200,
                    label_visibility="visible",
                    key="nl_copy_area",
                )

        # ── Past newsletters ──────────────────────────────────
        _nl_past = load_newsletters(limit=10)
        if _nl_past:
            st.markdown("---")
            with st.expander(f"Saved Newsletters ({len(_nl_past)})", expanded=False):
                for _np in _nl_past:
                    _np_c1, _np_c2 = st.columns([5, 1])
                    with _np_c1:
                        st.markdown(
                            f"**#{_np['id']}** — {_np['title'] or 'Untitled'} "
                            f"· {(_np['coverage_period'] or '')} "
                            f"· {_np['updated_ts'][:16].replace('T', ' ')} UTC"
                        )
                    with _np_c2:
                        if st.button(
                            "Load", key=f"nl_load_{_np['id']}", use_container_width=True
                        ):
                            st.session_state["nl_draft"] = _np["content_md"] or ""
                            st.session_state["nl_saved_id"] = _np["id"]
                            st.session_state["nl_undo_stack"] = []
                            st.rerun()


# ============================================================
# Tab 1: Q & A
# ============================================================
with tab_qa:
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

        _qa_pbar = st.progress(0, text="Loading knowledge base…")
        try:
            df = load_docs_df()
            uids, mat = load_sqlite_vectors()

            _qa_pbar.progress(25, text="Embedding your question…")
            qvec = embed_text(question)

            _qa_pbar.progress(50, text="Searching for relevant articles…")
            used_faiss = False
            if use_faiss_pref and have_faiss():
                matches = faiss_search(qvec, top_k)
                ordered = [u for u, _ in matches]
                used_faiss = True
            else:
                matches = cosine_search(qvec, uids, mat, top_k)
                ordered = [u for u, _ in matches]

            if not ordered:
                _qa_pbar.empty()
                st.warning("No matches found in the current dataset.")
                st.stop()

            context, metas = assemble_context(df, ordered)
            history = st.session_state["messages"][:-1]

            _qa_pbar.progress(75, text="Generating answer…")
            answer, perf = chat_answer(
                question=question,
                context=context,
                history=history,
                chat_deployment=st.session_state["chat_deployment"],
            )
            _qa_pbar.progress(100, text="Done")
            _qa_pbar.empty()

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
            _qa_pbar.empty()
            st.error(str(e))
        except Exception as e:  # pragma: no cover
            _qa_pbar.empty()
            st.error(f"Error while answering: {e}")

    # Query history (collapsed by default)
    with st.expander("🕓 Query History", expanded=False):
        if not os.path.exists(LOG_DB_PATH):
            st.info("No logs yet — run a query above first.")
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
                for r in rows:
                    qid, ts, q, topk, faiss_flag, lat, toks = r
                    col1, col2, col3 = st.columns([5, 2, 1])
                    with col1:
                        st.markdown(f"**{qid}. {q[:100]}**")
                        st.caption(f"{ts}")
                    with col2:
                        st.caption(
                            f"Top {topk} | {'FAISS' if faiss_flag else 'cosine'}"
                        )
                    with col3:
                        if st.button("Re-run", key=f"r{qid}"):
                            st.session_state["messages"].append(
                                {"role": "user", "content": q}
                            )
                            st.rerun()

# ============================================================
# Tab 2: AI Article Search & Summary
# ============================================================
with tab_search:

    # Lazy imports — only load pipeline when this tab is used
    try:
        from src.pipelines.ai_search import (
            generate_search_queries,
            run_search_provider,
            normalize_results as ai_normalize,
            deduplicate_results,
            score_results,
            synthesize_briefing,
            save_ai_search_run,
            load_recent_runs,
            CHAT_DEPLOYMENT_5 as _AI_DEPLOY_5,
            CHAT_DEPLOYMENT_4O as _AI_DEPLOY_4O,
            AI_SEARCH_DB,
        )
        from src.connectors.search_provider import (
            get_provider,
            diagnose_provider,
            SearchError,
            ProviderDiagnostic,
        )

        _ai_search_ok = True
    except Exception as _ai_import_err:
        _ai_search_ok = False
        st.error(f"Could not load AI Search pipeline: {_ai_import_err}")

    if _ai_search_ok:

        # Briefing panel CSS handled by unified theme block

        # ── Domain → source-type mapping for badge colours ────
        _DOMAIN_SOURCE_MAP: Dict[str, str] = {
            "pubmed.ncbi.nlm.nih.gov": "pubmed",
            "ncbi.nlm.nih.gov": "pubmed",
            "europepmc.org": "pubmed",
            "clinicaltrials.gov": "ctgov",
            "medrxiv.org": "medrxiv",
            "biorxiv.org": "biorxiv",
            "nih.gov": "nih_reporter",
            "reporter.nih.gov": "nih_reporter",
            "semanticscholar.org": "semantic_scholar",
            "api.semanticscholar.org": "semantic_scholar",
            "doi.org": "crossref",
        }

        _WEB_LABEL_MAP: Dict[str, str] = {
            "pubmed": "PubMed",
            "openalex": "OpenAlex",
            "ctgov": "ClinicalTrials",
            "semantic_scholar": "Semantic Scholar",
            "crossref": "Crossref",
            "web_search": "Web",
            "nih_reporter": "NIH Reporter",
            "biorxiv": "bioRxiv",
            "medrxiv": "medRxiv",
        }

        def _search_badge(source_name: str) -> tuple:
            """Return (label, badge_css_class) for a search result domain."""
            src = _DOMAIN_SOURCE_MAP.get(source_name.lower(), "web_search")
            label = _WEB_LABEL_MAP.get(src, source_name or "Web")
            cls = f"badge-{src}" if src in _WEB_LABEL_MAP else "badge-default"
            return label, cls

        def _render_search_card(r: Dict[str, Any]) -> str:
            """Render a scored search result dict as an .endo-card."""
            source_name = r.get("source_name", "")
            label, badge_cls = _search_badge(source_name)

            title = str(r.get("title") or "Untitled").strip()
            url = str(r.get("url") or "").strip()
            title_html = (
                f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
                if url
                else title
            )

            date_raw = r.get("published_date") or ""
            try:
                date_str = (
                    datetime.fromisoformat(str(date_raw)[:10])
                    .strftime("%b %d, %Y")
                    .replace(" 0", " ")
                    if date_raw
                    else ""
                )
            except Exception:
                date_str = str(date_raw)[:10]

            domain_label = source_name if source_name else ""
            meta_parts = [p for p in [date_str, domain_label] if p]
            meta_html = " &nbsp;·&nbsp; ".join(meta_parts)

            excerpt = str(r.get("snippet") or "").strip().replace("\n", " ")
            score_pct = int(r.get("composite_score", 0) * 100)
            readmore = (
                f'<a class="endo-readmore" href="{url}" target="_blank" rel="noopener">'
                f"Read more &rarr;</a>"
                if url
                else ""
            )

            return f"""
            <div class="endo-card">
              <div>
                <span class="endo-badge {badge_cls}">{label}</span>
                <span class="endo-quality">Relevance {score_pct}%</span>
              </div>
              <div class="endo-title">{title_html}</div>
              <div class="endo-meta">{meta_html}</div>
              <div class="endo-excerpt">{excerpt[:600]}</div>
              {readmore}
            </div>
            """

        # ── Settings ──────────────────────────────────────────
        with st.expander("Search Settings", expanded=False):
            _s_col1, _s_col2, _s_col3 = st.columns([2, 3, 2])
            with _s_col1:
                _date_range = st.select_slider(
                    "Date range",
                    options=[7, 14, 30, 60, 90, 180],
                    value=90,
                    format_func=lambda d: f"{d} days",
                    key="ai_date_range",
                )
            with _s_col2:
                _all_cats = list(
                    {
                        "Clinical Trials",
                        "Treatments & Drug Pipeline",
                        "Basic Research",
                        "Regulatory & Policy",
                        "Surgery",
                        "Patient Outcomes",
                    }
                )
                _categories = st.multiselect(
                    "Focus areas (all if empty)",
                    _all_cats,
                    default=[],
                    key="ai_categories",
                )
            with _s_col3:
                _ai_model = st.radio(
                    "Synthesis model",
                    ["GPT-5.1", "GPT-4o-mini"],
                    index=0,
                    key="ai_model_choice",
                )
                _deployment = _AI_DEPLOY_5 if _ai_model == "GPT-5.1" else _AI_DEPLOY_4O

        # ── Pre-flight: provider status panel ────────────────
        _diag: ProviderDiagnostic = diagnose_provider()

        _diag_labels = {
            "MISSING_KEY": (
                "error",
                "API key not set",
                "Set **GOOGLE_CSE_KEY** in your `.env` file.",
            ),
            "MISSING_CX": (
                "error",
                "Search Engine ID not set",
                "Set **GOOGLE_CSE_CX** in your `.env` file.",
            ),
            "MISSING_CREDENTIALS": (
                "error",
                "Credentials missing",
                "Set **GOOGLE_CSE_KEY** and **GOOGLE_CSE_CX** in your `.env`.",
            ),
            "INVALID_KEY": (
                "error",
                "API key rejected",
                "Google returned 403. Check that GOOGLE_CSE_KEY is valid and the Custom Search API is enabled in your Google Cloud project.",
            ),
            "INVALID_CX": (
                "error",
                "Search Engine ID rejected",
                "Google returned a CX error. Verify GOOGLE_CSE_CX matches your Custom Search Engine ID.",
            ),
            "QUOTA_EXCEEDED": (
                "warning",
                "Quota exceeded",
                "Your Google CSE free tier (100 queries/day) may be exhausted, or billing is not enabled.",
            ),
            "NO_RESULTS": (
                "warning",
                "No results returned",
                "Credentials are valid but the probe query returned nothing. Make sure the Custom Search Engine is set to search the entire web (not a specific site).",
            ),
            "NETWORK_ERROR": (
                "error",
                "Network error",
                "Could not reach the Google CSE API. Check your internet connection.",
            ),
            "HTTP_ERROR": ("error", "HTTP error", _diag.error_detail),
            "UNKNOWN": ("error", "Unknown error", _diag.error_detail),
        }

        if _diag.ok:
            st.success(
                f"Search provider: **Google CSE** — connected  "
                f"(key: `{_diag.key_hint}` · cx: `{_diag.cx_hint}`)"
            )
        else:
            _sev, _title, _guidance = _diag_labels.get(
                _diag.error_category or "UNKNOWN",
                ("error", _diag.error_category, _diag.error_detail),
            )
            _msg = (
                f"**{_title}** — Google CSE  \n"
                f"Key hint: `{_diag.key_hint}` · CX hint: `{_diag.cx_hint}`  \n"
                f"{_guidance}"
            )
            if _sev == "error":
                st.error(_msg)
            else:
                st.warning(_msg)

        # ── Run button ────────────────────────────────────────
        _run_btn = st.button(
            "Run AI Search",
            type="primary",
            use_container_width=True,
            disabled=not _diag.ok,
            key="ai_run_btn",
        )

        if _run_btn:
            _topic = "endometriosis"
            _provider = get_provider()
            _pbar = st.progress(0, text="Generating search queries…")
            _raw_placeholder = st.empty()

            try:
                # Step 1 — queries
                _queries = generate_search_queries(_topic, _date_range, _categories)
                _pbar.progress(
                    10, text=f"Running {len(_queries)} queries via {_provider.name}…"
                )

                # Step 2 — search
                _raw, _errs = run_search_provider(
                    _queries,
                    _provider,
                    num_per_query=10,
                    date_range_days=_date_range,
                )

                if not _raw:
                    _pbar.empty()
                    st.error(
                        "**No results returned from Google CSE.**  \n"
                        "The search succeeded (no API error) but all queries returned empty results.  \n"
                        "Check that the Custom Search Engine is configured to search the **entire web**, "
                        "not a restricted site list. You can verify this at "
                        "[cse.google.com](https://cse.google.com)."
                    )
                    st.stop()

                # Show raw results immediately so you can verify before synthesis
                with _raw_placeholder.expander(
                    f"Raw search results ({len(_raw)} items from {len(_queries)} queries)",
                    expanded=True,
                ):
                    for _ri, _r in enumerate(_raw[:20]):
                        st.markdown(
                            f"**{_ri + 1}.** [{_r.title}]({_r.url})  \n"
                            f"<span style='color:#888;font-size:0.78rem'>{_r.source_name}"
                            f"{' · ' + _r.published_date if _r.published_date else ''}</span>  \n"
                            f"{_r.snippet[:180]}",
                            unsafe_allow_html=True,
                        )
                    if len(_raw) > 20:
                        st.caption(
                            f"…and {len(_raw) - 20} more (shown in scored sources below)"
                        )

                _pbar.progress(35, text=f"Normalizing {len(_raw)} results…")
                _normed = ai_normalize(_raw)

                _pbar.progress(50, text="Deduplicating…")
                _deduped = deduplicate_results(_normed)

                _pbar.progress(60, text=f"Scoring {len(_deduped)} unique articles…")
                _scored = score_results(_deduped, _topic, _date_range)

                _pbar.progress(
                    70, text="Synthesizing intelligence briefing via Azure GPT…"
                )
                try:
                    _briefing, _perf = synthesize_briefing(
                        _scored, _topic, _date_range, _deployment
                    )
                except Exception as _gpt_exc:
                    _pbar.empty()
                    st.error(
                        f"**Azure GPT synthesis failed.**  \n"
                        f"Search returned {len(_scored)} scored articles successfully.  \n"
                        f"Error: `{_gpt_exc}`  \n"
                        "Check AZURE_OPENAI_CHAT_ENDPOINT, AZURE_OPENAI_CHAT_API_KEY, "
                        "and that the deployment name is correct."
                    )
                    st.stop()

                _pbar.progress(95, text="Saving run…")
                _run_id = save_ai_search_run(
                    topic=_topic,
                    date_range_days=_date_range,
                    categories=_categories,
                    queries=_queries,
                    n_raw=len(_raw),
                    scored_results=_scored,
                    provider_name=_provider.name,
                    model_used=_deployment,
                    latency_ms=_perf.get("latency_ms", 0),
                    briefing_text=_briefing,
                )
                _pbar.progress(100, text="Done")
                _pbar.empty()
                _raw_placeholder.empty()  # collapse raw results once briefing is ready

                st.session_state["ai_search_result"] = {
                    "run_id": _run_id,
                    "briefing": _briefing,
                    "scored": _scored,
                    "n_raw": len(_raw),
                    "n_deduped": len(_deduped),
                    "n_queries": len(_queries),
                    "provider": _provider.name,
                    "model": _deployment,
                    "latency_ms": _perf.get("latency_ms", 0),
                    "date_range": _date_range,
                    "categories": _categories,
                    "errors": _errs,
                    "ts": datetime.now().strftime("%B %d, %Y %H:%M UTC"),
                }

            except SearchError as _exc:
                _pbar.empty()
                _sev2, _t2, _g2 = _diag_labels.get(
                    _exc.category,
                    ("error", _exc.category, _exc.message),
                )
                st.error(f"**Search error — {_t2}**  \n{_g2}")

            except Exception as _exc:
                _pbar.empty()
                st.error(f"**Unexpected error during search run:**  \n`{_exc}`")

        # ── Two-column layout: source cards left, briefing panel right ──
        _result = st.session_state.get("ai_search_result")
        _main_col, _brief_col = st.columns([1.55, 1], gap="large")

        # ── Right: Intelligence Briefing flyout panel ─────────
        with _brief_col:
            if _result:
                _cats_label = (
                    ", ".join(_result["categories"])
                    if _result["categories"]
                    else "All areas"
                )
                _run_meta = (
                    f"Run #{_result['run_id']} &nbsp;·&nbsp; {_result['ts']}  \n"
                    f"{_result['n_raw']} raw &rarr; {_result['n_deduped']} unique &nbsp;·&nbsp; "
                    f"{_result['model']} &nbsp;·&nbsp; {_result['latency_ms']:,} ms"
                )
                st.markdown(
                    f"""
                    <div class="brief-panel">
                      <div class="brief-panel-header">
                        Intelligence Briefing
                        <div class="brief-panel-run">{_run_meta}</div>
                      </div>
                      <div class="brief-panel-body">
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(_result["briefing"])
                st.markdown("</div></div>", unsafe_allow_html=True)

                # ── Email this briefing ──────────────────────────
                with st.expander("Email this briefing", expanded=False):
                    _cfg = load_smtp_config()
                    if not smtp_configured():
                        st.warning(
                            "SMTP not configured. Add `EMAIL_SMTP_HOST`, `EMAIL_FROM`, "
                            "and `EMAIL_PASSWORD` to your `.env` file."
                        )
                    else:
                        _default_to = _cfg.default_to if _cfg else ""
                        _search_to = st.text_input(
                            "Recipient",
                            value=_default_to,
                            key="search_email_to",
                            placeholder="you@example.com",
                        )
                        if st.button("Send", key="search_email_send"):
                            if not _search_to.strip():
                                st.error("Enter a recipient email address.")
                            else:
                                try:
                                    _search_meta = (
                                        f"Run #{_result['run_id']} · {_result['ts']} · "
                                        f"{_result['n_deduped']} sources · {_result['model']}"
                                    )
                                    send_briefing(
                                        to_addr=_search_to.strip(),
                                        subject=f"Evidence Gap Intelligence Briefing — {_result['ts'][:10]}",
                                        title="Intelligence Briefing",
                                        meta=_search_meta,
                                        briefing_md=_result["briefing"],
                                    )
                                    st.success(f"Sent to {_search_to.strip()}")
                                except Exception as _mail_exc:
                                    st.error(f"Send failed: {_mail_exc}")

                if _result.get("errors"):
                    with st.expander(
                        f"Search warnings ({len(_result['errors'])})", expanded=False
                    ):
                        for _e in _result["errors"]:
                            st.caption(_e)
            else:
                st.markdown(
                    """
                    <div class="brief-panel">
                      <div class="brief-panel-header">Intelligence Briefing</div>
                      <div class="brief-panel-body" style="color:#7fa8cc;font-style:italic;
                           text-align:center;padding:40px 18px;">
                        Run a search to generate your<br>intelligence briefing.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Left: source cards + past runs ────────────────────
        with _main_col:
            if _result:
                st.markdown(
                    f"**{_result['n_deduped']} unique sources** &nbsp;·&nbsp; "
                    f"showing top {min(15, len(_result['scored']))}"
                )
                _top_src = _result["scored"][:15]
                _sc_left, _sc_right = st.columns(2, gap="small")
                for _si, _src in enumerate(_top_src):
                    with _sc_left if _si % 2 == 0 else _sc_right:
                        st.markdown(_render_search_card(_src), unsafe_allow_html=True)

            # Past runs always visible in the left column
            _past_runs = load_recent_runs(AI_SEARCH_DB, limit=10)
            if _past_runs:
                with st.expander(f"Past Runs ({len(_past_runs)})", expanded=False):
                    for _pr in _past_runs:
                        _pr_cols = st.columns([4, 2, 1])
                        with _pr_cols[0]:
                            st.markdown(
                                f"**Run #{_pr['id']}** &nbsp;·&nbsp; "
                                f"{_pr['run_ts'][:16].replace('T', ' ')} UTC"
                            )
                            st.caption(
                                f"{_pr['n_scored']} sources &nbsp;·&nbsp; "
                                f"{_pr['model_used']} &nbsp;·&nbsp; {_pr['latency_ms']:,} ms"
                            )
                        with _pr_cols[1]:
                            st.caption(f"Window: {_pr['date_range_days']} days")
                        with _pr_cols[2]:
                            if st.button("View", key=f"pr_{_pr['id']}"):
                                _pr_sources = json.loads(_pr["sources_json"] or "[]")
                                st.session_state["ai_search_result"] = {
                                    "run_id": _pr["id"],
                                    "briefing": _pr["briefing_text"],
                                    "scored": _pr_sources,
                                    "n_raw": _pr["n_raw"],
                                    "n_deduped": _pr["n_scored"],
                                    "n_queries": _pr["n_queries"],
                                    "provider": _pr["provider"],
                                    "model": _pr["model_used"],
                                    "latency_ms": _pr["latency_ms"],
                                    "date_range": _pr["date_range_days"],
                                    "categories": json.loads(_pr["categories"] or "[]"),
                                    "errors": [],
                                    "ts": _pr["run_ts"][:16].replace("T", " ") + " UTC",
                                }
                                st.rerun()

# ============================================================
# Tab 3: RAG Article Presentation
# ============================================================
with tab_rag:

    # CSS handled by unified theme block at top of file

    # ── Helpers ───────────────────────────────────────────────
    _SOURCE_LABELS: Dict[str, str] = {
        "pubmed": "PubMed",
        "openalex": "OpenAlex",
        "ctgov": "ClinicalTrials",
        "semantic_scholar": "Semantic Scholar",
        "crossref": "Crossref",
        "web_search": "Web",
        "nih_reporter": "NIH Reporter",
        "biorxiv": "bioRxiv",
        "medrxiv": "medRxiv",
    }

    def _badge_class(source: str) -> str:
        return f"badge-{source}" if source in _SOURCE_LABELS else "badge-default"

    def _format_date(raw: str) -> str:
        if not raw:
            return ""
        try:
            d = datetime.fromisoformat(str(raw)[:10])
            return d.strftime("%b %d, %Y").replace(" 0", " ")
        except Exception:
            return str(raw)[:10]

    def _authors_line(authors: Any) -> str:
        if not authors:
            return ""
        if isinstance(authors, list):
            names = [str(a) for a in authors[:3]]
            suffix = " et al." if len(authors) > 3 else ""
            return ", ".join(names) + suffix
        return str(authors)

    def _tags_html(topics: Any, mesh: Any) -> str:
        tags: List[str] = []
        if isinstance(topics, list):
            tags += [str(t) for t in topics[:4]]
        if isinstance(mesh, list) and len(tags) < 4:
            tags += [str(m) for m in mesh[: 4 - len(tags)]]
        return "".join(f'<span class="endo-tag">{t}</span>' for t in tags if t)

    def _render_card(row: pd.Series) -> str:
        source = str(row.get("source", "")).lower()
        label = _SOURCE_LABELS.get(source, source.replace("_", " ").title())
        badge_cls = _badge_class(source)

        title = str(row.get("title") or "Untitled").strip()
        url = str(row.get("url") or row.get("doi") or "").strip()
        title_html = (
            f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
            if url
            else title
        )

        date_str = _format_date(str(row.get("published_date") or ""))
        venue = str(row.get("journal_or_venue") or "").strip()
        authors = _authors_line(row.get("authors"))
        meta_parts = [p for p in [date_str, venue, authors] if p]
        meta_html = " &nbsp;·&nbsp; ".join(meta_parts)

        abstract = (
            str(
                row.get("abstract")
                or row.get("summary")
                or row.get("description")
                or ""
            )
            .strip()
            .replace("\n", " ")
        )

        tags_html = _tags_html(row.get("topics"), row.get("mesh_terms"))
        quality = float(row.get("quality_score") or 0)
        readmore = (
            f'<a class="endo-readmore" href="{url}" target="_blank" rel="noopener">'
            f"Read more &rarr;</a>"
            if url
            else ""
        )

        return f"""
        <div class="endo-card">
          <div>
            <span class="endo-badge {badge_cls}">{label}</span>
            <span class="endo-quality">Quality {quality:.0%}</span>
          </div>
          <div class="endo-title">{title_html}</div>
          <div class="endo-meta">{meta_html}</div>
          <div class="endo-excerpt">{abstract[:600]}</div>
          {"<div class='endo-tags'>" + tags_html + "</div>" if tags_html else ""}
          {readmore}
        </div>
        """

    # Lazy import of synthesis pipeline
    try:
        from src.pipelines.ai_search import (
            synthesize_briefing as _rag_synthesize,
            CHAT_DEPLOYMENT_5 as _RAG_DEPLOY_5,
        )

        _rag_synth_ok = True
    except Exception:
        _rag_synth_ok = False

    # ── Load data ─────────────────────────────────────────────
    try:
        rag_df = load_docs_df()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    all_sources = sorted(rag_df["source"].dropna().unique().tolist())

    # ── Controls (full-width, above columns) ──────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])
    with ctrl_col1:
        search_filter = st.text_input(
            "Filter by keyword",
            placeholder="e.g. biomarker, laparoscopy, IL-6 …",
            key="rag_search",
        )
    with ctrl_col2:
        source_filter = st.multiselect(
            "Source",
            all_sources,
            default=[],
            format_func=lambda s: _SOURCE_LABELS.get(s, s),
            key="rag_sources",
        )
    with ctrl_col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Newest first", "Oldest first", "Quality score"],
            key="rag_sort",
        )

    # ── Filter & sort ─────────────────────────────────────────
    view = rag_df.copy()

    if source_filter:
        view = view[view["source"].isin(source_filter)]

    if search_filter.strip():
        kw = search_filter.strip().lower()
        mask = (
            view["title"].str.lower().str.contains(kw, na=False)
            | view["abstract"].str.lower().str.contains(kw, na=False)
            | view.get("topics", pd.Series(dtype=str))
            .astype(str)
            .str.lower()
            .str.contains(kw, na=False)
        )
        view = view[mask]

    if sort_by == "Newest first":
        view = view.sort_values("published_date", ascending=False, na_position="last")
    elif sort_by == "Oldest first":
        view = view.sort_values("published_date", ascending=True, na_position="last")
    else:
        view = view.sort_values("quality_score", ascending=False, na_position="last")

    # ── Pagination state ──────────────────────────────────────
    PAGE_SIZE = 20
    total = len(view)
    if "rag_page" not in st.session_state:
        st.session_state["rag_page"] = 0

    filter_key = f"{search_filter}|{'|'.join(source_filter)}|{sort_by}"
    if st.session_state.get("_rag_filter_key") != filter_key:
        st.session_state["rag_page"] = 0
        st.session_state["_rag_filter_key"] = filter_key
        st.session_state.pop(
            "rag_brief_result", None
        )  # reset briefing when filters change

    page = st.session_state["rag_page"]
    page_start = page * PAGE_SIZE
    page_end = page_start + PAGE_SIZE
    page_df = view.iloc[page_start:page_end]

    # ── Two-column layout ─────────────────────────────────────
    _rag_cards_col, _rag_brief_col = st.columns([1.55, 1], gap="large")

    # ── Right: AI Briefing panel ──────────────────────────────
    with _rag_brief_col:
        _rag_brief = st.session_state.get("rag_brief_result")

        if _rag_synth_ok:
            _rag_gen_btn = st.button(
                "Generate AI Briefing",
                type="primary",
                use_container_width=True,
                key="rag_gen_brief",
            )
            if _rag_gen_btn and total > 0:
                # Build source dicts from top-quality articles in the filtered view
                _brief_src_df = view.sort_values("quality_score", ascending=False).head(
                    15
                )
                _brief_sources = []
                for _, _brow in _brief_src_df.iterrows():
                    _brief_sources.append(
                        {
                            "title": str(_brow.get("title") or ""),
                            "url": str(_brow.get("url") or _brow.get("doi") or ""),
                            "snippet": str(
                                _brow.get("abstract") or _brow.get("summary") or ""
                            )[:600],
                            "source_name": str(
                                _brow.get("journal_or_venue")
                                or _brow.get("source")
                                or ""
                            ),
                            "published_date": str(_brow.get("published_date") or ""),
                            "composite_score": float(_brow.get("quality_score") or 0),
                        }
                    )

                _topic_ctx = (
                    f"endometriosis — {search_filter.strip()}"
                    if search_filter.strip()
                    else "endometriosis"
                )
                _rag_pbar = st.progress(0, text="Synthesizing briefing from dataset…")
                try:
                    _rag_briefing_text, _rag_perf = _rag_synthesize(
                        _brief_sources, _topic_ctx, 365, _RAG_DEPLOY_5
                    )
                    _rag_pbar.progress(100, text="Done")
                    _rag_pbar.empty()
                    _rag_filter_label = search_filter.strip() or "All articles"
                    _rag_run_id = save_rag_brief(
                        filter_label=_rag_filter_label,
                        n_sources=len(_brief_sources),
                        latency_ms=_rag_perf.get("latency_ms", 0),
                        briefing_text=_rag_briefing_text,
                    )
                    st.session_state["rag_brief_result"] = {
                        "briefing": _rag_briefing_text,
                        "n_sources": len(_brief_sources),
                        "filter": _rag_filter_label,
                        "latency_ms": _rag_perf.get("latency_ms", 0),
                        "ts": datetime.now().strftime("%b %d, %Y %H:%M"),
                        "run_id": _rag_run_id,
                    }
                    _rag_brief = st.session_state["rag_brief_result"]
                except Exception as _rb_exc:
                    _rag_pbar.empty()
                    st.error(f"Briefing failed: {_rb_exc}")

        if _rag_brief:
            st.markdown(
                f"""
                <div class="brief-panel">
                  <div class="brief-panel-header">
                    AI Briefing
                    <div class="brief-panel-run">
                      {_rag_brief['ts']} &nbsp;·&nbsp;
                      {_rag_brief['n_sources']} sources &nbsp;·&nbsp;
                      {_rag_brief['latency_ms']:,} ms<br>
                      Filter: {_rag_brief['filter']}
                    </div>
                  </div>
                  <div class="brief-panel-body">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(_rag_brief["briefing"])
            st.markdown("</div></div>", unsafe_allow_html=True)

            # ── Email this briefing ──────────────────────────
            with st.expander("Email this briefing", expanded=False):
                _rag_cfg = load_smtp_config()
                if not smtp_configured():
                    st.warning(
                        "SMTP not configured. Add `EMAIL_SMTP_HOST`, `EMAIL_FROM`, "
                        "and `EMAIL_PASSWORD` to your `.env` file."
                    )
                else:
                    _rag_default_to = _rag_cfg.default_to if _rag_cfg else ""
                    _rag_email_to = st.text_input(
                        "Recipient",
                        value=_rag_default_to,
                        key="rag_email_to",
                        placeholder="you@example.com",
                    )
                    if st.button("Send", key="rag_email_send"):
                        if not _rag_email_to.strip():
                            st.error("Enter a recipient email address.")
                        else:
                            try:
                                _rag_meta = (
                                    f"{_rag_brief['ts']} · {_rag_brief['n_sources']} sources · "
                                    f"Filter: {_rag_brief['filter']}"
                                )
                                send_briefing(
                                    to_addr=_rag_email_to.strip(),
                                    subject=f"Evidence Gap AI Briefing — {_rag_brief['ts'][:10]}",
                                    title="AI Briefing",
                                    meta=_rag_meta,
                                    briefing_md=_rag_brief["briefing"],
                                )
                                st.success(f"Sent to {_rag_email_to.strip()}")
                            except Exception as _rag_mail_exc:
                                st.error(f"Send failed: {_rag_mail_exc}")
        else:
            st.markdown(
                """
                <div class="brief-panel">
                  <div class="brief-panel-header">AI Briefing</div>
                  <div class="brief-panel-body" style="color:#7fa8cc;font-style:italic;
                       text-align:center;padding:40px 18px;">
                    Apply filters then click<br><strong style="color:#a8c8e8">
                    Generate AI Briefing</strong><br>to synthesize the current view.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Left: article cards + pagination ─────────────────────
    with _rag_cards_col:
        st.caption(
            f"Showing {page_start + 1}–{min(page_end, total)} of {total:,} articles"
            + (f' matching "{search_filter}"' if search_filter.strip() else "")
        )

        if page_df.empty:
            st.info("No articles match your filters.")
        else:
            left_col, right_col = st.columns(2, gap="medium")
            rows_list = list(page_df.iterrows())
            for i, (_, row) in enumerate(rows_list):
                col = left_col if i % 2 == 0 else right_col
                with col:
                    st.markdown(_render_card(row), unsafe_allow_html=True)

        if total > PAGE_SIZE:
            max_page = (total - 1) // PAGE_SIZE
            nav_l, nav_c, nav_r = st.columns([1, 3, 1])
            with nav_l:
                if st.button("← Previous", disabled=page == 0, key="rag_prev"):
                    st.session_state["rag_page"] -= 1
                    st.rerun()
            with nav_c:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;color:#888;font-size:0.82rem;'>"
                    f"Page {page + 1} of {max_page + 1}</div>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                if st.button("Next →", disabled=page >= max_page, key="rag_next"):
                    st.session_state["rag_page"] += 1
                    st.rerun()
