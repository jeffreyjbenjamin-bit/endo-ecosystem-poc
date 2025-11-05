"""
rag_log.py — logging & caching for RAG queries.

Writes to:
  - ./data/rag_logs.sqlite  (tables: queries, query_docs)
  - ./data/rag_logs.jsonl   (append-only line-per-query)

Call `log_query(...)` once per RAG answer.
"""

import os
import json
import sqlite3
from typing import List, Dict, Optional

DATA_DIR = "./data"
SQLITE_PATH = os.path.join(DATA_DIR, "rag_logs.sqlite")
JSONL_PATH = os.path.join(DATA_DIR, "rag_logs.jsonl")


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            question TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            used_faiss INTEGER NOT NULL,
            embed_endpoint TEXT,
            embed_deployment TEXT,
            chat_endpoint TEXT,
            chat_deployment TEXT,
            latency_ms INTEGER,
            total_tokens INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            answer TEXT
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS query_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            uid TEXT NOT NULL,
            score REAL,
            title TEXT,
            source TEXT,
            url TEXT,
            FOREIGN KEY(query_id) REFERENCES queries(id)
        )
    """
    )
    conn.commit()


def _open_db() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(SQLITE_PATH)
    _ensure_schema(conn)
    return conn


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
    usage: Optional[
        Dict
    ] = None,  # expects keys like total_tokens, prompt_tokens, completion_tokens if available
    answer: Optional[str] = None,
    docs: Optional[List[Dict]] = None,  # [{rank, uid, score, title, source, url}]
) -> int:
    """
    Insert a query + docs rowset, and append to JSONL.
    Returns the inserted query_id.
    """
    conn = _open_db()
    cur = conn.cursor()

    total_tokens = (usage or {}).get("total_tokens")
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")

    cur.execute(
        """
        INSERT INTO queries (
            ts_utc, question, top_k, used_faiss,
            embed_endpoint, embed_deployment,
            chat_endpoint, chat_deployment,
            latency_ms, total_tokens, prompt_tokens, completion_tokens,
            answer
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            ts_utc,
            question,
            int(top_k),
            1 if used_faiss else 0,
            embed_endpoint,
            embed_deployment,
            chat_endpoint,
            chat_deployment,
            int(latency_ms),
            total_tokens,
            prompt_tokens,
            completion_tokens,
            answer or "",
        ),
    )
    qid = cur.lastrowid

    if docs:
        cur.executemany(
            """
            INSERT INTO query_docs (query_id, rank, uid, score, title, source, url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    qid,
                    int(d.get("rank", i + 1)),
                    d.get("uid", ""),
                    float(d.get("score")) if d.get("score") is not None else None,
                    d.get("title", ""),
                    d.get("source", ""),
                    d.get("url", ""),
                )
                for i, d in enumerate(docs)
            ],
        )

    conn.commit()
    conn.close()

    # JSONL append (human-friendly & exportable)
    _ensure_dirs()
    record = {
        "ts_utc": ts_utc,
        "question": question,
        "top_k": top_k,
        "used_faiss": used_faiss,
        "embed_endpoint": embed_endpoint,
        "embed_deployment": embed_deployment,
        "chat_endpoint": chat_endpoint,
        "chat_deployment": chat_deployment,
        "latency_ms": latency_ms,
        "usage": usage or {},
        "answer": (answer or "")[:100000],  # cap for safety
        "docs": docs or [],
        "query_id": qid,
    }
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return qid
