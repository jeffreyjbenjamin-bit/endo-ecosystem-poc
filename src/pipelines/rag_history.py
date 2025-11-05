"""
rag_history.py — view recent RAG logs
Usage:
  python src/pipelines/rag_history.py [limit]
"""

import os
import sys
import sqlite3

SQLITE_PATH = "./data/rag_logs.sqlite"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    if not os.path.exists(SQLITE_PATH):
        print("No log DB found. Run rag_ask.py at least once.")
        return
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, ts_utc, substr(question,1,120), top_k, used_faiss, chat_deployment, latency_ms, total_tokens
        FROM queries
        ORDER BY id DESC
        LIMIT ?
    """,
        (limit,),
    )
    rows = cur.fetchall()
    con.close()

    print(f"Last {len(rows)} queries:")
    for r in rows:
        qid, ts, q, topk, faiss, chat_dep, lat, toks = r
        print(
            f"#{qid} | {ts} | top_k={topk} | {'FAISS' if faiss else 'cosine'} | chat={chat_dep} | "
            f"lat={lat}ms | tokens={toks} | Q: {q}"
        )


if __name__ == "__main__":
    main()
