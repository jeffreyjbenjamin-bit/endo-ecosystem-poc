"""
paths.py — where the app keeps data it writes while running.

STATE_DIR holds the app's saved state: AI search runs (Intelligence Briefings),
RAG briefs, comparisons, newsletters, the mailing list, and Q&A logs.

- Locally it defaults to <repo>/data, so nothing changes for local runs.
- On Azure, APP_STATE_DIR points it at persistent storage (/home/data), so the
  data survives deploys and restarts. The read-only corpus (documents.parquet,
  vector store, FAISS index) stays in <repo>/data inside the image.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

STATE_DIR = Path(os.getenv("APP_STATE_DIR") or (REPO_ROOT / "data"))
STATE_DIR.mkdir(parents=True, exist_ok=True)


def state_path(filename: str) -> str:
    """Absolute path (as str) for a file in the app's state directory."""
    return str(STATE_DIR / filename)
