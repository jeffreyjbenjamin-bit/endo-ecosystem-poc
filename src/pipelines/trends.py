"""AI analysis of top endometriosis trends over the past week, month and year.

Input: everything the app has collected — Intelligence Briefing search results,
the evidence library, and clinical trials — filtered by date for each window.
One model call writes the trends for all three windows. The result is cached as
JSON in the app state directory; the Home page shows it and can regenerate it,
and the daily send regenerates it after each new briefing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.paths import state_path

CACHE_FILE = "trends.json"
WINDOWS = (
    ("week", "Past week", 7),
    ("month", "Past month", 30),
    ("year", "Past year", 365),
)
MAX_ITEMS_PER_WINDOW = 160

_SYSTEM = """\
You are a senior life sciences analyst. From dated lists of endometriosis research \
articles, news, company press releases and clinical trials, identify the top trends \
for each time window.

Return ONLY a JSON object, no Markdown fences, shaped exactly like:
{"week": {"headline": "...", "trends": [{"title": "...", "detail": "...", \
"signal": "rising|steady|new"}]}, "month": {...}, "year": {...}}

Rules:
- 3 to 5 trends per window, most important first.
- title: a short noun phrase (max 8 words). detail: 1-2 sentences naming the \
specific compounds, tests, companies, trial phases or findings behind the trend.
- headline: one sentence summarising the window.
- Use only the items given. Do not invent studies, numbers or names.
- Prefer themes supported by several items; note when a theme is driven by one \
company or study.
- The year window should describe longer-running shifts, not repeat the week.
- If a window has fewer than 5 items, say in the headline that the data is \
limited and give at most 2 trends."""


def _parse_date(value: Any) -> Optional[date]:
    s = str(value or "").strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m = re.match(r"(\d{4})-(\d{2})$", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), 1)
        except ValueError:
            return None
    return None


def collect_items(
    runs: List[Dict[str, Any]],
    docs: Optional[List[Dict[str, Any]]] = None,
    trials: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """All dated items as {date, kind, title, source}, de-duplicated by title."""
    items: Dict[str, Dict[str, Any]] = {}

    def add(d: Optional[date], kind: str, title: str, source: str) -> None:
        title = " ".join(str(title or "").split())
        if not d or not title:
            return
        key = re.sub(r"\W+", "", title.lower())[:120]
        if key not in items or items[key]["date"] < d:
            items[key] = {"date": d, "kind": kind, "title": title, "source": source}

    for run in runs:
        run_day = _parse_date(run.get("run_ts"))
        try:
            sources = json.loads(run.get("sources_json") or "[]")
        except ValueError:
            sources = []
        for s in sources:
            d = _parse_date(s.get("published_date")) or run_day
            kind = "press release" if s.get("is_press_release") else "article"
            add(d, kind, s.get("title"), s.get("source_name", ""))
    for doc in docs or []:
        add(
            _parse_date(doc.get("published_date")),
            "article",
            doc.get("title"),
            doc.get("source", ""),
        )
    for t in trials or []:
        add(
            _parse_date(t.get("first_posted")),
            "clinical trial",
            f"{t.get('title')} ({t.get('lead_sponsor')}; {t.get('study_phase')})",
            "ClinicalTrials.gov",
        )
    return sorted(items.values(), key=lambda i: i["date"], reverse=True)


def window_items(
    items: List[Dict[str, Any]], days: int, today: Optional[date] = None
) -> List[Dict[str, Any]]:
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=days)
    return [i for i in items if start <= i["date"] <= today]


def build_prompt(items: List[Dict[str, Any]], today: Optional[date] = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    parts = [f"Today: {today.isoformat()}"]
    for key, label, days in WINDOWS:
        sel = window_items(items, days, today)
        parts.append(f"\n=== {key.upper()} ({label.lower()}, {len(sel)} items) ===")
        for i in sel[:MAX_ITEMS_PER_WINDOW]:
            parts.append(
                f"- {i['date'].isoformat()} | {i['kind']} | {i['title']} | {i['source']}"
            )
    return "\n".join(parts)


def parse_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1])
    out: Dict[str, Any] = {}
    for key, _label, _days in WINDOWS:
        w = data.get(key) or {}
        trends = [
            {
                "title": str(t.get("title", "")).strip(),
                "detail": str(t.get("detail", "")).strip(),
                "signal": str(t.get("signal", "")).strip().lower(),
            }
            for t in (w.get("trends") or [])
            if t.get("title")
        ][:5]
        out[key] = {"headline": str(w.get("headline", "")).strip(), "trends": trends}
    return out


def _load_runs(limit: int = 400) -> List[Dict[str, Any]]:
    from src.pipelines.ai_search import AI_SEARCH_DB

    if not Path(AI_SEARCH_DB).exists():
        return []
    conn = sqlite3.connect(AI_SEARCH_DB)
    try:
        cur = conn.execute(
            "SELECT run_ts, sources_json FROM ai_search_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [{"run_ts": r[0], "sources_json": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()


def _load_docs() -> List[Dict[str, Any]]:
    try:
        import pandas as pd

        from src.common.paths import REPO_ROOT

        path = REPO_ROOT / "data" / "documents.parquet"
        df = pd.read_parquet(path, columns=["title", "source", "published_date"])
        return df.fillna("").to_dict("records")
    except Exception:
        return []


def _load_trials() -> List[Dict[str, Any]]:
    try:
        from src.pipelines.clinical_trials import load_trials

        return (load_trials() or {}).get("trials") or []
    except Exception:
        return []


def generate_trends(deployment: Optional[str] = None) -> Dict[str, Any]:
    """Run the analysis, cache it, and return it."""
    from src.pipelines.ai_search import CHAT_DEPLOYMENT_5, make_chat_client

    deployment = deployment or CHAT_DEPLOYMENT_5
    items = collect_items(_load_runs(), _load_docs(), _load_trials())
    today = datetime.now(timezone.utc).date()
    prompt = build_prompt(items, today)
    client = make_chat_client(deployment)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=2500,
    )
    windows = parse_response(resp.choices[0].message.content or "")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": deployment,
        "counts": {k: len(window_items(items, d, today)) for k, _l, d in WINDOWS},
        "windows": windows,
    }
    path = Path(state_path(CACHE_FILE))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return payload


def load_trends() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(state_path(CACHE_FILE)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
