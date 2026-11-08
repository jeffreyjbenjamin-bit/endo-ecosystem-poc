"""AI analysis of top endometriosis trends over the past week, month and year.

Input: everything the app has collected — Intelligence Briefing search results,
the evidence library, and clinical trials — filtered by date for each window.
Each item gets an id; one model call writes the trends for all three windows and
lists, for every trend, the ids of the items that support it. Those ids are
checked against the window and turned into counts and dates for the Home page
charts, so the charts show the source data, not numbers the model wrote.

The model is Claude through the shared AI service (src/services/ai_service.py,
feature "trends", with a JSON schema so the answer is always valid), or Azure
OpenAI GPT-5.1 when TRENDS_AI_PROVIDER=azure or no Anthropic key is set.

The result is cached as JSON in the app state directory; the Home page shows it
and can regenerate it, and the daily send regenerates it after each briefing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
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
# Most recent items listed per window (older ones in a window are left out).
MAX_ITEMS = {"week": 160, "month": 250, "year": 450}
MAX_EVIDENCE = 40
KIND_LABELS = {
    "article": "Articles",
    "press release": "Press releases",
    "clinical trial": "Clinical trials",
}

_SYSTEM = """\
You are a senior life sciences analyst. From dated lists of endometriosis research \
articles, news, company press releases and clinical trials, identify the top trends \
for each time window. Every item has an id in square brackets, e.g. [42].

For each window return a one-sentence headline and 3 to 5 trends, most important \
first. For each trend:
- title: a short noun phrase (max 8 words)
- detail: 1-2 sentences naming the specific compounds, tests, companies, trial \
phases or findings behind the trend
- signal: "rising", "steady" or "new"
- evidence: the ids of up to 40 items from that window's list that support the \
trend, most relevant first. Only use ids listed in that window.

Rules:
- Use only the items given. Do not invent studies, numbers or names.
- Prefer themes supported by several items; note when a theme is driven by one \
company or study.
- The year window should describe longer-running shifts, not repeat the week.
- If a window has fewer than 5 items, say in the headline that the data is \
limited and give at most 2 trends."""

_AZURE_SHAPE = """

Return ONLY a JSON object, no Markdown fences, shaped exactly like:
{"week": {"headline": "...", "trends": [{"title": "...", "detail": "...", \
"signal": "rising|steady|new", "evidence": [12, 40]}]}, "month": {...}, "year": {...}}"""

_TREND_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "detail": {"type": "string"},
        "signal": {"type": "string", "enum": ["rising", "steady", "new"]},
        "evidence": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["title", "detail", "signal", "evidence"],
    "additionalProperties": False,
}
_WINDOW_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "trends": {"type": "array", "items": _TREND_SCHEMA},
    },
    "required": ["headline", "trends"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {k: _WINDOW_SCHEMA for k, _l, _d in WINDOWS},
    "required": [k for k, _l, _d in WINDOWS],
    "additionalProperties": False,
}


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
    """All dated items as {id, date, kind, title, source}, newest first,
    de-duplicated by title. id is the item's position in this list."""
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
    out = sorted(items.values(), key=lambda i: i["date"], reverse=True)
    for n, item in enumerate(out):
        item["id"] = n
    return out


def window_items(
    items: List[Dict[str, Any]], days: int, today: Optional[date] = None
) -> List[Dict[str, Any]]:
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=days)
    return [i for i in items if start <= i["date"] <= today]


def listed_items(
    items: List[Dict[str, Any]], key: str, days: int, today: Optional[date] = None
) -> List[Dict[str, Any]]:
    """The items of a window that are shown to the model (newest first, capped)."""
    return window_items(items, days, today)[: MAX_ITEMS[key]]


def build_prompt(items: List[Dict[str, Any]], today: Optional[date] = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    parts = [f"Today: {today.isoformat()}"]
    for key, label, days in WINDOWS:
        total = len(window_items(items, days, today))
        sel = listed_items(items, key, days, today)
        shown = f"{total} items" + (
            f", newest {len(sel)} listed" if len(sel) < total else ""
        )
        parts.append(f"\n=== {key.upper()} ({label.lower()}, {shown}) ===")
        for i in sel:
            parts.append(
                f"[{i['id']}] {i['date'].isoformat()} | {i['kind']} | {i['title']} | "
                f"{i['source']}"
            )
    return "\n".join(parts)


def parse_response(
    text: str,
    items: Optional[List[Dict[str, Any]]] = None,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Parse the model's JSON. Evidence ids are kept only if they were listed in
    that window, and become dates and kinds for the charts."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1])
    by_id = {i["id"]: i for i in items or []}
    out: Dict[str, Any] = {}
    for key, _label, days in WINDOWS:
        allowed = (
            {i["id"] for i in listed_items(items, key, days, today)} if items else set()
        )
        w = data.get(key) or {}
        trends = []
        for t in w.get("trends") or []:
            if not t.get("title"):
                continue
            ids: List[int] = []
            for v in t.get("evidence") or []:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                if v in allowed and v not in ids:
                    ids.append(v)
            ids = ids[:MAX_EVIDENCE]
            ev = [by_id[v] for v in ids]
            trends.append(
                {
                    "title": str(t.get("title", "")).strip(),
                    "detail": str(t.get("detail", "")).strip(),
                    "signal": str(t.get("signal", "")).strip().lower(),
                    "n_evidence": len(ev),
                    "evidence": [
                        {
                            "date": e["date"].isoformat(),
                            "kind": e["kind"],
                            "title": e["title"],
                        }
                        for e in ev
                    ],
                }
            )
        out[key] = {
            "headline": str(w.get("headline", "")).strip(),
            "trends": trends[:5],
        }
    return out


def monthly_activity(
    items: List[Dict[str, Any]], today: Optional[date] = None, months: int = 12
) -> List[Dict[str, Any]]:
    """Items per calendar month by kind for the last `months` months (zeros kept)."""
    today = today or datetime.now(timezone.utc).date()
    keys = []
    y, m = today.year, today.month
    for _ in range(months):
        keys.append(f"{y:04d}-{m:02d}")
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    keys.reverse()
    counts = Counter(
        (i["date"].strftime("%Y-%m"), i["kind"]) for i in items if i["date"] <= today
    )
    return [
        {"month": k, "kind": KIND_LABELS[kind], "count": counts.get((k, kind), 0)}
        for k in keys
        for kind in KIND_LABELS
    ]


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


def _run_azure(prompt: str, deployment: Optional[str]) -> tuple:
    from src.pipelines.ai_search import CHAT_DEPLOYMENT_5, make_chat_client

    deployment = deployment or CHAT_DEPLOYMENT_5
    client = make_chat_client(deployment)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM + _AZURE_SHAPE},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=8000,
    )
    return resp.choices[0].message.content or "", deployment


def generate_trends(deployment: Optional[str] = None) -> Dict[str, Any]:
    """Run the analysis, cache it, and return it.

    `deployment` forces an Azure OpenAI deployment (used only on the Azure path).
    """
    from src.services import ai_service as ai

    items = collect_items(_load_runs(), _load_docs(), _load_trials())
    today = datetime.now(timezone.utc).date()
    prompt = build_prompt(items, today)
    if ai.provider_for("trends") == "anthropic" and not deployment:
        result = ai.generate(
            "trends", system=_SYSTEM, prompt=prompt, json_schema=SCHEMA
        )
        text, model = result.text, result.model
        provider = "anthropic"
    else:
        text, model = _run_azure(prompt, deployment)
        provider = "azure"
    windows = parse_response(text, items, today)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "counts": {k: len(window_items(items, d, today)) for k, _l, d in WINDOWS},
        "activity": monthly_activity(items, today),
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
