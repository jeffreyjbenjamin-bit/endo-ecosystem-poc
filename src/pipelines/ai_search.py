"""
ai_search.py — Intelligence briefing pipeline for Evidence Gap.

Pipeline steps
--------------
1. generate_search_queries   — build targeted query set (deterministic + fast)
2. run_search_provider       — execute queries through pluggable search backend
3. normalize_results         — coerce to flat dict schema
4. deduplicate_results       — URL + title-fingerprint dedup
5. score_results             — composite relevance/recency/credibility/importance score
6. synthesize_briefing       — GPT-5.1 synthesis into structured intelligence brief
7. save_ai_search_run        — persist run to SQLite for history & comparison
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI

from src.connectors.search_provider import SearchProvider, SearchResult

load_dotenv()

# ── Paths / config ────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_SEARCH_DB = str(REPO_ROOT / "data" / "ai_search_runs.sqlite")

CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or os.getenv(
    "AZURE_OPENAI_ENDPOINT", ""
)
CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY") or os.getenv(
    "AZURE_OPENAI_API_KEY", ""
)
CHAT_DEPLOYMENT_5 = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_5", "gpt-5.1-chat")
CHAT_DEPLOYMENT_4O = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_4O", "gpt-4o-mini")

# ── Source credibility registry ───────────────────────────────
_HIGH_CREDIBILITY: set = {
    "pubmed.ncbi.nlm.nih.gov",
    "nejm.org",
    "thelancet.com",
    "jamanetwork.com",
    "bmj.com",
    "nature.com",
    "science.org",
    "cell.com",
    "nih.gov",
    "fda.gov",
    "ema.europa.eu",
    "who.int",
    "endocrine.org",
    "asrm.org",
    "acog.org",
    "fertstert.org",
    "ncbi.nlm.nih.gov",
}
_MEDIUM_CREDIBILITY: set = {
    "medscape.com",
    "healio.com",
    "mdedge.com",
    "medpagetoday.com",
    "reuters.com",
    "statnews.com",
    "endpts.com",
    "firstwordpharma.com",
    "fiercepharma.com",
    "biopharmadive.com",
    "clinicaltrials.gov",
    "medrxiv.org",
    "biorxiv.org",
    "webmd.com",
    "healthline.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "oxford.com",
}
_IMPORTANCE_TERMS = [
    "fda approval",
    "fda approved",
    "ema approval",
    "phase 3",
    "phase iii",
    "phase 2",
    "phase ii",
    "clinical trial",
    "randomized controlled",
    "rct",
    "breakthrough",
    "novel therapy",
    "first-in-class",
    "biomarker",
    "diagnostic",
    "laparoscopy",
    "excision",
    "elagolix",
    "linzagolix",
    "relugolix",
    "dienogest",
    "il-6",
    "il-8",
    "vegf",
    "tnf",
]


# ============================================================
# Step 1 — Generate search queries
# ============================================================

_CATEGORY_TEMPLATES: Dict[str, List[str]] = {
    "Clinical Trials": [
        "{topic} clinical trial results {year}",
        "{topic} phase 2 phase 3 trial {year}",
        "{topic} randomized controlled trial {year}",
    ],
    "Treatments & Drug Pipeline": [
        "{topic} new treatment drug approval {year}",
        "{topic} hormonal therapy GnRH antagonist {year}",
        "{topic} drug pipeline FDA {year}",
    ],
    "Basic Research": [
        "{topic} pathogenesis mechanism research {year}",
        "{topic} biomarker diagnosis {year}",
        "{topic} genetics epigenetics {year}",
    ],
    "Regulatory & Policy": [
        "{topic} FDA EMA regulatory action {year}",
        "{topic} guideline consensus {year}",
    ],
    "Surgery": [
        "{topic} surgical treatment laparoscopy excision {year}",
        "{topic} minimally invasive surgery outcomes {year}",
    ],
    "Patient Outcomes": [
        "{topic} quality of life patient outcomes {year}",
        "{topic} pain management fertility {year}",
    ],
}

_BASE_TEMPLATES = [
    "{topic} latest news {year}",
    "{topic} research developments {year}",
    "{topic} treatment advances {year}",
]


def generate_search_queries(
    topic: str,
    date_range_days: int,
    categories: List[str],
) -> List[str]:
    """
    Build a deduplicated list of targeted search queries.
    Deterministic — no LLM call needed at this stage.
    """
    year = datetime.now().year
    ctx = {"topic": topic, "year": year}

    queries: List[str] = [t.format(**ctx) for t in _BASE_TEMPLATES]

    active = categories if categories else list(_CATEGORY_TEMPLATES.keys())
    for cat in active:
        for tpl in _CATEGORY_TEMPLATES.get(cat, []):
            queries.append(tpl.format(**ctx))

    return list(dict.fromkeys(queries))  # preserve order, remove exact dupes


# ============================================================
# Step 2 — Run search provider
# ============================================================


def _date_restrict(days: int, provider_name: str) -> Optional[str]:
    """Translate days → provider-specific date filter value."""
    if provider_name == "bing_grounding":
        if days <= 7:
            return "Day"
        if days <= 31:
            return "Week"
        return "Month"
    # Google CSE: d{n} or m{n}
    if days <= 31:
        return f"d{days}"
    return f"m{max(1, round(days / 30))}"


def run_search_provider(
    queries: List[str],
    provider: SearchProvider,
    num_per_query: int = 10,
    date_range_days: int = 90,
) -> Tuple[List[SearchResult], List[str]]:
    """
    Execute each query. Returns (results, errors).
    One failing query does not abort the run.
    """
    dr = _date_restrict(date_range_days, provider.name)
    all_results: List[SearchResult] = []
    errors: List[str] = []

    for q in queries:
        try:
            kwargs: Dict[str, Any] = {"num": num_per_query}
            if dr:
                if provider.name == "bing_grounding":
                    kwargs["freshness"] = dr
                else:
                    kwargs["date_restrict"] = dr
            results = provider.search(q, **kwargs)
            all_results.extend(results)
        except Exception as exc:
            errors.append(f"{q!r}: {exc}")

    return all_results, errors


# ============================================================
# Step 3 — Normalize
# ============================================================


def normalize_results(raw: List[SearchResult]) -> List[Dict[str, Any]]:
    """Coerce SearchResult objects to flat dicts."""
    return [
        {
            "title": (r.title or "").strip(),
            "url": (r.url or "").strip(),
            "snippet": (r.snippet or "").strip(),
            "source_name": (r.source_name or "").strip(),
            "published_date": r.published_date,
        }
        for r in raw
        if r.url and r.title
    ]


# ============================================================
# Step 4 — Deduplicate
# ============================================================


def _url_key(url: str) -> str:
    u = url.lower().rstrip("/")
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u


def _title_fingerprint(title: str) -> str:
    tokens = sorted(frozenset(re.sub(r"[^a-z0-9 ]", "", title.lower()).split()))
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_urls: set = set()
    seen_titles: set = set()
    out = []
    for r in results:
        uk = _url_key(r["url"])
        tk = _title_fingerprint(r["title"])
        if uk in seen_urls or tk in seen_titles:
            continue
        seen_urls.add(uk)
        seen_titles.add(tk)
        out.append(r)
    return out


# ============================================================
# Step 5 — Score
# ============================================================


def _score_recency(pub_date: Optional[str], date_range_days: int) -> float:
    if not pub_date:
        return 0.35
    try:
        age = (datetime.now() - datetime.fromisoformat(pub_date[:10])).days
        return max(0.0, 1.0 - age / max(date_range_days, 1))
    except Exception:
        return 0.35


def _score_credibility(source_name: str) -> float:
    host = source_name.lower()
    if any(h in host for h in _HIGH_CREDIBILITY):
        return 1.0
    if any(h in host for h in _MEDIUM_CREDIBILITY):
        return 0.6
    return 0.25


def _score_relevance(r: Dict[str, Any], topic: str) -> float:
    tokens = set(topic.lower().split())
    text = (r["title"] + " " + r["snippet"]).lower()
    return min(1.0, sum(1 for t in tokens if t in text) / max(len(tokens), 1))


def _score_importance(r: Dict[str, Any]) -> float:
    text = (r["title"] + " " + r["snippet"]).lower()
    hits = sum(1 for term in _IMPORTANCE_TERMS if term in text)
    return min(1.0, hits / 3)


def score_results(
    results: List[Dict[str, Any]],
    topic: str,
    date_range_days: int,
) -> List[Dict[str, Any]]:
    """Add per-dimension scores + composite, return sorted descending."""
    for r in results:
        r["score_recency"] = _score_recency(r.get("published_date"), date_range_days)
        r["score_credibility"] = _score_credibility(r.get("source_name", ""))
        r["score_relevance"] = _score_relevance(r, topic)
        r["score_importance"] = _score_importance(r)
        r["composite_score"] = (
            0.30 * r["score_recency"]
            + 0.30 * r["score_credibility"]
            + 0.25 * r["score_relevance"]
            + 0.15 * r["score_importance"]
        )
    return sorted(results, key=lambda x: x["composite_score"], reverse=True)


# ============================================================
# Step 6 — Synthesize briefing
# ============================================================

_SYSTEM_PROMPT = """\
You are a senior analyst producing a professional intelligence briefing for life sciences professionals.
Your style mirrors leading life sciences intelligence products: crisp, factual, and actionable.

Write in clear, confident prose. Use the following Markdown structure exactly:

## Intelligence Summary
2–3 sentence executive overview of the most significant recent developments.

## Key Developments
4–6 bullet points. Each bullet begins with a **bolded one-line headline**, followed by 1–2 sentences of context. Cite the source(s) inline using [N] immediately after the relevant claim.

## Clinical & Regulatory Highlights
Describe clinical trial results, FDA/EMA actions, or regulatory milestones visible in the sources. Cite inline with [N]. If none are present, say so explicitly.

## Emerging Research Themes
What patterns, mechanisms, or scientific directions appear across multiple sources? Cite inline with [N].

## What to Watch
2–3 forward-looking observations: upcoming trial readouts, pending regulatory decisions, or open scientific questions actively being investigated.

## References
List every source cited above, numbered to match inline [N] citations. Use this exact format for each:
[N] Author/Org (if available). "Title." Source name. Date. URL

---
CITATION RULES:
- Every factual claim must have at least one inline [N] citation.
- Only cite sources from the numbered list provided. Do not invent or hallucinate sources.
- If a source has no URL, omit the URL field in the References section.
- If a date is unknown, write "date unknown" in the References section.
- Do not list a source in References unless it was cited inline."""


def make_chat_client(deployment: str) -> AzureOpenAI:
    if not (CHAT_ENDPOINT and CHAT_KEY):
        raise RuntimeError(
            "Azure OpenAI chat credentials not configured. "
            "Set AZURE_OPENAI_CHAT_ENDPOINT and AZURE_OPENAI_CHAT_API_KEY."
        )
    api_version = (
        "2025-04-01-preview"
        if deployment == CHAT_DEPLOYMENT_5
        else "2025-01-01-preview"
    )
    return AzureOpenAI(
        api_key=CHAT_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        api_version=api_version,
    )


def synthesize_briefing(
    results: List[Dict[str, Any]],
    topic: str,
    date_range_days: int,
    deployment: str,
    max_sources: int = 15,
) -> Tuple[str, Dict[str, Any]]:
    """
    Pass top-scored sources to GPT for synthesis.
    Returns (briefing_markdown, perf_dict).
    """
    top = results[:max_sources]
    today = datetime.now().strftime("%B %d, %Y")

    def _fmt_source(i: int, r: Dict[str, Any]) -> str:
        title = (r.get("title") or "Untitled").strip()
        source_name = (r.get("source_name") or "").strip()
        url = (r.get("url") or "").strip()
        date = (r.get("published_date") or "date unknown")[:10]
        snippet = (r.get("snippet") or "").strip()
        parts = [f"[{i + 1}] {title}"]
        parts.append(
            f"    Source: {source_name}" if source_name else "    Source: unknown"
        )
        parts.append(f"    Date: {date}")
        if url:
            parts.append(f"    URL: {url}")
        if snippet:
            parts.append(f"    Excerpt: {snippet[:400]}")
        return "\n".join(parts)

    source_block = "\n\n".join(_fmt_source(i, r) for i, r in enumerate(top))

    user_msg = (
        f"Topic: {topic}\n"
        f"Briefing date: {today}\n"
        f"Search window: last {date_range_days} days\n"
        f"Sources reviewed: {len(top)}\n\n"
        f"SOURCES:\n{source_block}"
    )

    client = make_chat_client(deployment)
    t0 = time.time()
    usage: Dict[str, Any] = {}

    # GPT-5.1 uses the Responses API; GPT-4o-mini uses Chat Completions
    if deployment == CHAT_DEPLOYMENT_5:
        try:
            resp = client.responses.create(
                model=deployment,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_output_tokens=2000,
            )
            text = (
                resp.output_text.strip() if hasattr(resp, "output_text") else str(resp)
            )
        except Exception:
            # Fall back to chat completions if responses API not available
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()
    else:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_completion_tokens=2000,
        )
        text = resp.choices[0].message.content.strip()
        u = getattr(resp, "usage", None)
        if u:
            usage = {"total_tokens": getattr(u, "total_tokens", None)}

    latency_ms = int((time.time() - t0) * 1000)
    return text, {"latency_ms": latency_ms, "usage": usage}


# ============================================================
# Step 7 — Save run
# ============================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_search_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts          TEXT    NOT NULL,
    topic           TEXT    NOT NULL,
    date_range_days INTEGER,
    categories      TEXT,
    n_queries       INTEGER,
    n_raw           INTEGER,
    n_scored        INTEGER,
    provider        TEXT,
    model_used      TEXT,
    latency_ms      INTEGER,
    briefing_text   TEXT,
    sources_json    TEXT
);
"""


def _ensure_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def save_ai_search_run(
    topic: str,
    date_range_days: int,
    categories: List[str],
    queries: List[str],
    n_raw: int,
    scored_results: List[Dict[str, Any]],
    provider_name: str,
    model_used: str,
    latency_ms: int,
    briefing_text: str,
    db_path: str = AI_SEARCH_DB,
) -> int:
    conn = _ensure_db(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO ai_search_runs
          (run_ts, topic, date_range_days, categories, n_queries, n_raw, n_scored,
           provider, model_used, latency_ms, briefing_text, sources_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            ts,
            topic,
            date_range_days,
            json.dumps(categories, ensure_ascii=False),
            len(queries),
            n_raw,
            len(scored_results),
            provider_name,
            model_used,
            latency_ms,
            briefing_text,
            json.dumps(scored_results[:20], ensure_ascii=False),
        ),
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def load_recent_runs(
    db_path: str = AI_SEARCH_DB,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """
        SELECT id, run_ts, topic, date_range_days, n_queries, n_raw, n_scored,
               provider, model_used, latency_ms, briefing_text, sources_json
        FROM ai_search_runs
        ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows
