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
from src.common.page_images import image_from_search_item
from src.common.paths import state_path

load_dotenv()

# ── Paths / config ────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_SEARCH_DB = state_path("ai_search_runs.sqlite")

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
    # Press-release wires and biotech trade press (company-reported news)
    "businesswire.com",
    "prnewswire.com",
    "globenewswire.com",
    "accessnewswire.com",
    "newswire.ca",
    "biospace.com",
    "fiercebiotech.com",
    "fiercehealthcare.com",
    "medtechdive.com",
    "massdevice.com",
}

# Hosts whose pages are company press releases
_PRESS_RELEASE_HOSTS: set = {
    "businesswire.com",
    "prnewswire.com",
    "globenewswire.com",
    "accessnewswire.com",
    "accesswire.com",
    "newswire.ca",
    "einpresswire.com",
    "newswire.com",
    "prweb.com",
}
_PRESS_RELEASE_PATTERN = re.compile(
    r"press[-_ ]releases?|news[-_ ]releases?|/investors?/news|/media/news", re.I
)
_IMPORTANCE_TERMS = [
    "fda approval",
    "fda approved",
    "ema approval",
    "phase 3",
    "phase iii",
    "phase 2",
    "phase ii",
    "phase 1",
    "phase i ",
    "topline",
    "first patient dosed",
    "ind clearance",
    "510(k)",
    "breakthrough device",
    "ce mark",
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
    "Diagnostics": [
        "{topic} diagnostic biomarker {year}",
        "{topic} non-invasive early detection test {year}",
        "{topic} imaging diagnostic tool approval {year}",
        "{topic} blood test serum marker diagnosis {year}",
    ],
    "Therapeutics": [
        "{topic} novel therapeutic target {year}",
        "{topic} hormonal non-hormonal therapy {year}",
        "{topic} immunotherapy anti-inflammatory treatment {year}",
        "{topic} pain management excision ablation outcomes {year}",
    ],
    "Epidemiology": [
        "{topic} prevalence incidence epidemiology {year}",
        "{topic} risk factors comorbidities population {year}",
        "{topic} diagnosis delay time to diagnosis {year}",
        "{topic} economic burden healthcare utilization {year}",
    ],
    "Basic Research": [
        "{topic} pathogenesis mechanism research {year}",
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
    # Company press releases from diagnostics and therapeutics companies with
    # Phase 1–3 programs (wire services plus announcement-style queries)
    "Industry & Press Releases": [
        "{topic} clinical trial press release site:businesswire.com",
        "{topic} clinical trial press release site:prnewswire.com",
        "{topic} clinical trial press release site:globenewswire.com",
        "{topic} biotech announces phase 1 OR phase 2 OR phase 3 {year}",
        "{topic} company announces topline results candidate {year}",
        "{topic} diagnostic test company announces press release {year}",
    ],
}

# Ordered list for the UI's focus-area picker
SEARCH_CATEGORIES: List[str] = list(_CATEGORY_TEMPLATES.keys())

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
            "image": image_from_search_item(r.raw or {}) or "",
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


def is_press_release(r: Dict[str, Any]) -> bool:
    """Company press release: from a wire service, or a page labelled as one.

    Journals, government agencies and medical societies (the high-credibility list)
    also publish "press releases", but they aren't company news and already rank
    well, so they are not counted.
    """
    host = (r.get("source_name") or "").lower()
    if any(host == h or host.endswith("." + h) for h in _PRESS_RELEASE_HOSTS):
        return True
    if any(host == h or host.endswith("." + h) for h in _HIGH_CREDIBILITY):
        return False
    text = " ".join(str(r.get(k) or "") for k in ("url", "title", "snippet"))
    return bool(_PRESS_RELEASE_PATTERN.search(text))


# How many sources the briefing is written from, and how many of those slots
# relevant press releases may claim if they would otherwise be ranked out.
BRIEFING_SOURCES = 15
MIN_PRESS_RELEASES = 3


def _promote_press_releases(
    ranked: List[Dict[str, Any]],
    topic: str,
    top_n: int = BRIEFING_SOURCES,
    min_n: int = MIN_PRESS_RELEASES,
) -> List[Dict[str, Any]]:
    """Make sure up to `min_n` on-topic press releases are within the top `top_n`.

    Press releases are company-issued, so they score lower on credibility than
    journals and would rarely make the cut on score alone. Displaced results move
    just below the cut, keeping their relative order.
    """
    stem = topic.lower()[:8]  # "endometr" matches endometriosis/endometrioma
    on_topic = lambda r: stem in (r["title"] + " " + r["snippet"]).lower()  # noqa: E731
    top, rest = ranked[:top_n], ranked[top_n:]
    have = sum(1 for r in top if r.get("is_press_release"))
    wanted = [r for r in rest if r.get("is_press_release") and on_topic(r)]
    wanted = wanted[: max(0, min_n - have)]
    if not wanted:
        return ranked
    displaced: List[Dict[str, Any]] = []
    for _ in wanted:
        for i in range(len(top) - 1, -1, -1):
            if not top[i].get("is_press_release"):
                displaced.insert(0, top.pop(i))
                break
    top += wanted
    top.sort(key=lambda x: x["composite_score"], reverse=True)
    rest = [r for r in rest if r not in wanted]
    return top + displaced + rest


def score_results(
    results: List[Dict[str, Any]],
    topic: str,
    date_range_days: int,
) -> List[Dict[str, Any]]:
    """Add per-dimension scores + composite, return sorted descending.

    Press releases are tagged (is_press_release) and a few on-topic ones are kept
    among the top BRIEFING_SOURCES so company news reaches the briefing.
    """
    for r in results:
        r["is_press_release"] = is_press_release(r)
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
    ranked = sorted(results, key=lambda x: x["composite_score"], reverse=True)
    return _promote_press_releases(ranked, topic)


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
4–6 entries. Each entry is a standalone paragraph block — do NOT use bullet dashes or list markers.
Follow this exact structure for each entry, with one blank line between entries:

**[Bolded one-line headline that captures the finding or announcement.]** [N] Two to three sentences of analytical context: explain what was found or announced, the clinical or strategic significance, and any key data points, outcomes, or named compounds/trials involved. Be specific — avoid vague summaries.

*→ [Descriptive link text drawn from the article title](URL)*

Use the exact URL from the source list. If no URL is available, omit the link line.

## Clinical & Regulatory Highlights
Describe clinical trial results, FDA/EMA actions, or regulatory milestones visible in the sources. Cite inline with [N]. If none are present, say so explicitly.

## Diagnostics
Summarize advances in biomarkers, imaging, and non-invasive or early-detection methods visible in the sources. Include diagnostic tool approvals, CE marks, FDA clearances, or pipeline items where present. Cite inline with [N]. If no diagnostic developments are present, say so explicitly.

## Therapeutics
Summarize novel therapeutic approaches, targets, or treatment outcomes visible in the sources. Include hormonal, non-hormonal, immunological, and surgical modalities. Note stage of evidence (preclinical, Phase I/II/III, approved). Cite inline with [N]. If no therapeutic developments are present, say so explicitly.

## Industry & Company News
Summarize company press releases (sources marked "Type: Company press release") from diagnostics and therapeutics companies about endometriosis programs. For each, name the company, the product or candidate, its modality (diagnostic test, drug, device), its development stage (Phase 1, Phase 2, Phase 3, or regulatory status such as filing, clearance or approval), and the news (trial start, enrollment, topline results, partnership, financing, regulatory step). Attribute claims to the company (e.g. "Company X reported…"), since press releases are not peer reviewed. Cite inline with [N]. If no press releases are present, say so explicitly.

## Epidemiology
Summarize findings on prevalence, incidence, diagnosis delay, risk factors, comorbidities, or healthcare burden visible in the sources. Highlight any population-level or demographic patterns. Cite inline with [N]. If no epidemiological findings are present, say so explicitly.

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
- Do not list a source in References unless it was cited inline.

DATE RULES:
- Where a source has a known publication date, include it inline immediately after the citation number in parentheses, e.g. [3] (June 28, 2025).
- In the References section, always include the full date when known.
- If most sources fall outside the 7-day window, note this explicitly in the Intelligence Summary."""


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
    max_sources: int = BRIEFING_SOURCES,
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
        if r.get("is_press_release"):
            parts.append("    Type: Company press release (company-reported)")
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
                max_output_tokens=3000,
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
                max_completion_tokens=3000,
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
            max_completion_tokens=3000,
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
