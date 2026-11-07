"""
newsletter_from_briefing.py — "Build Newsletter from Intel Briefing" tab.

Workflow
--------
1. Pick a past Intelligence Briefing (AI Article Search run); defaults to the latest.
2. Pick one or two articles from that briefing as the basis for the newsletter.
3. Search for similar articles (local evidence library via embeddings, and/or web
   search) and choose supporting articles.
4. Generate a newsletter draft from the base + supporting articles, edit it, and
   save it to newsletters.sqlite (the same store the original Newsletter Builder
   uses, so saved drafts can be loaded and sent from there).

The Intelligence Briefing and the Newsletter remain separate reports: this tab only
reads briefings and never modifies them.

rag_panel.py is a Streamlit script, not an importable module, so the helpers this
tab needs are passed in through `render_tab(...)`.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import streamlit as st

MAX_BASE_ARTICLES = 2
_SS = "nb2_"  # session-state key prefix for this tab

_NB2_SYSTEM = """\
You are a professional medical intelligence editor producing a newsletter for life \
sciences professionals focused on endometriosis.

Your output must be clean, publication-ready Markdown that follows the structure in \
the user message exactly.

EDITORIAL RULES:
- Build the newsletter around the FEATURED articles; supporting articles add context, \
corroboration, contrast, or broader landscape.
- Write in confident, precise, third-person editorial prose.
- Every factual claim must cite at least one source using [N] inline, where N is the \
article number given in the user message.
- Only use facts present in the provided article titles, summaries, and metadata. \
If a summary is thin, say less rather than inventing detail.
- Do not invent new findings, numbers, trial names, or quotes.
- Preserve URLs exactly as provided.
- Do not use em dashes excessively. Do not use filler phrases."""


# ============================================================
# Helpers
# ============================================================


def _k(name: str) -> str:
    return f"{_SS}{name}"


def _init_state() -> None:
    for name, default in [
        ("supporting", {}),  # key -> article dict
        ("results", []),
        ("results_sig", None),
        ("search_errors", []),
        ("draft", ""),
        ("saved_id", None),
        ("active_run", None),
    ]:
        if _k(name) not in st.session_state:
            st.session_state[_k(name)] = default


def _clean_title(title: str) -> str:
    """Strip trailing site names ('... - PubMed', '... | Nature') and ellipses."""
    t = (title or "").strip()
    t = re.sub(r"\s*(\.\.\.|…)\s*", " ", t)
    t = re.sub(r"\s+[-|–]\s+[^-|–]{2,40}$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _url_key(url: str) -> str:
    u = (url or "").lower().strip()
    u = u.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    u = re.sub(r"^https?://", "", u)
    return re.sub(r"^www\.", "", u)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _clean_title(t).lower()).strip()


def _is_featured_copy(title: str, url: str, base: List[Dict[str, Any]]) -> bool:
    """True if a result is a featured article itself (same URL, or same paper on
    another site — briefing titles are often truncated, so match on prefix)."""
    uk, nt = _url_key(url), _norm_title(title)
    for b in base:
        if uk == _url_key(b["url"]):
            return True
        bt = _norm_title(b["title"])
        if len(bt) >= 30 and nt.startswith(bt):
            return True
    return False


def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _run_label(run: Dict[str, Any]) -> str:
    ts = (run.get("run_ts") or "")[:16].replace("T", " ")
    try:
        n_src = len(json.loads(run.get("sources_json") or "[]"))
    except Exception:
        n_src = 0
    days = run.get("date_range_days")
    window = f" · last {days}d" if days else ""
    return (
        f"#{run['id']} · {ts} UTC · {run.get('topic', '')}{window} · {n_src} articles"
    )


def _briefing_articles(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Articles in a briefing = its scored sources; reference [N] is position N."""
    try:
        sources = json.loads(run.get("sources_json") or "[]")
    except Exception:
        sources = []
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", run.get("briefing_text") or "")}
    out = []
    for i, s in enumerate(sources, start=1):
        if not s.get("url"):
            continue
        out.append(
            {
                "key": f"brief_{run['id']}_{i}",
                "ref": i,
                "origin": "briefing",
                "title": _clean_title(s.get("title", "")) or s.get("url", ""),
                "url": s.get("url", ""),
                "source": s.get("source_name", ""),
                "published_date": (s.get("published_date") or "")[:10],
                "snippet": (s.get("snippet") or "").strip(),
                "score": s.get("composite_score"),
                "cited": i in cited,
            }
        )
    return out


def _article_header_md(a: Dict[str, Any]) -> str:
    title = a["title"].replace("[", "(").replace("]", ")")
    meta = " · ".join(
        x for x in [a.get("source"), a.get("published_date") or "date unknown"] if x
    )
    return f"**[{title}]({a['url']})**  \n<small>{meta}</small>"


# ============================================================
# Similar-article search
# ============================================================


def _library_search(
    base: List[Dict[str, Any]],
    top_k: int,
    keywords: str,
    deps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Embedding search over the local evidence library, one query per base article."""
    df = deps["load_docs_df"]()
    uids, mat = deps["load_sqlite_vectors"]()
    if mat.size == 0:
        raise RuntimeError("Vector store is empty — run the embeddings pipeline first.")
    by_uid = df.set_index("uid", drop=False)

    merged: Dict[str, Dict[str, Any]] = {}
    for b in base:
        query = f"{b['title']}. {b['snippet']} {keywords}".strip()
        vec = deps["embed_text"](query)
        for uid, sim in deps["cosine_search"](vec, uids, mat, top_k + 5):
            if uid not in by_uid.index:
                continue
            row = by_uid.loc[uid]
            if isinstance(row, type(df)):  # duplicate uid rows
                row = row.iloc[0]
            url = str(row.get("url", "") or "")
            if _is_featured_copy(str(row.get("title", "")), url, base):
                continue
            body = str(
                row.get("abstract", "")
                or row.get("summary", "")
                or row.get("description", "")
            ).strip()
            key = f"lib_{uid}"
            hit = merged.get(key)
            if hit is None:
                merged[key] = {
                    "key": key,
                    "origin": "library",
                    "title": str(row.get("title", "")).strip() or url,
                    "url": url,
                    "source": str(row.get("source", "")),
                    "published_date": str(row.get("published_date", "") or "")[:10],
                    "snippet": body[:1200],
                    "score": float(sim),
                    "matched": [b["ref"]],
                }
            else:
                hit["score"] = max(hit["score"], float(sim))
                if b["ref"] not in hit["matched"]:
                    hit["matched"].append(b["ref"])
    ranked = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
    return ranked[: top_k * max(1, len(base))]


def _web_search(
    base: List[Dict[str, Any]],
    per_article: int,
    keywords: str,
    days: int,
    briefing_urls: Set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    from src.connectors.search_provider import get_provider
    from src.pipelines.ai_search import (
        deduplicate_results,
        normalize_results,
        run_search_provider,
    )

    provider = get_provider()
    if not provider.available():
        return [], [f"Web search provider '{provider.name}' is not configured."]

    out: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for b in base:
        q = b["title"]
        if "endometrio" not in q.lower():
            q += " endometriosis"
        if keywords:
            q += f" {keywords}"
        raw, errs = run_search_provider(
            [q], provider, num_per_query=per_article, date_range_days=days
        )
        errors.extend(errs)
        for rank, r in enumerate(deduplicate_results(normalize_results(raw))):
            uk = _url_key(r["url"])
            if _is_featured_copy(r["title"], r["url"], base):
                continue
            key = f"web_{_hash(uk)}"
            if key in out:
                if b["ref"] not in out[key]["matched"]:
                    out[key]["matched"].append(b["ref"])
                continue
            out[key] = {
                "key": key,
                "origin": "web",
                "title": _clean_title(r["title"]) or r["url"],
                "url": r["url"],
                "source": r.get("source_name", ""),
                "published_date": (r.get("published_date") or "")[:10],
                "snippet": r.get("snippet", ""),
                "score": None,
                "rank": rank + 1,
                "matched": [b["ref"]],
                "in_briefing": uk in briefing_urls,
            }
    return list(out.values()), errors


# ============================================================
# Newsletter generation
# ============================================================


def _article_block(n: int, a: Dict[str, Any]) -> str:
    return (
        f"[{n}] {a['title']}\n"
        f"    Source: {a.get('source') or 'unknown'} | Date: "
        f"{a.get('published_date') or 'unknown'}\n"
        f"    URL: {a['url']}\n"
        f"    Summary: {(a.get('snippet') or '(no summary available)')[:1200]}"
    )


def _build_prompt(
    base: List[Dict[str, Any]],
    supporting: List[Dict[str, Any]],
    title: str,
    briefing_label: str,
    note: str,
) -> str:
    today = datetime.now().strftime("%B %d, %Y")
    numbered = list(enumerate(base + supporting, start=1))
    featured_block = "\n\n".join(_article_block(n, a) for n, a in numbered[: len(base)])
    supporting_block = (
        "\n\n".join(_article_block(n, a) for n, a in numbered[len(base) :])
        or "(none selected)"
    )
    feature_heading = "Lead Story" if len(base) == 1 else "Feature Stories"
    feature_instr = (
        "A focused 3-4 paragraph editorial on featured article [1], weaving in "
        "supporting articles where they add context."
        if len(base) == 1
        else "Two sections, one per featured article, each with a ### headline and "
        "2-3 paragraphs, weaving in supporting articles where they add context. "
        "Close with one short paragraph connecting the two."
    )
    return f"""Produce an endometriosis newsletter built around the FEATURED article(s) below, \
using the SUPPORTING articles for context.

NEWSLETTER CONFIG:
- Title: {title}
- Seeded from Intelligence Briefing: {briefing_label}
- Published: {today}
{('- Editorial note: ' + note) if note else ''}

FEATURED ARTICLES:
{featured_block}

SUPPORTING ARTICLES:
{supporting_block}

OUTPUT STRUCTURE — use this exact Markdown structure:

# {title}
**Endometriosis Newsletter**
*Published {today}*

---

## Executive Summary
[2-3 sentences on what the featured article(s) show and why readers should care.]

---

## {feature_heading}
[{feature_instr} Cite with [N].]

---

## Key Developments
For each supporting article, one standalone block, separated by blank lines:

**[Bolded one-line headline.]** [N] Two to three sentences: what it reports and how it \
relates to the featured story.

*→ [Descriptive link text](URL)*

---

## Why It Matters
[1-2 paragraphs on clinical, scientific, or strategic significance.]

---

## What to Watch
[3-5 concrete follow-ups grounded in the articles.]

---

## Sources
[Every article cited inline: [N] Title. Source. Date. URL]
"""


def _generate(
    deps: Dict[str, Any], user_msg: str, system: str = _NB2_SYSTEM, max_tokens=4000
) -> str:
    deploy = deps["chat_deployment"]
    client = deps["make_chat_client"](deploy)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = client.responses.create(
            model=deploy, input=messages, max_output_tokens=max_tokens
        )
        return (resp.output_text or "").strip()
    except Exception:
        resp = client.chat.completions.create(
            model=deploy, messages=messages, max_completion_tokens=max_tokens
        )
        return (resp.choices[0].message.content or "").strip()


# ============================================================
# Why each similar article is related
# ============================================================

_REASON_SYSTEM = """\
You explain to a medical newsletter editor why a candidate article is related to a \
featured endometriosis article. Be specific: name the shared condition subtype, \
intervention, biomarker, method, population or finding. If the link is weak or only \
general, say so plainly. Use only the titles and summaries given. Reply with JSON only."""

_STOPWORDS = set(
    "a an and are as at be by for from in into is it its of on or that the their this "
    "to with without via vs versus using use study studies patients women endometriosis "
    "new among based between during after before review analysis".split()
)


def _keyword_reason(cand: Dict[str, Any], base: List[Dict[str, Any]]) -> str:
    """Fallback when the model is unavailable: shared title/summary terms."""

    def terms(a: Dict[str, Any]) -> Set[str]:
        words = re.findall(
            r"[a-z][a-z0-9\-]{3,}", (a["title"] + " " + a.get("snippet", "")).lower()
        )
        return {w for w in words if w not in _STOPWORDS}

    ct = terms(cand)
    best: Tuple[int, List[str]] = (0, [])
    for b in base:
        if b["ref"] in cand.get("matched", []):
            shared = sorted(ct & terms(b), key=len, reverse=True)[:4]
            if len(shared) > best[0]:
                best = (len(shared), shared)
    if best[1]:
        return "Shares key terms with the featured article: " + ", ".join(best[1]) + "."
    return "Similar overall topic to the featured article."


def _explain_matches(
    deps: Dict[str, Any], base: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> None:
    """Add a short 'reason' to each result explaining how it relates to its featured article."""
    featured = "\n".join(
        f"[{b['ref']}] {b['title']} — {(b.get('snippet') or '')[:300]}" for b in base
    )
    for start in range(0, len(results), 25):
        chunk = results[start : start + 25]
        cands = "\n".join(
            f"C{i} (similar to {', '.join(f'[{m}]' for m in r.get('matched', []))}): "
            f"{r['title']} — {(r.get('snippet') or '')[:300]}"
            for i, r in enumerate(chunk)
        )
        prompt = (
            f"FEATURED ARTICLES:\n{featured}\n\nCANDIDATES:\n{cands}\n\n"
            'Return a JSON object mapping each candidate id ("C0", "C1", ...) to one '
            "sentence of at most 25 words saying how it relates to the featured article "
            "it is similar to."
        )
        reasons: Dict[str, str] = {}
        try:
            text = _generate(deps, prompt, system=_REASON_SYSTEM, max_tokens=2500)
            m = re.search(r"\{.*\}", text, re.S)
            reasons = json.loads(m.group(0)) if m else {}
        except Exception:
            reasons = {}
        for i, r in enumerate(chunk):
            reason = str(reasons.get(f"C{i}", "")).strip()
            r["reason"] = reason or _keyword_reason(r, base)
            r["reason_source"] = "model" if reason else "keywords"


# ============================================================
# UI
# ============================================================

_DOMAIN_BADGES = {
    "pubmed": ("PubMed", "badge-pubmed"),
    "ncbi.nlm.nih.gov": ("PubMed", "badge-pubmed"),
    "europepmc": ("PubMed", "badge-pubmed"),
    "clinicaltrials": ("ClinicalTrials", "badge-ctgov"),
    "ctgov": ("ClinicalTrials", "badge-ctgov"),
    "medrxiv": ("medRxiv", "badge-medrxiv"),
    "biorxiv": ("bioRxiv", "badge-biorxiv"),
    "semantic": ("Semantic Scholar", "badge-semantic_scholar"),
    "openalex": ("OpenAlex", "badge-openalex"),
    "crossref": ("Crossref", "badge-crossref"),
    "doi.org": ("Crossref", "badge-crossref"),
    "nih_reporter": ("NIH Reporter", "badge-nih_reporter"),
    "reporter.nih": ("NIH Reporter", "badge-nih_reporter"),
    "fda": ("FDA", "badge-default"),
    "ema": ("EMA", "badge-default"),
}


def _badge(source: str) -> Tuple[str, str]:
    s = (source or "").lower()
    for needle, val in _DOMAIN_BADGES.items():
        if needle in s:
            return val
    return ("Web", "badge-web_search")


def _esc(s: str) -> str:
    return html_lib.escape(str(s or ""), quote=True)


def _card_html(a: Dict[str, Any], right_label: str, extra_html: str = "") -> str:
    label, cls = _badge(a.get("source", ""))
    url = _esc(a.get("url", ""))
    title = _esc(a["title"])
    title_html = (
        f'<a href="{url}" target="_blank" rel="noopener">{title}</a>' if url else title
    )
    meta = " &nbsp;·&nbsp; ".join(
        _esc(x)
        for x in [a.get("published_date") or "date unknown", a.get("source")]
        if x
    )
    excerpt = _esc((a.get("snippet") or "").replace("\n", " ")[:500])
    return (
        f'<div><span class="endo-badge {cls}">{label}</span>'
        f'<span class="endo-quality">{right_label}</span></div>'
        f'<div class="endo-title">{title_html}</div>'
        f'<div class="endo-meta">{meta}</div>'
        f"{extra_html}"
        f'<div class="endo-excerpt nb2-excerpt">{excerpt}</div>'
    )


def _css(flyout_open: bool, selected_keys: List[str]) -> str:
    tile_sel = "".join(
        f".st-key-nb2_tile_{k} {{ border:2px solid #1F6FEB !important; "
        f"background:#f4f8ff !important; }}"
        for k in selected_keys
    )
    pad = (
        """
        @media (min-width: 1280px) {
          [data-baseweb="tab-panel"]:has(.st-key-nb2_flyout) { padding-right: 470px; }
        }"""
        if flyout_open
        else ""
    )
    return f"""<style>
      [class*="st-key-nb2_tile_"], [class*="st-key-nb2_res_"] {{
        background:#ffffff; border:1px solid #e4e9f0; border-radius:10px;
        padding:14px 16px 6px 16px; box-shadow:0 2px 6px rgba(0,0,0,0.05);
        gap: 0.25rem;
      }}
      [class*="st-key-nb2_tile_"]:hover {{ box-shadow:0 4px 16px rgba(0,0,0,0.10); }}
      .nb2-excerpt {{ -webkit-line-clamp:3 !important; margin-bottom:4px !important; }}
      .nb2-reason {{ font-size:0.84rem; color:#0d2137; background:#eef5ff;
        border-left:3px solid #1F6FEB; border-radius:4px; padding:6px 9px; margin:6px 0 8px 0; }}
      .st-key-nb2_flyout {{
        position:fixed; top:60px; right:0; bottom:52px; width:min(450px, 100vw);
        z-index:999; background:#f8faff; border-left:1px solid #dde4ef;
        box-shadow:-8px 0 24px rgba(0,0,0,0.12); padding:16px 18px;
        overflow-y:auto; display:block;
      }}
      .st-key-nb2_flyout [class*="st-key-nb2_res_"] {{ margin-bottom:10px; }}
      {tile_sel}{pad}
    </style>"""


def _on_toggle_supporting(article: Dict[str, Any], cb_key: str) -> None:
    sup = st.session_state[_k("supporting")]
    if st.session_state.get(cb_key):
        sup[article["key"]] = article
    else:
        sup.pop(article["key"], None)


def _set_flyout(open_: bool) -> None:
    st.session_state[_k("flyout_open")] = open_


def _run_search(
    base: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    opts: Dict[str, Any],
    deps: Dict[str, Any],
) -> None:
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    where = opts["where"]
    with st.spinner("Finding similar articles…"):
        if where in ("Evidence library + Web", "Evidence library"):
            try:
                results += _library_search(
                    base, opts["per_article"], opts["keywords"], deps
                )
            except Exception as e:
                errors.append(f"Evidence library: {e}")
        if where in ("Evidence library + Web", "Web"):
            try:
                briefing_urls = {_url_key(a["url"]) for a in articles}
                web, werrs = _web_search(
                    base,
                    opts["per_article"],
                    opts["keywords"],
                    opts["web_days"],
                    briefing_urls,
                )
                results += web
                errors += [f"Web: {x}" for x in werrs]
            except Exception as e:
                errors.append(f"Web: {e}")
    if results:
        with st.spinner(f"Explaining how {len(results)} articles are related…"):
            _explain_matches(deps, base, results)
    st.session_state[_k("results")] = results
    st.session_state[_k("results_sig")] = opts["sig"]
    st.session_state[_k("search_errors")] = errors
    st.session_state[_k("flyout_open")] = True


def _render_flyout(base: List[Dict[str, Any]], sig: str) -> None:
    results = st.session_state[_k("results")]
    res_sig = st.session_state[_k("results_sig")]
    with st.container(key="nb2_flyout"):
        h1, h2 = st.columns([0.8, 0.2], vertical_alignment="center")
        with h1:
            st.markdown(f"#### Similar articles ({len(results)})")
        with h2:
            st.button(
                "✕",
                key=_k("close_flyout"),
                on_click=_set_flyout,
                args=(False,),
                help="Close",
            )
        st.caption(
            "Similar to: " + "; ".join(f"[{b['ref']}] {b['title']}" for b in base)
        )
        if res_sig != sig:
            st.info(
                "Your featured articles or search options changed. Click "
                "**Find similar articles** to refresh."
            )
        for err in st.session_state[_k("search_errors")]:
            st.warning(err)
        if not results:
            st.info("No similar articles found. Try different keywords or sources.")
            return

        origins = sorted({r["origin"] for r in results})
        show = "All"
        if len(origins) > 1:
            show = st.radio(
                "Show",
                ["All", "Evidence library", "Web"],
                horizontal=True,
                key=_k("origin_filter"),
                label_visibility="collapsed",
            )
        view = [
            r
            for r in results
            if show == "All"
            or (r["origin"] == "library") == (show == "Evidence library")
        ]
        n_sup = len(st.session_state[_k("supporting")])
        st.caption(f"Tick to add as supporting · {n_sup} selected")

        for r in view:
            cb_key = _k(f"sup_cb_{r['key']}")
            if cb_key not in st.session_state:
                st.session_state[cb_key] = (
                    r["key"] in st.session_state[_k("supporting")]
                )
            right = (
                f"Similarity {r['score']:.0%}"
                if r.get("score") is not None
                else f"Web result #{r.get('rank', '')}"
            )
            reason = _esc(r.get("reason", ""))
            matched = ", ".join(f"[{m}]" for m in r.get("matched", []))
            reason_html = (
                f'<div class="nb2-reason"><b>Why it\'s related to {matched}:</b> {reason}</div>'
                if reason
                else ""
            )
            with st.container(key=f"nb2_res_{r['key']}"):
                st.markdown(_card_html(r, right, reason_html), unsafe_allow_html=True)
                st.checkbox(
                    "Add as supporting article",
                    key=cb_key,
                    on_change=_on_toggle_supporting,
                    args=(r, cb_key),
                )


def render_tab(deps: Dict[str, Any]) -> None:
    """
    deps keys:
      load_ai_runs(limit) -> list of runs (newest first)
      load_docs_df(), load_sqlite_vectors(), embed_text(str), cosine_search(...)
      make_chat_client(deployment), chat_deployment
      save_newsletter(...), update_newsletter(id, md), render_newsletter_html(...)
      html_component(html, height, scrolling)
    """
    _init_state()
    if _k("flyout_open") not in st.session_state:
        st.session_state[_k("flyout_open")] = False

    runs = deps["load_ai_runs"](50)
    if not runs:
        st.info(
            "No Intelligence Briefings yet. Run one in the AI Article Search tab first."
        )
        return

    # ── Top bar: briefing picker · Find similar · search options ──
    run_by_id = {r["id"]: r for r in runs}
    top1, top2, top3, top4 = st.columns(
        [3.2, 1.5, 0.9, 0.5], vertical_alignment="bottom"
    )
    with top1:
        run_id = st.selectbox(
            "Intelligence Briefing",
            options=list(run_by_id.keys()),
            index=0,
            format_func=lambda rid: _run_label(run_by_id[rid])
            + ("  (most recent)" if rid == runs[0]["id"] else ""),
            key=_k("run_id"),
        )
    run = run_by_id[run_id]
    if st.session_state[_k("active_run")] != run_id:
        # New briefing → clear old results; keep chosen supporting articles
        st.session_state[_k("active_run")] = run_id
        st.session_state[_k("results")] = []
        st.session_state[_k("results_sig")] = None
        st.session_state[_k("flyout_open")] = False

    articles = _briefing_articles(run)
    base_key = lambda a: _k(f"base_{a['key']}")  # noqa: E731
    base = [a for a in articles if st.session_state.get(base_key(a))]
    n_base = len(base)

    with top4:
        with st.popover("⚙️", help="Search options"):
            where = st.radio(
                "Search in",
                ["Evidence library + Web", "Evidence library", "Web"],
                key=_k("where"),
                help="Evidence library = the app's ingested corpus (PubMed, "
                "ClinicalTrials.gov, bioRxiv, FDA, …) searched by meaning. Web = the "
                "same search provider as AI Article Search.",
            )
            per_article = st.slider(
                "Results per featured article", 5, 20, 10, key=_k("per_article")
            )
            web_days = st.selectbox(
                "Web recency",
                [30, 90, 365, 1825],
                index=2,
                format_func=lambda d: {30: "30 days", 90: "90 days", 365: "1 year"}.get(
                    d, "5 years"
                ),
                key=_k("web_days"),
                disabled=where == "Evidence library",
            )
            keywords = st.text_input(
                "Keywords to steer the search",
                key=_k("keywords"),
                placeholder="e.g. adolescents, biomarkers",
            )
    sig = _hash(
        "|".join(sorted(b["key"] for b in base))
        + f"|{where}|{per_article}|{web_days}|{keywords}"
    )
    with top2:
        find = st.button(
            f"🔎 Find similar articles ({n_base})",
            type="primary",
            disabled=n_base == 0,
            use_container_width=True,
            key=_k("search_btn"),
            help="Select 1–2 articles below first" if n_base == 0 else None,
        )
    with top3:
        n_res = len(st.session_state[_k("results")])
        if n_res and not st.session_state[_k("flyout_open")]:
            st.button(
                f"Show results ({n_res})",
                key=_k("show_flyout"),
                on_click=_set_flyout,
                args=(True,),
                use_container_width=True,
            )

    if find and base:
        _run_search(
            base,
            articles,
            {
                "where": where,
                "per_article": per_article,
                "web_days": web_days,
                "keywords": keywords,
                "sig": sig,
            },
            deps,
        )

    flyout_open = bool(st.session_state[_k("flyout_open")]) and bool(base)
    st.markdown(_css(flyout_open, [a["key"] for a in base]), unsafe_allow_html=True)

    # ── Article tiles ──────────────────────────────────────────
    if not articles:
        st.warning("This briefing has no stored articles.")
        return
    c1, c2, c3 = st.columns([2.2, 1.6, 1], vertical_alignment="center")
    with c1:
        st.caption(
            f"**Choose up to {MAX_BASE_ARTICLES} articles to feature** · {n_base} selected · "
            f"{sum(a['cited'] for a in articles)} of {len(articles)} cited in the briefing text"
        )
    with c2:
        only_cited = st.toggle(
            "Only articles cited in the briefing", value=False, key=_k("only_cited")
        )
    with c3:
        with st.popover("View briefing text", use_container_width=True):
            st.markdown(run.get("briefing_text") or "_(empty)_")

    shown = [
        a
        for a in articles
        if a["cited"] or not only_cited or st.session_state.get(base_key(a))
    ]
    n_cols = 2 if flyout_open else 3
    for row_start in range(0, len(shown), n_cols):
        cols = st.columns(n_cols, gap="small")
        for col, a in zip(cols, shown[row_start : row_start + n_cols]):
            checked = bool(st.session_state.get(base_key(a)))
            right = f"[{a['ref']}]" + (" · Cited" if a["cited"] else "")
            if a.get("score") is not None:
                right += f" · Score {a['score']:.0%}"
            with col:
                with st.container(key=f"nb2_tile_{a['key']}"):
                    st.markdown(_card_html(a, right), unsafe_allow_html=True)
                    st.checkbox(
                        "Feature this article",
                        key=base_key(a),
                        disabled=(not checked and n_base >= MAX_BASE_ARTICLES),
                    )

    if flyout_open:
        _render_flyout(base, sig)

    if not base:
        return

    # ── Supporting articles + build ────────────────────────────
    supporting: List[Dict[str, Any]] = list(st.session_state[_k("supporting")].values())
    st.markdown("---")
    st.markdown(f"#### Supporting articles ({len(supporting)})")
    if supporting:
        for a in supporting:
            s1, s2 = st.columns([0.9, 0.1], vertical_alignment="center")
            with s1:
                st.markdown(_article_header_md(a), unsafe_allow_html=True)
            with s2:
                if st.button("Remove", key=_k(f"rm_{a['key']}")):
                    st.session_state[_k("supporting")].pop(a["key"], None)
                    st.session_state.pop(_k(f"sup_cb_{a['key']}"), None)
                    st.rerun()
    else:
        st.caption(
            "None yet: use **Find similar articles** at the top and tick the ones to "
            "include. You can also generate with the featured article(s) alone."
        )

    st.markdown("#### Build the newsletter")
    g1, g2, g3 = st.columns([1.2, 1.6, 0.9], vertical_alignment="bottom")
    with g1:
        nl_title = st.text_input(
            "Newsletter title", value="Endometriosis Newsletter", key=_k("title")
        )
    with g2:
        nl_note = st.text_input(
            "Editorial note (optional)",
            key=_k("note"),
            placeholder="e.g. audience is clinicians; keep it under 600 words",
        )
    with g3:
        gen = st.button(
            "📰 Generate Newsletter",
            type="primary",
            key=_k("gen_btn"),
            use_container_width=True,
        )

    if gen:
        prompt = _build_prompt(base, supporting, nl_title, _run_label(run), nl_note)
        with st.spinner("Writing newsletter…"):
            try:
                text = _generate(deps, prompt)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                text = ""
        if text:
            sources = [
                {
                    "role": "featured" if i < len(base) else "supporting",
                    "origin": a["origin"],
                    "title": a["title"],
                    "url": a["url"],
                    "source": a.get("source", ""),
                    "published_date": a.get("published_date", ""),
                    "reason": a.get("reason", ""),
                }
                for i, a in enumerate(base + supporting)
            ]
            sid = deps["save_newsletter"](
                title=nl_title,
                coverage_period=f"Seeded from Intelligence Briefing #{run_id}",
                content_md=text,
                sources=sources,
                config={
                    "workflow": "briefing_seeded_v2",
                    "briefing_run_id": run_id,
                    "featured_refs": [b["ref"] for b in base],
                    "editorial_note": nl_note,
                },
            )
            st.session_state[_k("draft")] = text
            st.session_state[_k("saved_id")] = sid
            st.session_state[_k("editor")] = text
            st.session_state[_k("flyout_open")] = False
            st.rerun()

    draft = st.session_state[_k("draft")]
    if not draft:
        return

    sid = st.session_state[_k("saved_id")]
    st.success(
        f"Draft saved as Newsletter #{sid}. It also appears in the original Newsletter "
        "Builder's saved list, where it can be sent or scheduled."
    )
    t_edit, t_prev = st.tabs(["✏️ Edit", "👁 Preview"])
    with t_edit:
        if _k("editor") not in st.session_state:
            st.session_state[_k("editor")] = draft
        edited = st.text_area("Newsletter Markdown", height=600, key=_k("editor"))
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button(
                "💾 Save changes", key=_k("save_btn"), disabled=edited == draft
            ):
                deps["update_newsletter"](sid, edited)
                st.session_state[_k("draft")] = edited
                st.rerun()
        with b2:
            st.download_button(
                "⬇️ Download Markdown",
                data=edited.encode("utf-8"),
                file_name=f"newsletter_{sid}.md",
                mime="text/markdown",
                key=_k("dl_md"),
            )
    with t_prev:
        html = deps["render_newsletter_html"](
            nl_title,
            f"Endometriosis Newsletter · Seeded from Intelligence Briefing #{run_id}",
            st.session_state[_k("draft")],
        )
        deps["html_component"](html, height=900, scrolling=True)
