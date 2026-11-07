"""
newsletter_from_briefing.py — "Newsletter Builder v2" tab.

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


def _generate(deps: Dict[str, Any], user_msg: str) -> str:
    deploy = deps["chat_deployment"]
    client = deps["make_chat_client"](deploy)
    messages = [
        {"role": "system", "content": _NB2_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = client.responses.create(
            model=deploy, input=messages, max_output_tokens=4000
        )
        return (resp.output_text or "").strip()
    except Exception:
        resp = client.chat.completions.create(
            model=deploy, messages=messages, max_completion_tokens=4000
        )
        return (resp.choices[0].message.content or "").strip()


# ============================================================
# UI sections
# ============================================================


def _on_toggle_supporting(article: Dict[str, Any], cb_key: str) -> None:
    sup = st.session_state[_k("supporting")]
    if st.session_state.get(cb_key):
        sup[article["key"]] = article
    else:
        sup.pop(article["key"], None)


def _render_result(a: Dict[str, Any], sig: str) -> None:
    cb_key = _k(f"sup_cb_{sig}_{a['key']}")
    if cb_key not in st.session_state:
        st.session_state[cb_key] = a["key"] in st.session_state[_k("supporting")]
    with st.container(border=True):
        c1, c2 = st.columns([0.06, 0.94])
        with c1:
            st.checkbox(
                "Include",
                key=cb_key,
                label_visibility="collapsed",
                on_change=_on_toggle_supporting,
                args=(a, cb_key),
            )
        with c2:
            st.markdown(_article_header_md(a), unsafe_allow_html=True)
            tags = [
                "📚 Evidence library" if a["origin"] == "library" else "🌐 Web",
                "Similar to " + ", ".join(f"[{m}]" for m in a.get("matched", [])),
            ]
            if a.get("score") is not None:
                tags.append(f"similarity {a['score']:.2f}")
            if a.get("in_briefing"):
                tags.append("also in this briefing")
            st.caption(" · ".join(tags))
            if a.get("snippet"):
                snip = a["snippet"]
                st.write(snip[:400] + ("…" if len(snip) > 400 else ""))


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

    st.markdown("### 🆕 Newsletter Builder v2")
    st.caption(
        "Start a newsletter from a past Intelligence Briefing: pick one or two of its "
        "articles, find similar articles, and choose supporting ones. The "
        "Intelligence Briefing itself is not changed."
    )

    runs = deps["load_ai_runs"](50)
    if not runs:
        st.info(
            "No Intelligence Briefings yet. Run one in the AI Article Search tab first."
        )
        return

    # ── Step 1: pick briefing ──────────────────────────────────
    st.markdown("#### 1 · Intelligence Briefing")
    run_by_id = {r["id"]: r for r in runs}
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
        # New briefing → reset similar-article results; keep chosen supporting articles
        st.session_state[_k("active_run")] = run_id
        st.session_state[_k("results")] = []
        st.session_state[_k("results_sig")] = None

    with st.expander("View briefing text"):
        st.markdown(run.get("briefing_text") or "_(empty)_")

    # ── Step 2: pick 1–2 base articles ─────────────────────────
    articles = _briefing_articles(run)
    st.markdown("#### 2 · Choose the featured article(s)")
    if not articles:
        st.warning("This briefing has no stored articles.")
        return

    only_cited = st.toggle(
        "Only show articles cited in the briefing text",
        value=False,
        key=_k("only_cited"),
    )
    base_key = lambda a: _k(f"base_{a['key']}")  # noqa: E731
    # Keep already-selected articles visible so the filter never drops a selection
    shown = [
        a
        for a in articles
        if a["cited"] or not only_cited or st.session_state.get(base_key(a))
    ]
    n_base = sum(1 for a in articles if st.session_state.get(base_key(a)))
    st.caption(
        f"Select up to {MAX_BASE_ARTICLES}. {n_base} selected · "
        f"{sum(a['cited'] for a in articles)} of {len(articles)} articles are cited "
        "in the briefing text."
    )

    for a in shown:
        checked = bool(st.session_state.get(base_key(a)))
        with st.container(border=True):
            c1, c2 = st.columns([0.06, 0.94])
            with c1:
                st.checkbox(
                    "Feature",
                    key=base_key(a),
                    label_visibility="collapsed",
                    disabled=(not checked and n_base >= MAX_BASE_ARTICLES),
                )
            with c2:
                st.markdown(
                    f"`[{a['ref']}]` " + _article_header_md(a), unsafe_allow_html=True
                )
                tags = ["✅ Cited in briefing" if a["cited"] else "Not cited in text"]
                if a.get("score") is not None:
                    tags.append(f"briefing score {a['score']:.2f}")
                st.caption(" · ".join(tags))
                if a["snippet"]:
                    st.write(a["snippet"])

    base = [a for a in articles if st.session_state.get(base_key(a))]
    if not base:
        st.info("Select one or two articles above to continue.")
        return

    # ── Step 3: find similar articles ──────────────────────────
    st.markdown("#### 3 · Find similar articles")
    st.markdown(
        "Featured: " + "; ".join(f"**[{b['ref']}]** {b['title']}" for b in base)
    )

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        where = st.radio(
            "Search in",
            ["Evidence library + Web", "Evidence library", "Web"],
            horizontal=True,
            key=_k("where"),
            help="Evidence library = the app's ingested corpus (PubMed, ClinicalTrials.gov, "
            "bioRxiv, FDA, …) searched by meaning. Web = the same search provider as AI "
            "Article Search.",
        )
    with c2:
        per_article = st.slider(
            "Results per featured article", 5, 20, 10, key=_k("per_article")
        )
    with c3:
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
        "Optional keywords to steer the search",
        key=_k("keywords"),
        placeholder="e.g. adolescents, biomarkers, phase 3",
    )

    sig = _hash(
        "|".join(sorted(b["key"] for b in base))
        + f"|{where}|{per_article}|{web_days}|{keywords}"
    )
    if st.button("🔎 Find similar articles", type="primary", key=_k("search_btn")):
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        with st.spinner("Searching…"):
            if where in ("Evidence library + Web", "Evidence library"):
                try:
                    results += _library_search(base, per_article, keywords, deps)
                except Exception as e:
                    errors.append(f"Evidence library: {e}")
            if where in ("Evidence library + Web", "Web"):
                try:
                    briefing_urls = {_url_key(a["url"]) for a in articles}
                    web, werrs = _web_search(
                        base, per_article, keywords, web_days, briefing_urls
                    )
                    results += web
                    errors += [f"Web: {x}" for x in werrs]
                except Exception as e:
                    errors.append(f"Web: {e}")
        st.session_state[_k("results")] = results
        st.session_state[_k("results_sig")] = sig
        st.session_state[_k("search_errors")] = errors

    for err in st.session_state[_k("search_errors")]:
        st.warning(err)

    results = st.session_state[_k("results")]
    res_sig = st.session_state[_k("results_sig")]
    if res_sig and res_sig != sig:
        st.caption(
            "Results below are from your previous search settings — "
            "click Find similar articles to refresh."
        )
    if results:
        origins = sorted({r["origin"] for r in results})
        f1, f2 = st.columns([1, 1])
        with f1:
            origin_filter = st.radio(
                "Show",
                ["All"]
                + [("Evidence library" if o == "library" else "Web") for o in origins],
                horizontal=True,
                key=_k("origin_filter"),
            )
        with f2:
            text_filter = st.text_input(
                "Filter results",
                key=_k("text_filter"),
                placeholder="word in title or summary",
            )
        view = results
        if origin_filter == "Evidence library":
            view = [r for r in view if r["origin"] == "library"]
        elif origin_filter == "Web":
            view = [r for r in view if r["origin"] == "web"]
        if text_filter:
            tf = text_filter.lower()
            view = [
                r
                for r in view
                if tf in (r["title"] + " " + r.get("snippet", "")).lower()
            ]
        st.caption(f"{len(view)} of {len(results)} results · tick to add as supporting")
        for r in view:
            _render_result(r, res_sig or sig)
    elif res_sig:
        st.info("No similar articles found. Try different keywords or sources.")

    # ── Selected supporting articles ───────────────────────────
    supporting: List[Dict[str, Any]] = list(st.session_state[_k("supporting")].values())
    st.markdown(f"#### 4 · Supporting articles ({len(supporting)} selected)")
    if supporting:
        for a in supporting:
            s1, s2 = st.columns([0.88, 0.12])
            with s1:
                st.markdown(_article_header_md(a), unsafe_allow_html=True)
            with s2:
                if st.button("Remove", key=_k(f"rm_{a['key']}")):
                    st.session_state[_k("supporting")].pop(a["key"], None)
                    for k in list(st.session_state.keys()):
                        if k.startswith(_k("sup_cb_")) and k.endswith(a["key"]):
                            del st.session_state[k]
                    st.rerun()
    else:
        st.caption(
            "None yet. You can also generate with the featured article(s) alone."
        )

    # ── Step 5: generate newsletter ────────────────────────────
    st.markdown("#### 5 · Build the newsletter")
    g1, g2 = st.columns([1, 1])
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

    if st.button("📰 Generate Newsletter", type="primary", key=_k("gen_btn")):
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
