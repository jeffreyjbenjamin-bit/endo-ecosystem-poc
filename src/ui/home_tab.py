"""Home tab: a summary of everything else in the app.

- Intelligence Summary and Key Developments from the most recent briefing
- At a glance: latest briefing, clinical trials, latest newsletter
- AI trends for the past week, month and year (src.pipelines.trends)
- Most recent search results as cards with preview images
- The 5 most recently posted open clinical trials
"""

from __future__ import annotations

import re
from datetime import datetime
from html import escape
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

_CSS = """
<style>
.home-sub { font-size:13px; color:#7f8c8d; margin:-6px 0 14px 0; }
.home-h { font-size:17px; font-weight:700; color:#1a2b3c; margin:4px 0 8px 0; }
.home-summary { font-size:15px; line-height:1.55; color:#1f2d3d; }
.home-stat { border:1px solid #e3e6ea; border-radius:10px; padding:10px 14px; margin-bottom:10px;
  background:#fbfcfd; }
.home-stat .k { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#7f8c8d; }
.home-stat .v { font-size:22px; font-weight:700; color:#1a2b3c; line-height:1.2; }
.home-stat .d { font-size:12px; color:#566573; margin-top:2px; }
.home-trend-head { font-size:13px; color:#566573; font-style:italic; margin-bottom:8px; min-height:2.6em; }
.home-trend { margin-bottom:10px; }
.home-trend b { color:#1a2b3c; }
.home-trend .det { font-size:13px; color:#34495e; line-height:1.45; }
.home-chip { display:inline-block; font-size:10px; font-weight:700; text-transform:uppercase;
  padding:1px 6px; border-radius:4px; margin-left:6px; vertical-align:1px; }
.home-chip.rising { background:#e8f6ee; color:#1e7e4a; }
.home-chip.new { background:#fff3cd; color:#8a6d00; }
.home-chip.steady { background:#eef2f7; color:#4a6078; }
.home-card { border:1px solid #e3e6ea; border-radius:10px; overflow:hidden; background:#fff;
  margin-bottom:14px; height:100%; }
.home-card .img { height:150px; background:#eef2f7 center/cover no-repeat; display:flex;
  align-items:center; justify-content:center; color:#9aa9b8; font-size:30px; font-weight:700; }
.home-card .img img { width:100%; height:150px; object-fit:cover; display:block; }
.home-card .body { padding:10px 12px 12px 12px; }
.home-card .meta { font-size:11px; color:#7f8c8d; margin:4px 0; }
.home-card .title a { font-size:14px; font-weight:600; color:#1a2b3c; text-decoration:none;
  line-height:1.35; }
.home-card .title a:hover { text-decoration:underline; }
.home-card .snip { font-size:12.5px; color:#566573; line-height:1.45; }
.home-badge { display:inline-block; font-size:10px; font-weight:700; padding:1px 6px; border-radius:4px;
  background:#e8f1fb; color:#1a5c99; }
.home-badge.press { background:#fde8f1; color:#9b1d5a; }
.home-trials { width:100%; border-collapse:collapse; font-size:13px; }
.home-trials th { text-align:left; color:#566573; font-weight:600; padding:6px 8px;
  border-bottom:2px solid #e3e6ea; white-space:nowrap; }
.home-trials td { padding:7px 8px; border-bottom:1px solid #eef0f2; vertical-align:top; }
.home-trials .sub { font-size:11px; color:#7f8c8d; }
</style>
"""

_CITE_RE = re.compile(
    r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\](?:\s*[,;]?\s*\[\d+(?:\s*[,–-]\s*\d+)*\])*"
)


def split_sections(md: str) -> Dict[str, str]:
    """{section heading: body} for a briefing's '## ' sections."""
    out: Dict[str, str] = {}
    for m in re.finditer(r"^##\s+(.+?)\s*$(.*?)(?=^##\s|\Z)", md or "", re.M | re.S):
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def strip_citations(text: str) -> str:
    return _CITE_RE.sub("", text)


def key_development_blocks(body: str) -> List[str]:
    """Key Developments entries; a '*→ link*' line stays with its entry."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body or "") if b.strip()]
    out: List[str] = []
    for b in blocks:
        if out and b.lstrip("*_ ").startswith("→"):
            out[-1] += "\n\n" + b
        else:
            out.append(b)
    return out


def _fmt_date(value: Any) -> str:
    s = str(value or "")[:10]
    try:
        return datetime.fromisoformat(s).strftime("%b %d, %Y").replace(" 0", " ")
    except ValueError:
        return s


def recent_results(
    runs: List[Dict[str, Any]], n: int = 6, sources_of: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """Newest results across the latest runs, newest publication date first."""
    seen, out = set(), []
    for run in runs:
        for s in sources_of(run) if sources_of else []:
            url = (s.get("url") or "").split("?")[0].rstrip("/").lower()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(s)
        if len(out) >= n * 3:
            break
    out.sort(key=lambda s: str(s.get("published_date") or "0000"), reverse=True)
    return out[:n]


def _result_card(s: Dict[str, Any], image: str) -> str:
    url = escape(s.get("url") or "", quote=True)
    title = escape(re.sub(r"\s+", " ", s.get("title") or "Untitled"))
    source = escape(s.get("source_name") or "")
    initial = escape((s.get("source_name") or "?")[:1].upper())
    img = (
        f"<div class='img'><img src='{escape(image, quote=True)}' alt='' loading='lazy' "
        "referrerpolicy='no-referrer' onerror=\"this.parentNode.textContent='"
        f"{initial}'\"></div>"
        if image
        else f"<div class='img'>{initial}</div>"
    )
    badge = (
        "<span class='home-badge press'>Press Release</span>"
        if s.get("is_press_release")
        else f"<span class='home-badge'>{source or 'Web'}</span>"
    )
    date = _fmt_date(s.get("published_date")) or "date unknown"
    snip = escape((s.get("snippet") or "").replace("\n", " ")[:190])
    return (
        f"<div class='home-card'>{img}<div class='body'>{badge}"
        f"<div class='meta'>{date}</div>"
        f"<div class='title'><a href='{url}' target='_blank' rel='noopener'>{title}</a></div>"
        f"<div class='snip'>{snip}</div></div></div>"
    )


def _stat(label: str, value: str, detail: str = "") -> str:
    return (
        f"<div class='home-stat'><div class='k'>{escape(label)}</div>"
        f"<div class='v'>{escape(value)}</div>"
        + (f"<div class='d'>{detail}</div>" if detail else "")
        + "</div>"
    )


def _trials_table(trials: List[Dict[str, Any]]) -> str:
    rows = []
    for t in trials:
        new = " <span class='home-chip new'>New</span>" if t.get("is_new") else ""
        rows.append(
            "<tr>"
            f"<td><a href='{escape(t['url'], quote=True)}' target='_blank' rel='noopener'>"
            f"<b>{escape(t['id'])}</b></a><div class='sub'>{escape(t.get('source', ''))}</div></td>"
            f"<td>{escape(t['title'])}{new}</td>"
            f"<td>{escape(t['lead_sponsor'])}<div class='sub'>{escape(t['sponsor_type'])}</div></td>"
            f"<td>{escape(t['study_phase'])}</td>"
            f"<td>{escape(t['status'])}</td>"
            f"<td style='white-space:nowrap'>{escape(_fmt_date(t.get('first_posted')))}</td>"
            "</tr>"
        )
    return (
        "<table class='home-trials'><thead><tr><th>ID</th><th>Title</th><th>Lead Sponsor</th>"
        "<th>Study / Phase</th><th>Status</th><th>First Posted</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _images(urls: tuple) -> Dict[str, str]:
    from src.common.page_images import images_for

    return images_for(list(urls))


def _render_trends() -> None:
    from src.pipelines.trends import WINDOWS, generate_trends, load_trends

    with st.container(border=True, key="home_trends"):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        head_l.markdown(
            "<div class='home-h'>✨ AI trend analysis</div>", unsafe_allow_html=True
        )
        refresh = head_r.button(
            "🔄 Refresh analysis",
            key="home_trends_refresh",
            use_container_width=True,
            help="Re-analyze everything collected (briefings, evidence library and "
            "clinical trials). Takes about 15–30 seconds. Also runs after each "
            "daily briefing.",
        )
        if refresh:
            with st.spinner("Analyzing trends…"):
                try:
                    generate_trends()
                except Exception as exc:
                    st.error(f"Trend analysis failed: {exc}")
        data = load_trends()
        if not data:
            st.caption(
                "No analysis yet. Click **Refresh analysis** to analyze the top trends "
                "of the past week, month and year."
            )
            return
        cols = st.columns(3, gap="medium")
        for col, (key, label, _days) in zip(cols, WINDOWS):
            w = (data.get("windows") or {}).get(key) or {}
            n = (data.get("counts") or {}).get(key, 0)
            html = [
                f"<div class='home-h' style='font-size:15px'>{label} "
                f"<span style='font-weight:400;color:#7f8c8d;font-size:12px'>"
                f"· {n} items</span></div>",
                f"<div class='home-trend-head'>{escape(w.get('headline', ''))}</div>",
            ]
            for t in w.get("trends") or []:
                sig = t.get("signal", "")
                chip = (
                    f"<span class='home-chip {escape(sig)}'>{escape(sig)}</span>"
                    if sig in ("rising", "new", "steady")
                    else ""
                )
                html.append(
                    f"<div class='home-trend'><b>{escape(t['title'])}</b>{chip}"
                    f"<div class='det'>{escape(t.get('detail', ''))}</div></div>"
                )
            col.markdown("".join(html), unsafe_allow_html=True)
        st.caption(
            f"Analyzed {_fmt_date(data.get('generated_at'))} "
            f"{str(data.get('generated_at', ''))[11:16]} UTC by AI ({data.get('model', '')}) "
            "from Intelligence Briefing results, the evidence library and clinical "
            "trials. Check important points against the sources."
        )


def render_tab(deps: Dict[str, Any]) -> None:
    """
    deps:
      load_runs(limit) -> saved briefing runs, newest first
      sources_of(run) -> list of source dicts for a run
      load_newsletters(limit) -> newsletters, newest first
      open_newsletter(nid) -> on_click callback opening the editor
    """
    from src.pipelines.clinical_trials import load_trials

    st.markdown(_CSS, unsafe_allow_html=True)
    runs = deps["load_runs"](10)
    latest = runs[0] if runs else None

    sub = "Your endometriosis intelligence at a glance"
    if latest:
        sub += f" · latest briefing {_fmt_date(latest.get('run_ts'))}"
    st.markdown(f"<div class='home-sub'>{sub}</div>", unsafe_allow_html=True)

    # ── Summary + key developments | at a glance ─────────────────
    left, right = st.columns([2.1, 1], gap="large")
    with left:
        if not latest:
            st.info(
                "No Intelligence Briefing yet. Run one in **Build Intelligence Briefing**."
            )
        else:
            sections = split_sections(latest.get("briefing_text") or "")
            summary = sections.get("Intelligence Summary", "")
            with st.container(border=True, key="home_summary"):
                st.markdown(
                    "<div class='home-h'>Intelligence Summary</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        f"<div class='home-summary'>{escape(strip_citations(summary))}</div>"
                        if summary
                        else "_No summary in this briefing._"
                    ),
                    unsafe_allow_html=True,
                )
            devs = key_development_blocks(sections.get("Key Developments", ""))
            if devs:
                st.markdown(
                    "<div class='home-h'>Key Developments</div>", unsafe_allow_html=True
                )
                for b in devs:
                    with st.container(border=True):
                        st.markdown(strip_citations(b))

    trials_data = load_trials() or {}
    trials = trials_data.get("trials") or []
    with right:
        st.markdown("<div class='home-h'>At a glance</div>", unsafe_allow_html=True)
        if latest:
            n_src = latest.get("n_scored") or 0
            st.markdown(
                _stat(
                    "Latest Intelligence Briefing",
                    _fmt_date(latest.get("run_ts")),
                    f"Run #{latest['id']} · {n_src} sources · "
                    f"{len(runs)} recent briefings",
                ),
                unsafe_allow_html=True,
            )
        n_new = sum(1 for t in trials if t.get("is_new"))
        n_co = sum(1 for t in trials if t.get("is_company"))
        st.markdown(
            _stat(
                "Open US clinical trials",
                str(len(trials)) if trials else "—",
                (
                    f"{n_new} new · {n_co} company-sponsored"
                    if trials
                    else "Refresh the Clinical Trials tab to load"
                ),
            ),
            unsafe_allow_html=True,
        )
        nls = deps["load_newsletters"](1)
        if nls:
            nl = nls[0]
            st.markdown(
                _stat(
                    "Latest newsletter",
                    _fmt_date(nl.get("updated_ts") or nl.get("created_ts")),
                    escape(nl.get("title") or "Untitled"),
                ),
                unsafe_allow_html=True,
            )
            st.button(
                "📝 Open latest newsletter",
                key="home_open_nl",
                on_click=deps["open_newsletter"],
                args=(nl["id"],),
                use_container_width=True,
            )

    # ── AI trends ────────────────────────────────────────────────
    _render_trends()

    # ── Recent search results ───────────────────────────────────
    results = recent_results(runs, 6, deps["sources_of"])
    if results:
        st.markdown(
            "<div class='home-h' style='margin-top:10px'>Recent search results</div>",
            unsafe_allow_html=True,
        )
        need = tuple(s["url"] for s in results if not s.get("image"))
        imgs = _images(need) if need else {}
        for row_start in range(0, len(results), 3):
            cols = st.columns(3, gap="small")
            for col, s in zip(cols, results[row_start : row_start + 3]):
                img = s.get("image") or imgs.get(s.get("url", ""), "")
                col.markdown(_result_card(s, img), unsafe_allow_html=True)

    # ── Most recent clinical trials ─────────────────────────────
    st.markdown(
        "<div class='home-h' style='margin-top:10px'>Most recent clinical trials</div>",
        unsafe_allow_html=True,
    )
    if not trials:
        st.caption("No trials loaded yet. They load automatically each day.")
    else:
        newest = sorted(
            trials, key=lambda t: (t.get("first_posted") or "", t["id"]), reverse=True
        )[:5]
        st.markdown(_trials_table(newest), unsafe_allow_html=True)
        st.caption(
            f"The 5 most recently posted of {len(trials)} open endometriosis trials "
            "with a US site. See all in the Clinical Trials tab."
        )
