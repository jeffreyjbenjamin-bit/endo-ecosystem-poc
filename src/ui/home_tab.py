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
import streamlit.components.v1 as components

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
.home-trials { width:100%; border-collapse:collapse; font-size:12.5px; }
.home-dev { display:flex; gap:14px; padding:14px 16px; margin-bottom:12px; background:#f8faff;
  border:1px solid #e4e9f0; border-left:3px solid #1a7fb5; border-radius:4px; }
.home-dev .dev-img { flex:0 0 150px; }
.home-dev .dev-img img { width:150px; height:100px; object-fit:cover; border-radius:4px; display:block; }
.home-dev .dev-text { flex:1; min-width:0; }
.home-dev .dev-head { font-size:15px; font-weight:700; color:#0d2137; line-height:1.4; margin-bottom:4px; }
.home-dev .dev-body { font-size:14px; line-height:1.65; color:#1a2535; }
.home-dev .dev-src { font-size:12px; color:#7f8c8d; margin-top:6px; font-style:italic; }
.home-dev .dev-src a { color:#1a7fb5; font-style:normal; }
@media (max-width: 640px) { .home-dev { flex-direction:column; }
  .home-dev .dev-img img { width:100%; height:160px; } }
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
    """Compact list for the right column: ID, Date (first posted), Title."""
    rows = []
    for t in trials:
        new = " <span class='home-chip new'>New</span>" if t.get("is_new") else ""
        rows.append(
            "<tr>"
            f"<td><a href='{escape(t['url'], quote=True)}' target='_blank' "
            f"rel='noopener'><b>{escape(t['id'])}</b></a></td>"
            f"<td style='white-space:nowrap'>{escape(_fmt_date(t.get('first_posted')) or '—')}</td>"
            f"<td>{escape(t['title'])}{new}</td>"
            "</tr>"
        )
    return (
        "<table class='home-trials'><thead><tr><th>ID</th><th>Date</th><th>Title</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def reference_urls(md: str) -> Dict[int, str]:
    """{N: url} from a briefing's References section."""
    refs = split_sections(md).get("References", "")
    out: Dict[int, str] = {}
    for m in re.finditer(r"^\s*\[(\d+)\].*?(https?://\S+)", refs, re.M):
        out[int(m.group(1))] = m.group(2).rstrip(").,")
    return out


def parse_link_line(line: str) -> tuple:
    """(text, url) from a '*→ ...*' line.

    Handles '[text](url)' and the 'text(url)' form the model sometimes writes,
    including URLs that contain parentheses.
    """
    s = line.strip().strip("*_ ").lstrip("→").strip().rstrip("*_ ")
    i = s.find("http")
    if i < 0:
        return s, ""
    url = s[i:].split()[0]
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    text = s[:i].rstrip("(").rstrip().rstrip("]").lstrip("[").strip()
    return text, url


def parse_development(
    block: str, refs: Optional[Dict[int, str]] = None
) -> Dict[str, str]:
    """Split a Key Developments entry into headline, body, link text and URL.

    The URL is the entry's "→ link" line, or else the first cited reference.
    """
    link_text, url = "", ""
    lines = block.splitlines()
    body_lines = []
    for line in lines:
        if line.strip().lstrip("*_ ").startswith("→"):
            link_text, url = parse_link_line(line)
        else:
            body_lines.append(line)
    body = " ".join(" ".join(body_lines).split())
    if not url and refs:
        cite = re.search(r"\[(\d+)\]", body)
        if cite:
            url = refs.get(int(cite.group(1)), "")
    m = re.match(r"\*\*(.+?)\*\*\s*(.*)", body)
    headline, rest = (m.group(1), m.group(2)) if m else ("", body)
    return {
        "headline": strip_citations(headline).strip(),
        "body": strip_citations(rest).strip(),
        "link_text": link_text.strip(),
        "url": url,
    }


def _inline(text: str) -> str:
    """Escape, then allow **bold**, *italic* and [text](url) links."""
    html = escape(text)
    html = _LINK_RE.sub(
        lambda m: f"<a href='{m.group(2)}' target='_blank' rel='noopener'>{m.group(1)}</a>",
        html,
    )
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    return re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<em>\1</em>", html)


def _dev_card(d: Dict[str, str], image: str) -> str:
    """A Key Developments card in the briefing's style, with the article image."""
    url = escape(d["url"], quote=True)
    img = ""
    if image:
        img = (
            f"<a class='dev-img' href='{url}' target='_blank' rel='noopener'>"
            f"<img src='{escape(image, quote=True)}' alt='' loading='lazy' "
            "referrerpolicy='no-referrer' onerror=\"this.closest('.dev-img').remove()\">"
            "</a>"
        )
    head = (
        f"<div class='dev-head'>{_inline(d['headline'])}</div>" if d["headline"] else ""
    )
    src = ""
    if d["url"]:
        label = _inline(d["link_text"]) if d["link_text"] else "Read the source"
        src = (
            f"<div class='dev-src'>→ <a href='{url}' target='_blank' rel='noopener'>"
            f"{label}</a></div>"
        )
    return (
        f"<div class='home-dev'>{img}<div class='dev-text'>{head}"
        f"<div class='dev-body'>{_inline(d['body'])}</div>{src}</div></div>"
    )


def _briefing_html(md: str, title: str, meta: str) -> str:
    from src.common.email_briefing import render_newsletter_html

    return render_newsletter_html(title=title, meta=meta, briefing_md=md)


@st.dialog("Latest Intelligence Briefing", width="large")
def _briefing_dialog(run: Dict[str, Any]) -> None:
    meta = (
        f"Run #{run['id']} · {_fmt_date(run.get('run_ts'))} · "
        f"{run.get('n_scored') or 0} sources"
    )
    components.html(
        _briefing_html(run.get("briefing_text") or "", "Intelligence Briefing", meta),
        height=720,
        scrolling=True,
    )


@st.dialog("Latest newsletter", width="large")
def _newsletter_dialog(nl: Dict[str, Any], open_editor: Callable) -> None:
    title = nl.get("title") or "Endometriosis Newsletter"
    meta = "Newsletter · " + _fmt_date(nl.get("updated_ts") or nl.get("created_ts"))
    components.html(
        _briefing_html(nl.get("content_md") or "", title, meta),
        height=680,
        scrolling=True,
    )
    st.button(
        "✏️ Edit this newsletter",
        key="home_nl_edit",
        on_click=open_editor,
        args=(nl["id"],),
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
            text = latest.get("briefing_text") or ""
            sections = split_sections(text)
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
            devs = [
                parse_development(b, reference_urls(text))
                for b in key_development_blocks(sections.get("Key Developments", ""))
            ]
            if devs:
                st.markdown(
                    "<div class='home-h'>Key Developments</div>", unsafe_allow_html=True
                )
                known = {
                    s.get("url"): s.get("image")
                    for s in deps["sources_of"](latest)
                    if s.get("image")
                }
                need = tuple(
                    d["url"] for d in devs if d["url"] and d["url"] not in known
                )
                imgs = {**(_images(need) if need else {}), **known}
                st.markdown(
                    "".join(_dev_card(d, imgs.get(d["url"], "")) for d in devs),
                    unsafe_allow_html=True,
                )

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
            if st.button(
                "📰 Open latest Intelligence Briefing",
                key="home_open_brief",
                use_container_width=True,
            ):
                _briefing_dialog(latest)
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
            if st.button(
                "📝 Open latest newsletter",
                key="home_open_nl",
                use_container_width=True,
            ):
                _newsletter_dialog(nl, deps["open_newsletter"])

        n_new = sum(1 for t in trials if t.get("is_new"))
        n_co = sum(1 for t in trials if t.get("is_company"))
        st.markdown(
            _stat(
                "Open US clinical trials",
                str(len(trials)) if trials else "—",
                (
                    f"{n_new} new · {n_co} company-sponsored"
                    if trials
                    else "They load automatically each day"
                ),
            ),
            unsafe_allow_html=True,
        )
        if trials:
            newest = sorted(
                trials,
                key=lambda t: (t.get("first_posted") or "", t["id"]),
                reverse=True,
            )[:5]
            st.markdown(
                "<div class='home-h' style='font-size:15px;margin-top:6px'>"
                "Most recent clinical trials</div>" + _trials_table(newest),
                unsafe_allow_html=True,
            )
            st.caption(
                "Most recently posted · all trials are in the Clinical Trials tab."
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
