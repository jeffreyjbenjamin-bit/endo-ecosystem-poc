"""Clinical Trials tab: open endometriosis trials with a US site."""

from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import streamlit as st

from src.pipelines.clinical_trials import (
    NEW_BADGE_DAYS,
    SOURCE_NAME,
    filter_trials,
    load_trials,
    refresh_trials,
    set_in_newsletter,
)

# Table drawn as a grid of Streamlit columns so each row can hold a checkbox.
_WIDTHS = [0.75, 1.15, 3.2, 1.7, 1.25, 1.05, 1.05, 0.85, 1.3]
_COLUMNS = [
    "Newsletter",
    "ID",
    "Title",
    "Lead Sponsor",
    "Study / Phase",
    "Status",
    "Primary Completion",
    "Enrollment",
    "Site Country",
]

_CSS = """
<style>
.st-key-ct_head { background:#f4f6f8; border:1px solid #e3e6ea; border-bottom:2px solid #dfe3e8;
  border-radius:8px 8px 0 0; padding:6px 10px; margin-bottom:-1rem; }
.st-key-ct_head p { font-size:13px; font-weight:600; color:#2c3e50; margin:0; }
.st-key-ct_grid { border:1px solid #e3e6ea; border-top:none; border-radius:0 0 8px 8px;
  padding:0 10px; }
.st-key-ct_grid [data-testid="stHorizontalBlock"] { border-bottom:1px solid #eef0f2;
  padding:6px 0; gap:0.6rem; }
.st-key-ct_head [data-testid="stHorizontalBlock"] { gap:0.6rem; }
.st-key-ct_grid [data-testid="stHorizontalBlock"]:hover { background:#fafbfc; }
.st-key-ct_grid p, .ct-cell { font-size:13px; line-height:1.35; margin:0; }
.ct-id a { font-weight:600; white-space:nowrap; }
.ct-sub { font-size:11px; color:#7f8c8d; margin-top:2px; }
.ct-dx { display:inline-block; margin-top:4px; font-size:11px; padding:1px 6px;
  border-radius:4px; background:#e8f6ee; color:#1e7e4a; cursor:help; }
.ct-new { display:inline-block; margin-right:6px; font-size:10px; font-weight:700;
  padding:1px 6px; border-radius:4px; background:#fff3cd; color:#8a6d00;
  vertical-align:1px; }
</style>
"""


def _cells(t: dict) -> list[str]:
    dx = ""
    if t.get("confirmed_dx"):
        dx = (
            f"<div class='ct-dx' title='{escape(t['confirmed_dx_phrase'], quote=True)}'>"
            "✓ Confirmed endometriosis required</div>"
        )
    new = (
        f"<span class='ct-new' title='First listed {escape(t.get('first_seen', '')[:10])}'>"
        "NEW</span>"
        if t.get("is_new")
        else ""
    )
    enroll = "—" if t.get("enrollment") is None else f"{t['enrollment']:,}"
    country = escape(", ".join(t.get("countries") or [])) or "—"
    if t.get("us_sites"):
        n = t["us_sites"]
        country += f"<div class='ct-sub'>{n} US site{'s' if n != 1 else ''}</div>"
    return [
        f"<div class='ct-cell ct-id'><a href='{escape(t['url'], quote=True)}' "
        f"target='_blank' rel='noopener'>{escape(t['id'])}</a>"
        f"<div class='ct-sub'>{escape(t.get('source') or SOURCE_NAME)}</div></div>",
        f"<div class='ct-cell'>{new}{escape(t['title'])}{dx}</div>",
        f"<div class='ct-cell'>{escape(t['lead_sponsor'])}"
        f"<div class='ct-sub'>{escape(t['sponsor_type'])}</div></div>",
        f"<div class='ct-cell'>{escape(t['study_phase'])}</div>",
        f"<div class='ct-cell'>{escape(t['status'])}</div>",
        f"<div class='ct-cell'>{escape(t['primary_completion'] or '—')}"
        f"<div class='ct-sub'>{escape(t['primary_completion_type'])}</div></div>",
        f"<div class='ct-cell'>{enroll}"
        f"<div class='ct-sub'>{escape(t['enrollment_type'])}</div></div>",
        f"<div class='ct-cell'>{country}</div>",
    ]


def _on_newsletter_toggle(nct_id: str, key: str) -> None:
    set_in_newsletter(nct_id, bool(st.session_state.get(key)))


def _grid(trials: list[dict]) -> None:
    with st.container(key="ct_head"):
        for col, name in zip(st.columns(_WIDTHS), _COLUMNS):
            col.markdown(name)
    with st.container(key="ct_grid", height=640):
        for t in trials:
            key = f"ct_nl_{t['id']}"
            st.session_state[key] = bool(t.get("in_newsletter"))
            cols = st.columns(_WIDTHS, vertical_alignment="top")
            cols[0].checkbox(
                "Include in newsletter",
                key=key,
                label_visibility="collapsed",
                help="Include in newsletter: makes this trial available on Build "
                "Newsletter from Intel Briefing",
                on_change=_on_newsletter_toggle,
                args=(t["id"], key),
            )
            for col, html in zip(cols[1:], _cells(t)):
                col.markdown(html, unsafe_allow_html=True)


def _refreshed_label(iso: str | None, tz_name: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso).astimezone(ZoneInfo(tz_name))
        return dt.strftime("%b %d, %Y %I:%M %p %Z")
    except (ValueError, KeyError):
        return iso


def render_tab(tz_name: str = "America/New_York") -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:22px;font-weight:700;margin-bottom:4px'>"
        "Clinical Trials</div>"
        "<div style='font-size:13px;color:#7f8c8d;margin-bottom:16px'>"
        "Open endometriosis trials (recruiting, not yet recruiting, or enrolling by "
        f"invitation) with at least one site in the United States, from {SOURCE_NAME}. "
        "Updated automatically every day; new trials are added to the next "
        "Intelligence Briefing. Tick <b>Newsletter</b> to offer a trial on Build "
        "Newsletter from Intel Briefing.</div>",
        unsafe_allow_html=True,
    )

    with st.container(horizontal=True, vertical_alignment="center"):
        refresh = st.button("🔄 Refresh", type="primary", key="ct_refresh")
        companies = st.toggle(
            "Companies only",
            key="ct_companies_only",
            help="Industry-sponsored trials only; hides universities, hospitals, "
            "government agencies and other non-company sponsors.",
        )
        confirmed = st.toggle(
            "Confirmed endometriosis only",
            key="ct_confirmed_only",
            help="Trials whose inclusion criteria require an endometriosis diagnosis "
            "confirmed by surgery or pathology (for example 'surgically / "
            "pathologically confirmed' or 'operative confirmation of endometriosis'). "
            "Trials where surgery is only planned, or where imaging or a clinical "
            "diagnosis is accepted instead, are left out.",
        )

    if refresh:
        with st.spinner(f"Searching {SOURCE_NAME}…"):
            try:
                refresh_trials()
            except Exception as exc:  # network / API errors
                st.error(f"Could not reach {SOURCE_NAME}: {exc}")

    data = load_trials()
    if not data:
        st.info(
            "No trials loaded yet. Click **Refresh** to search ClinicalTrials.gov "
            "(the list also updates automatically once a day)."
        )
        return

    trials = data.get("trials") or []
    shown = filter_trials(trials, companies_only=companies, confirmed_only=confirmed)
    n_new = sum(1 for t in trials if t.get("is_new"))
    n_nl = sum(1 for t in trials if t.get("in_newsletter"))
    extra = ""
    if n_new:
        extra += f" · **{n_new}** new in the last {NEW_BADGE_DAYS} days"
    if n_nl:
        extra += f" · **{n_nl}** marked for the newsletter"
    st.caption(
        f"Showing **{len(shown)}** of {len(trials)} open trials{extra} · last "
        f"refreshed {_refreshed_label(data.get('refreshed_at'), tz_name)}"
    )
    if not shown:
        st.info("No trials match the selected filters.")
        return
    _grid(shown)
