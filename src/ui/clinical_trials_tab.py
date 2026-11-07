"""Clinical Trials tab: open endometriosis trials with a US site."""

from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import streamlit as st

from src.pipelines.clinical_trials import (
    SOURCE_NAME,
    filter_trials,
    load_trials,
    refresh_trials,
)

_CSS = """
<style>
.ct-wrap { max-height: 70vh; overflow: auto; border: 1px solid #e3e6ea; border-radius: 8px; }
.ct-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ct-table th { position: sticky; top: 0; background: #f4f6f8; color: #2c3e50; text-align: left;
  padding: 8px 10px; border-bottom: 2px solid #dfe3e8; white-space: nowrap; z-index: 1; }
.ct-table td { padding: 8px 10px; border-bottom: 1px solid #eef0f2; vertical-align: top; }
.ct-table tr:hover td { background: #fafbfc; }
.ct-id a { font-weight: 600; white-space: nowrap; }
.ct-sub { font-size: 11px; color: #7f8c8d; margin-top: 2px; }
.ct-dx { display: inline-block; margin-top: 4px; font-size: 11px; padding: 1px 6px;
  border-radius: 4px; background: #e8f6ee; color: #1e7e4a; cursor: help; }
.ct-status { white-space: nowrap; }
.ct-num { white-space: nowrap; }
</style>
"""

_COLUMNS = [
    "ID",
    "Title",
    "Lead Sponsor",
    "Study / Phase",
    "Status",
    "Primary Completion",
    "Enrollment",
    "Site Country",
]


def _row(t: dict) -> str:
    dx = ""
    if t.get("confirmed_dx"):
        dx = (
            f"<div class='ct-dx' title='{escape(t['confirmed_dx_phrase'], quote=True)}'>"
            "✓ Confirmed endometriosis required</div>"
        )
    enroll = "—" if t.get("enrollment") is None else f"{t['enrollment']:,}"
    countries = t.get("countries") or []
    country_html = escape(", ".join(countries)) or "—"
    if t.get("us_sites"):
        n = t["us_sites"]
        country_html += f"<div class='ct-sub'>{n} US site{'s' if n != 1 else ''}</div>"
    cells = [
        f"<td class='ct-id'><a href='{escape(t['url'], quote=True)}' target='_blank' "
        f"rel='noopener'>{escape(t['id'])}</a>"
        f"<div class='ct-sub'>{escape(t.get('source') or SOURCE_NAME)}</div></td>",
        f"<td>{escape(t['title'])}{dx}</td>",
        f"<td>{escape(t['lead_sponsor'])}"
        f"<div class='ct-sub'>{escape(t['sponsor_type'])}</div></td>",
        f"<td>{escape(t['study_phase'])}</td>",
        f"<td class='ct-status'>{escape(t['status'])}</td>",
        f"<td class='ct-num'>{escape(t['primary_completion'] or '—')}"
        f"<div class='ct-sub'>{escape(t['primary_completion_type'])}</div></td>",
        f"<td class='ct-num'>{enroll}"
        f"<div class='ct-sub'>{escape(t['enrollment_type'])}</div></td>",
        f"<td>{country_html}</td>",
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def _table(trials: list[dict]) -> str:
    head = "".join(f"<th>{c}</th>" for c in _COLUMNS)
    body = "".join(_row(t) for t in trials)
    return (
        f"<div class='ct-wrap'><table class='ct-table'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


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
        f"invitation) with at least one site in the United States, from {SOURCE_NAME}."
        "</div>",
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
        st.info("No trials loaded yet. Click **Refresh** to search ClinicalTrials.gov.")
        return

    trials = data.get("trials") or []
    shown = filter_trials(trials, companies_only=companies, confirmed_only=confirmed)
    st.caption(
        f"Showing **{len(shown)}** of {len(trials)} open trials · last refreshed "
        f"{_refreshed_label(data.get('refreshed_at'), tz_name)}"
    )
    if not shown:
        st.info("No trials match the selected filters.")
        return
    st.markdown(_table(shown), unsafe_allow_html=True)
