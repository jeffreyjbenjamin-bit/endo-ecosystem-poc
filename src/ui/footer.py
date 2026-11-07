"""
footer.py — persistent page footer and the in-app Changelog page.

The footer is pinned to the bottom of every page: copyright, app version, and a
Changelog link. The link switches the view inside the current session (no page
reload, so the password isn't asked again) and sets ?view=changelog in the URL,
which also opens the page directly.

The version shown is the first version number (vX.Y.Z) in the CHANGELOG.md table.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import streamlit as st

CHANGELOG_PATH = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
_VIEW_PARAM = "view"
_CHANGELOG_VIEW = "changelog"

_FOOTER_CSS = """
<style>
  /* Keep page content clear of the fixed footer */
  [data-testid="stMainBlockContainer"] { padding-bottom: 4.5rem; }

  .st-key-app_footer {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
    background: #ffffff; border-top: 1px solid #e4e9f0;
    padding: 6px 16px; gap: 0.75rem;
  }
  .st-key-app_footer > div {
    width: auto !important; flex: 0 0 auto !important;
  }
  .st-key-app_footer p {
    margin: 0; font-size: 12.5px; color: #5f6b7a; text-align: center;
  }
  .st-key-app_footer .stButton > button {
    background: transparent; color: #1a6eb5; border: none; border-radius: 4px;
    padding: 0 4px; min-height: 0; font-size: 12.5px; text-decoration: underline;
  }
  .st-key-app_footer .stButton > button:hover {
    background: transparent; color: #0d2137;
  }
  @media (max-width: 640px) {
    .st-key-app_footer { flex-wrap: wrap; justify-content: center; }
  }
</style>
"""


def app_version() -> str:
    try:
        text = re.sub(
            r"<!--.*?-->", "", CHANGELOG_PATH.read_text(encoding="utf-8"), flags=re.S
        )
        m = re.search(r"\bv(\d+\.\d+\.\d+)\b", text)
        return m.group(1) if m else ""
    except OSError:
        return ""


def _set_view(view: str | None) -> None:
    if view:
        st.query_params[_VIEW_PARAM] = view
    elif _VIEW_PARAM in st.query_params:
        del st.query_params[_VIEW_PARAM]


def render_footer() -> None:
    st.markdown(_FOOTER_CSS, unsafe_allow_html=True)
    version = app_version()
    with st.container(
        key="app_footer",
        horizontal=True,
        horizontal_alignment="center",
        vertical_alignment="center",
    ):
        st.markdown(
            f"© {datetime.now().year} BoldWave, Inc. All rights reserved."
            + (f" &nbsp;·&nbsp; Evidence Gap v{version}" if version else "")
        )
        st.button(
            "Changelog",
            key="footer_changelog",
            on_click=_set_view,
            args=(_CHANGELOG_VIEW,),
        )


def changelog_requested() -> bool:
    return st.query_params.get(_VIEW_PARAM) == _CHANGELOG_VIEW


def render_changelog_page() -> None:
    st.button(
        "← Back to Evidence Gap", key="changelog_back", on_click=_set_view, args=(None,)
    )
    try:
        text = CHANGELOG_PATH.read_text(encoding="utf-8")
    except OSError:
        st.error("Changelog not found.")
        return
    st.markdown(
        "<style>"
        "[data-testid='stMarkdownContainer'] table { width:100%; }"
        "[data-testid='stMarkdownContainer'] table td { vertical-align:top; }"
        "[data-testid='stMarkdownContainer'] table td:first-child"
        " { white-space:nowrap; font-weight:600; width:1%; padding-right:1.5rem; }"
        "[data-testid='stMarkdownContainer'] table td ul { margin:0.2rem 0 0.8rem 0; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.markdown(re.sub(r"<!--.*?-->", "", text, flags=re.S), unsafe_allow_html=True)
