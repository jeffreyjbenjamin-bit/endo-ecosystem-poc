"""
wysiwyg.py — visual (WYSIWYG) newsletter editor built on Tiptap.

The editor runs in a small Streamlit component (components/tiptap_editor/index.html,
no build step; Tiptap is loaded from a CDN in the viewer's browser). It edits
Markdown visually and returns Markdown, so newsletters keep the format the email
template and the rest of the app already use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "tiptap_editor"
_tiptap = components.declare_component("tiptap_editor", path=str(_COMPONENT_DIR))


def clean_markdown(md: str) -> str:
    """Convert the editor's Markdown back to the dialect the email template renders.

    - Citations like [1] come back escaped as \\[1\\]; underscores in URLs as \\_.
    - Hard line breaks come back as a trailing backslash; the template uses two spaces.
    - Bare URLs can come back as <https://...> autolinks; the template wants plain URLs.
    """
    md = md or ""
    md = re.sub(r"\\([\[\]_])", r"\1", md)
    md = re.sub(r"\\\n", "  \n", md)
    md = re.sub(r"<(https?://[^>\s]+)>", r"\1", md)
    return md.rstrip() + "\n"


def tiptap_editor(
    markdown: str, doc_id: str, key: str, height: int = 760
) -> Optional[str]:
    """Show the visual editor. Returns the edited Markdown, or None until the first edit.

    The editor only reloads its content when `doc_id` changes, so pass a new doc_id
    whenever the Markdown was changed elsewhere (e.g. in the Markdown view).
    """
    value = _tiptap(
        markdown=markdown, doc_id=doc_id, height=height, key=key, default=None
    )
    return clean_markdown(value) if isinstance(value, str) else None
