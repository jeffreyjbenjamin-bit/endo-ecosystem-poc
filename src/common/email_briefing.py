"""
email_briefing.py — Send AI/intelligence briefings via SMTP.

Required .env vars:
    EMAIL_SMTP_HOST      e.g. smtp.gmail.com
    EMAIL_SMTP_PORT      e.g. 587
    EMAIL_FROM           sender address
    EMAIL_PASSWORD       SMTP password / app password
    EMAIL_DEFAULT_TO     (optional) pre-fill recipient
"""

from __future__ import annotations

import os
import re
import smtplib
import textwrap
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env", override=True)


@dataclass
class SMTPConfig:
    host: str
    port: int
    from_addr: str
    password: str
    default_to: str


def load_smtp_config() -> Optional[SMTPConfig]:
    load_dotenv(_REPO_ROOT / ".env", override=True)
    host = os.getenv("EMAIL_SMTP_HOST", "").strip()
    port_str = os.getenv("EMAIL_SMTP_PORT", "587").strip()
    from_addr = os.getenv("EMAIL_FROM", "").strip()
    password = os.getenv("EMAIL_PASSWORD", "").strip()
    default_to = os.getenv("EMAIL_DEFAULT_TO", "").strip()

    if not (host and from_addr and password):
        return None

    try:
        port = int(port_str)
    except ValueError:
        port = 587

    return SMTPConfig(
        host=host,
        port=port,
        from_addr=from_addr,
        password=password,
        default_to=default_to,
    )


def smtp_configured() -> bool:
    return load_smtp_config() is not None


# ── Reference URL extraction ──────────────────────────────────────────────────


def _parse_reference_urls(briefing_md: str) -> Dict[int, str]:
    """Return {citation_number: url} from the ## References section."""
    urls: Dict[int, str] = {}
    in_refs = False
    for line in briefing_md.splitlines():
        if re.match(r"^##\s+References", line):
            in_refs = True
            continue
        if in_refs:
            if line.startswith("## "):
                break
            m = re.match(r"^\[(\d+)\].*?(https?://\S+)", line)
            if m:
                urls[int(m.group(1))] = m.group(2).rstrip(".,)")
    return urls


# ── Markdown → HTML ───────────────────────────────────────────────────────────


def _linkify_citations(text: str, ref_urls: Dict[int, str]) -> str:
    """Replace [N] with a linked superscript where a URL is known."""

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        url = ref_urls.get(n)
        if url:
            return f'<a href="{url}" style="color:#1a7fb5;font-size:11px;vertical-align:super">[{n}]</a>'
        return m.group(0)

    return re.sub(r"\[(\d+)\]", _replace, text)


def _md_inline(text: str) -> str:
    """Apply inline Markdown: bold, italic, links."""
    # Links: [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)",
        r'<a href="\2" style="color:#1a7fb5">\1</a>',
        text,
    )
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic (→ links)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


# Sections whose paragraph blocks get card treatment (same as Key Developments)
_CARD_SECTIONS = {
    "Key Developments",
    "Diagnostics",
    "Therapeutics",
    "Epidemiology",
    "Clinical & Regulatory Highlights",
    "Emerging Research Themes",
    "What to Watch",
}


def _markdown_to_html(md: str, ref_urls: Dict[int, str]) -> str:
    """
    Convert briefing Markdown to styled HTML blocks.

    Key Developments entries (paragraph blocks with **Headline** [N] … *→ link*)
    get rendered as visual cards.
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    in_key_dev = False  # accumulating paragraph blocks for card rendering
    in_card_section = False  # a key-dev-section div is open
    in_list = False
    pending_block: List[str] = []

    def _flush_key_dev_block(block: List[str]) -> None:
        """Render a Key Developments paragraph block as an HTML card."""
        raw = " ".join(block).strip()
        if not raw:
            return

        # Extract trailing italic link line  *→ ...*
        link_html = ""
        link_match = re.search(r"\*→\s*(.+?)\*\s*$", raw)
        if link_match:
            link_html = _md_inline("*→ " + link_match.group(1) + "*")
            raw = raw[: link_match.start()].strip()

        # Apply citation links + inline formatting
        raw = _linkify_citations(raw, ref_urls)
        raw = _md_inline(raw)

        out.append(
            f'<div class="dev-entry">'
            f'<div class="dev-body">{raw}</div>'
            + (f'<div class="dev-source">{link_html}</div>' if link_html else "")
            + "</div>"
        )

    def _close_card_section() -> None:
        nonlocal in_key_dev, in_card_section, pending_block
        if pending_block:
            _flush_key_dev_block(pending_block)
            pending_block = []
        if in_card_section:
            out.append("</div>")
            in_card_section = False
        in_key_dev = False

    def _close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i].rstrip()

        # Section headings
        if line.startswith("### "):
            _close_list()
            _close_card_section()
            text = _md_inline(line[4:])
            out.append(f"<h3>{text}</h3>")
            i += 1
            continue

        if line.startswith("## "):
            _close_list()
            _close_card_section()
            section = line[3:].strip()
            text = _md_inline(section)
            out.append(f"<h2>{text}</h2>")
            if section in _CARD_SECTIONS:
                in_key_dev = True
                in_card_section = True
                out.append('<div class="key-dev-section">')
            i += 1
            continue

        if line.startswith("# "):
            _close_list()
            _close_card_section()
            text = _md_inline(line[2:])
            out.append(f"<h1>{text}</h1>")
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^-{3,}$", line):
            _close_list()
            _close_card_section()
            out.append("<hr>")
            i += 1
            continue

        # Blank line — flush pending card block but keep section open
        if not line.strip():
            _close_list()
            if in_key_dev and pending_block:
                _flush_key_dev_block(pending_block)
                pending_block = []
            i += 1
            continue

        # Key Developments — accumulate paragraph block lines
        if in_key_dev:
            pending_block.append(line)
            i += 1
            continue

        # Bullet list
        if line.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = _linkify_citations(_md_inline(line[2:]), ref_urls)
            out.append(f"<li>{item}</li>")
            i += 1
            continue

        # Numbered list (references section)
        ref_m = re.match(r"^\[(\d+)\]\s+(.+)$", line)
        if ref_m:
            _close_list()
            n = ref_m.group(1)
            body = _linkify_citations(_md_inline(ref_m.group(2)), ref_urls)
            out.append(
                f'<p style="font-size:12px;color:#555;margin:4px 0">'
                f"<strong>[{n}]</strong> {body}</p>"
            )
            i += 1
            continue

        # Plain paragraph
        _close_list()
        text = _linkify_citations(_md_inline(line), ref_urls)
        out.append(f"<p>{text}</p>")
        i += 1

    _close_list()
    _close_card_section()

    return "\n".join(out)


# ── HTML email builder ────────────────────────────────────────────────────────

_NEW_SINCE_BLOCK_TMPL = """\
<div class="new-since">
  <div class="new-since-label">&#x1F195; New Since Last Report</div>
  <p style="font-size:13px;color:#1a3d5c;margin:6px 0 8px 0">
    The following sources were not included in the previous briefing:
  </p>
  <ul>
    {items}
  </ul>
</div>"""


def _build_html(
    title: str,
    meta: str,
    briefing_md: str,
    new_items: Optional[List[Dict]] = None,
) -> str:
    ref_urls = _parse_reference_urls(briefing_md)
    body_html = _markdown_to_html(briefing_md, ref_urls)

    new_since_block = ""
    if new_items:
        li_parts = []
        for item in new_items:
            t = item.get("title", "Untitled")
            u = item.get("url", "")
            if u:
                li_parts.append(f'<li><a href="{u}" style="color:#1a7fb5">{t}</a></li>')
            else:
                li_parts.append(f"<li>{t}</li>")
        new_since_block = _NEW_SINCE_BLOCK_TMPL.format(items="\n    ".join(li_parts))

    return textwrap.dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          /* Reset */
          body {{ margin:0; padding:0; background:#f0f4f8;
                 font-family:Georgia,'Times New Roman',serif; }}
          /* Wrapper */
          .wrapper {{ max-width:680px; margin:32px auto 48px;
                      background:#ffffff; border-radius:8px; overflow:hidden;
                      box-shadow:0 2px 16px rgba(0,0,0,0.10); }}
          /* Header */
          .header {{ background:#0d2137; padding:28px 36px 22px; }}
          .header-eyebrow {{ color:#7fa8cc; font-size:11px; text-transform:uppercase;
                             letter-spacing:0.10em; font-family:Arial,sans-serif;
                             margin-bottom:6px; }}
          .header-title {{ color:#ffffff; font-size:24px; font-weight:bold;
                           line-height:1.25; margin:0 0 6px 0; }}
          .header-meta {{ color:#a0c4e4; font-size:12px;
                          font-family:Arial,sans-serif; margin:0; }}
          /* New-since bar */
          .new-since {{ background:#e8f4fd; border-left:4px solid #1a7fb5;
                        padding:16px 24px; margin:0; }}
          .new-since-label {{ font-size:10px; text-transform:uppercase;
                              letter-spacing:0.10em; color:#1a7fb5;
                              font-family:Arial,sans-serif; font-weight:bold; }}
          .new-since ul {{ margin:4px 0 0 0; padding-left:20px; }}
          .new-since li {{ font-size:13px; color:#1a3d5c; margin-bottom:4px; }}
          .new-since a {{ color:#1a7fb5; }}
          /* Body */
          .body {{ padding:28px 36px 36px; }}
          h1 {{ font-size:20px; color:#0d3349; border-bottom:2px solid #1a7fb5;
                padding-bottom:8px; margin:0 0 6px 0; }}
          h2 {{ font-size:12px; color:#0d3349; margin:32px 0 14px 0;
                text-transform:uppercase; letter-spacing:0.07em;
                border-bottom:1px solid #e4e9f0; padding-bottom:5px;
                font-family:Arial,sans-serif; }}
          h3 {{ font-size:14px; color:#1a5276; margin:18px 0 4px 0; }}
          p {{ font-size:14px; line-height:1.7; color:#2c3e50; margin:0 0 10px 0; }}
          ul {{ margin:0 0 12px 0; padding-left:22px; }}
          li {{ font-size:14px; line-height:1.65; color:#2c3e50; margin-bottom:6px; }}
          a {{ color:#1a7fb5; }}
          strong {{ color:#0d2137; }}
          hr {{ border:none; border-top:1px solid #e4e9f0; margin:24px 0; }}
          /* Key Developments cards */
          .key-dev-section {{ margin:0; }}
          .dev-entry {{ padding:16px 20px; margin-bottom:12px;
                        background:#f8faff; border:1px solid #e4e9f0;
                        border-left:3px solid #1a7fb5; border-radius:4px; }}
          .dev-body {{ font-size:14px; line-height:1.72; color:#1a2535;
                       margin:0 0 8px 0; }}
          .dev-source {{ font-size:12px; color:#7f8c8d;
                         font-family:Arial,sans-serif; margin:0; }}
          .dev-source a {{ color:#1a7fb5; font-style:normal; }}
          /* Footer */
          .footer {{ background:#f8faff; border-top:1px solid #e4e9f0;
                     padding:16px 36px; font-size:11px; color:#999;
                     font-family:Arial,sans-serif; }}
          .footer a {{ color:#7fa8cc; }}
        </style>
        </head>
        <body>
        <div class="wrapper">

          <div class="header">
            <div class="header-eyebrow">Evidence Gap &nbsp;·&nbsp; Endometriosis Intelligence</div>
            <div class="header-title">{title}</div>
            <div class="header-meta">{meta}</div>
          </div>

          {new_since_block}

          <div class="body">
            {body_html}
          </div>

          <div class="footer">
            Generated by <strong>Evidence Gap</strong> &nbsp;·&nbsp; Bold Wave Productions<br>
            This briefing is intended for research and professional use only.
          </div>

        </div>
        </body>
        </html>
    """
    )


def _build_plain(
    title: str,
    meta: str,
    briefing_md: str,
    new_items: Optional[List[Dict]] = None,
) -> str:
    parts = [f"{title}\n{meta}\n"]
    if new_items:
        parts.append("NEW SINCE LAST REPORT")
        parts.append("-" * 30)
        for item in new_items:
            t = item.get("title", "Untitled")
            u = item.get("url", "")
            parts.append(f"  • {t}" + (f"\n    {u}" if u else ""))
        parts.append("")
    parts.append(briefing_md)
    parts.append("\n---\nGenerated by Evidence Gap · Bold Wave Productions")
    return "\n".join(parts)


# ── Public HTML renderer (also used by the Streamlit preview) ────────────────


def render_newsletter_html(
    title: str,
    meta: str,
    briefing_md: str,
    new_items: Optional[List[Dict]] = None,
) -> str:
    """Return styled newsletter HTML suitable for in-app preview or email."""
    return _build_html(title, meta, briefing_md, new_items)


# ── Public send function ──────────────────────────────────────────────────────


def send_briefing(
    to_addr: str,
    subject: str,
    title: str,
    meta: str,
    briefing_md: str,
    new_items: Optional[List[Dict]] = None,
) -> None:
    """
    Send a briefing email.

    new_items: optional list of {title, url} dicts shown in a
               "New Since Last Report" section at the top.

    Raises on any SMTP failure.
    """
    cfg = load_smtp_config()
    if cfg is None:
        raise RuntimeError(
            "EMAIL_SMTP_HOST, EMAIL_FROM, and EMAIL_PASSWORD must be set in .env"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.from_addr
    msg["To"] = to_addr

    plain = _build_plain(title, meta, briefing_md, new_items)
    html = _build_html(title, meta, briefing_md, new_items)

    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(cfg.host, cfg.port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.login(cfg.from_addr, cfg.password)
        server.sendmail(cfg.from_addr, to_addr, msg.as_string())
