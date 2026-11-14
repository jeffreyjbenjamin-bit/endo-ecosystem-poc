"""Endometriosis News naming, logo in emails/previews, story prompt, themes."""

from src.common.email_briefing import (
    LOGO_CID,
    NEWSLETTER_NAME,
    build_message,
    newsletter_subject,
    render_newsletter_html,
)


def _html_part(msg):
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode()
    raise AssertionError("no html part")


def test_newsletter_name_and_subject():
    assert NEWSLETTER_NAME == "Endometriosis News"
    assert newsletter_subject("Endometriosis News") == "Endometriosis News"
    assert newsletter_subject("") == "Endometriosis News"
    assert (
        newsletter_subject("Faster Diagnosis")
        == "Endometriosis News — Faster Diagnosis"
    )


def test_preview_has_inline_logo_and_newsletter_font():
    news = render_newsletter_html("T", "m", "Some text [1].", kind="newsletter")
    brief = render_newsletter_html("T", "m", "Some text [1].")
    assert "data:image/png;base64," in news and "data:image/png;base64," in brief
    assert "Calibri" in news and "Calibri" not in brief
    assert "Endometriosis News" in news
    assert "Endometriosis Intelligence Briefing" in brief


def test_email_attaches_logo_inline():
    msg = build_message(
        "a@x.org", "b@x.org", "Subj", "T", "m", "Hello", kind="newsletter"
    )
    assert msg.get_content_type() == "multipart/related"
    html = _html_part(msg)
    assert f'src="cid:{LOGO_CID}"' in html and "data:image" not in html
    imgs = [p for p in msg.walk() if p.get_content_type() == "image/png"]
    assert len(imgs) == 1 and imgs[0]["Content-ID"] == f"<{LOGO_CID}>"
    assert "inline" in imgs[0]["Content-Disposition"]
    types = [p.get_content_type() for p in msg.walk()]
    assert "text/plain" in types


def test_story_prompt_has_narrative_intro_not_executive_summary():
    from src.ui.newsletter_from_briefing import _NB2_SYSTEM, _build_prompt

    a = {
        "title": "A",
        "url": "https://a",
        "source": "s",
        "published_date": "2026",
        "snippet": "x",
    }
    p = _build_prompt([a], [a], "Endometriosis News", "Run #1", "")
    assert "## Executive Summary" not in p and "Executive Summary" not in p
    assert "Introduction, with no heading" in p and "thesis" in p
    assert "**Endometriosis News**" in p and "## What This Means" in p
    assert "[A headline for this issue" in p  # default title -> model writes a headline
    p2 = _build_prompt([a], [], "Faster Diagnosis", "Run #1", "")
    assert "# Faster Diagnosis" in p2
    assert "call-out" in _NB2_SYSTEM


def test_story_theme_buttons():
    from src.ui.newsletter_from_briefing import CHAT_SYSTEM, STORY_THEMES

    labels = [label for label, _ in STORY_THEMES]
    assert labels[0].endswith("Suggest themes") and len(labels) >= 4
    assert all("edit blocks" in ask for _, ask in STORY_THEMES[1:])
    assert "Story theme:" in CHAT_SYSTEM
