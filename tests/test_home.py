"""Home page helpers, trends parsing and preview images (no network)."""

import json
from datetime import date

from src.common.page_images import image_from_html, image_from_search_item
from src.pipelines.trends import (
    build_prompt,
    collect_items,
    parse_response,
    window_items,
)
from src.ui.home_tab import (
    key_development_blocks,
    recent_results,
    split_sections,
    strip_citations,
)

BRIEFING = """## Intelligence Summary
Big news this week [1], [2] (June 3, 2026).

## Key Developments
**Drug X hits Phase 3 endpoint.** [1] Details here.

*→ [Drug X results](https://a.com/x)*

**Blood test cleared.** [2] More.

*→ [Test](https://b.com/t)*

## References
[1] A.
"""


def test_sections_and_key_developments():
    s = split_sections(BRIEFING)
    assert list(s) == ["Intelligence Summary", "Key Developments", "References"]
    assert (
        strip_citations(s["Intelligence Summary"])
        == "Big news this week (June 3, 2026)."
    )
    blocks = key_development_blocks(s["Key Developments"])
    assert len(blocks) == 2
    assert blocks[0].startswith("**Drug X") and "https://a.com/x" in blocks[0]


def test_recent_results_dedupes_and_sorts_newest_first():
    runs = [
        {
            "s": [
                {"url": "https://a.com/1", "published_date": "2026-01-01"},
                {"url": "https://a.com/2", "published_date": "2026-03-01"},
            ]
        },
        {
            "s": [
                {"url": "https://a.com/1/", "published_date": "2026-01-01"},
                {"url": "https://a.com/3", "published_date": None},
            ]
        },
    ]
    out = recent_results(runs, 6, lambda r: r["s"])
    assert [r["url"] for r in out] == [
        "https://a.com/2",
        "https://a.com/1",
        "https://a.com/3",
    ]


def test_trend_windows_and_parse():
    runs = [
        {
            "run_ts": "2026-11-06T07:00:00+00:00",
            "sources_json": json.dumps(
                [
                    {
                        "title": "Week item",
                        "published_date": "2026-11-05",
                        "source_name": "x",
                    },
                    {
                        "title": "Undated item",
                        "published_date": None,
                        "source_name": "y",
                    },
                ]
            ),
        }
    ]
    docs = [
        {"title": "Old paper", "published_date": "2026-03-01", "source": "pubmed"},
        {"title": "Ancient", "published_date": "2019-01-01", "source": "pubmed"},
    ]
    trials = [
        {
            "title": "Trial A",
            "first_posted": "2026-10-20",
            "lead_sponsor": "Acme",
            "study_phase": "Phase 2",
        }
    ]
    items = collect_items(runs, docs, trials)
    today = date(2026, 11, 7)
    assert {i["title"] for i in window_items(items, 7, today)} == {
        "Week item",
        "Undated item",
    }
    assert len(window_items(items, 30, today)) == 3
    assert len(window_items(items, 365, today)) == 4
    prompt = build_prompt(items, today)
    assert "=== WEEK" in prompt and "Ancient" not in prompt

    raw = (
        "```json\n"
        + json.dumps(
            {
                "week": {
                    "headline": "H",
                    "trends": [{"title": "T1", "detail": "D", "signal": "Rising"}],
                },
                "month": {"headline": "M", "trends": []},
            }
        )
        + "\n```"
    )
    parsed = parse_response(raw)
    assert parsed["week"]["trends"][0]["signal"] == "rising"
    assert parsed["year"] == {"headline": "", "trends": []}


def test_preview_images():
    html = (
        '<head><meta content="/img/a.jpg" property="og:image">'
        '<meta name="twitter:image" content="https://cdn/b.jpg"></head>'
    )
    assert (
        image_from_html(html, "https://site.com/post") == "https://site.com/img/a.jpg"
    )
    assert image_from_html("<head></head>", "https://x") is None
    item = {"pagemap": {"cse_image": [{"src": "https://cdn/c.jpg"}]}}
    assert image_from_search_item(item) == "https://cdn/c.jpg"
    assert image_from_search_item({}) is None


def test_development_cards_parse_both_link_styles():
    from src.ui.home_tab import parse_development, parse_link_line, reference_urls

    assert parse_link_line("*→ [Drug X results](https://a.com/x)*") == (
        "Drug X results",
        "https://a.com/x",
    )
    # model sometimes writes text(url), and URLs can contain parentheses
    assert parse_link_line(
        "*→ Angiogenesis study - The Lancet(https://l.com/PIIS3050-5038(26)00048-8/f)*"
    ) == ("Angiogenesis study - The Lancet", "https://l.com/PIIS3050-5038(26)00048-8/f")

    d = parse_development(
        "**Drug X hits Phase 3 endpoint.** [1] Details here.\n\n"
        "*→ [Drug X results](https://a.com/x)*"
    )
    assert d == {
        "headline": "Drug X hits Phase 3 endpoint.",
        "body": "Details here.",
        "link_text": "Drug X results",
        "url": "https://a.com/x",
    }
    # no link line: fall back to the first cited reference
    refs = reference_urls(BRIEFING + "\n[2] B. https://b.com/t\n")
    d = parse_development("**Blood test cleared.** [2] More.", refs)
    assert d["url"] == "https://b.com/t"


def test_trend_evidence_is_checked_and_charted():
    from src.pipelines.trends import SCHEMA, monthly_activity
    from src.ui.home_tab import trend_month_counts

    runs = [
        {
            "run_ts": "2026-11-06T07:00:00+00:00",
            "sources_json": json.dumps(
                [
                    {
                        "title": "A new",
                        "published_date": "2026-11-05",
                        "source_name": "x",
                    },
                    {
                        "title": "B press",
                        "published_date": "2026-10-02",
                        "source_name": "y",
                        "is_press_release": True,
                    },
                ]
            ),
        }
    ]
    docs = [{"title": "C old", "published_date": "2026-01-15", "source": "pubmed"}]
    items = collect_items(runs, docs, [])
    today = date(2026, 11, 7)
    ids = {i["title"]: i["id"] for i in items}
    assert f"[{ids['A new']}] 2026-11-05" in build_prompt(items, today)

    raw = json.dumps(
        {
            "week": {
                "headline": "W",
                "trends": [
                    # C is not in the week window and 999 doesn't exist: both dropped
                    {
                        "title": "T",
                        "detail": "d",
                        "signal": "new",
                        "evidence": [ids["A new"], ids["C old"], 999, ids["A new"]],
                    }
                ],
            },
            "month": {"headline": "M", "trends": []},
            "year": {
                "headline": "Y",
                "trends": [
                    {
                        "title": "Y1",
                        "detail": "d",
                        "signal": "rising",
                        "evidence": [ids["A new"], ids["B press"], ids["C old"]],
                    }
                ],
            },
        }
    )
    parsed = parse_response(raw, items, today)
    week = parsed["week"]["trends"][0]
    assert week["n_evidence"] == 1 and week["evidence"][0]["title"] == "A new"
    year = parsed["year"]["trends"][0]
    assert year["n_evidence"] == 3

    act = monthly_activity(items, today)
    months = list(dict.fromkeys(a["month"] for a in act))
    assert months[0] == "2025-12" and months[-1] == "2026-11" and len(months) == 12
    nov = {a["kind"]: a["count"] for a in act if a["month"] == "2026-11"}
    assert nov == {"Articles": 1, "Press releases": 0, "Clinical trials": 0}

    rows = trend_month_counts(parsed["year"]["trends"], months)
    by = {r["month"]: r["count"] for r in rows}
    assert by["2026-11"] == 1 and by["2026-10"] == 1 and by["2026-01"] == 1
    assert sum(by.values()) == 3 and len(rows) == 12

    # schema is strict JSON-schema (structured outputs need additionalProperties false)
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == {"week", "month", "year"}
