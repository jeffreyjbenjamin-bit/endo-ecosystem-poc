"""Markdown coming back from the visual editor must render correctly in emails."""

from src.ui.wysiwyg import clean_markdown


def test_citations_and_underscores_unescaped():
    md = "Finding \\[2\\]\\[3\\] at https://x.example/a\\_b"
    assert clean_markdown(md) == "Finding [2][3] at https://x.example/a_b\n"


def test_hard_breaks_become_two_spaces():
    md = "**Endometriosis Intelligence**\\\n*Published June 27, 2026*"
    assert clean_markdown(md) == (
        "**Endometriosis Intelligence**  \n*Published June 27, 2026*\n"
    )


def test_autolinks_become_plain_urls():
    md = "[1] Title. PubMed. <https://pubmed.ncbi.nlm.nih.gov/41957705/>"
    assert clean_markdown(md) == (
        "[1] Title. PubMed. https://pubmed.ncbi.nlm.nih.gov/41957705/\n"
    )


def test_markdown_links_kept():
    md = "*→ [Read the study](https://example.org/s)*"
    assert clean_markdown(md) == md + "\n"
