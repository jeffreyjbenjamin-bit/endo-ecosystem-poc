"""Source citations and the Build Newsletter topic search (no network)."""

import numpy as np
import pandas as pd

from src.common.citations import (
    _ncbi_ids,
    authors_list,
    citation_from_html,
    citation_from_library,
    clean_doi,
    format_citation,
    title_case,
)


def test_library_citation_formats():
    row = {
        "source": "pubmed",
        "authors": np.array([{"name": "Hong D"}, {"name": "Xiao Y"}], dtype=object),
        "journal_or_venue": "Frontiers in pharmacology",
        "doi": "10.3389/fphar.2025.1656217",
        "published_date": "2025",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
    }
    c = citation_from_library(row)
    assert format_citation(c) == (
        "Hong D, Xiao Y. Frontiers in Pharmacology, 2025. "
        "doi:10.3389/fphar.2025.1656217. Indexed in PubMed."
    )
    many = {
        "authors": ["HAGIHARA, M., Ohkuma, R., Fujimaki, T., Tamaki, M."],
        "source": "biorxiv",
        "journal_or_venue": "bioRxiv",
        "published_date": "2025",
    }
    assert format_citation(citation_from_library(many)).startswith(
        "HAGIHARA M, Ohkuma R, Fujimaki T, et al. bioRxiv, 2025."
    )
    trial = {"source": "ctgov", "published_date": "2025-08-28"}
    assert format_citation(citation_from_library(trial)) == "ClinicalTrials.gov, 2025."


def test_helpers():
    assert clean_doi("https://doi.org/10.1016/x") == "10.1016/x"
    assert clean_doi("not a doi") == ""
    assert authors_list("Smith J; Lee K and Wu T") == ["Smith J", "Lee K", "Wu T"]
    assert title_case("Journal of minimally invasive gynecology") == (
        "Journal of Minimally Invasive Gynecology"
    )
    assert title_case("The BMJ") == "The BMJ"
    assert _ncbi_ids("https://pubmed.ncbi.nlm.nih.gov/41957705/?utm=x") == (
        "pubmed",
        "41957705",
    )
    assert _ncbi_ids("https://pmc.ncbi.nlm.nih.gov/articles/PMC13087501/") == (
        "pmc",
        "13087501",
    )
    assert _ncbi_ids("https://nature.com/a") is None


def test_citation_from_page_meta():
    html = """<head>
      <meta name="citation_title" content="T">
      <meta name="citation_author" content="Tiniakou, Ioanna">
      <meta name="citation_author" content="Bafligil, Cemsel">
      <meta name="citation_journal_title" content="Communications Medicine">
      <meta name="dc.publisher" content="Nature Publishing Group">
      <meta name="citation_publication_date" content="2026/05/20">
      <meta name="citation_doi" content="10.1038/s43856-026-01641-3">
    </head>"""
    c = citation_from_html(html, "https://www.nature.com/articles/x")
    assert format_citation(c) == (
        "Tiniakou I, Bafligil C. Communications Medicine (Nature Publishing Group), "
        "2026. doi:10.1038/s43856-026-01641-3."
    )
    # nothing found: fall back to the known publisher name
    assert format_citation(None, "thelancet.com") == (
        "Published by The Lancet (Elsevier) (thelancet.com)"
    )
    assert format_citation(None, "example.org") == "Published by example.org"


def test_topic_search_library_results(monkeypatch):
    from src.ui.newsletter_from_briefing import citation_text, topic_search

    df = pd.DataFrame(
        [
            {
                "uid": "u1",
                "title": "Adenomyosis MRI study",
                "url": "https://pubmed/9",
                "source": "pubmed",
                "published_date": "2026-03-01",
                "abstract": "Body",
                "authors": [{"name": "Lee K"}],
                "journal_or_venue": "Radiology",
                "doi": "10.1/abc",
            },
        ]
    )
    queries = []
    deps = {
        "load_docs_df": lambda: df,
        "load_sqlite_vectors": lambda: (["u1"], np.ones((1, 3))),
        "embed_text": lambda q: queries.append(q) or np.ones(3),
        "cosine_search": lambda vec, uids, mat, k: [("u1", 0.88)],
    }
    arts, errors = topic_search("adenomyosis imaging", deps, "Evidence library", 365)
    assert errors == [] and queries == ["adenomyosis imaging endometriosis"]
    a = arts[0]
    assert a["ref"] == 1 and a["origin"] == "library" and a["score"] == 0.88
    assert a["key"].startswith("topic_lib_") and a["cited"] is False
    assert (
        citation_text(a) == "Lee K. Radiology, 2026. doi:10.1/abc. Indexed in PubMed."
    )
