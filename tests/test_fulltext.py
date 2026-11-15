"""Phase 1 content access: PubMed abstracts, OpenAlex enrichment, Europe PMC
open-access full text, and their use in newsletter prompts (no network)."""

import pandas as pd

from src.common import fulltext as ft
from src.connectors.pubmed import parse_efetch_xml
from src.pipelines import enrich_library as el

EFETCH_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID>
<Article><Abstract>
<AbstractText Label="BACKGROUND">Endometriosis is common.</AbstractText>
<AbstractText Label="RESULTS">Pain fell by <i>30</i>%.</AbstractText>
</Abstract><PublicationTypeList><PublicationType>Randomized Controlled Trial
</PublicationType></PublicationTypeList></Article>
<MeshHeadingList><MeshHeading><DescriptorName>Endometriosis</DescriptorName>
</MeshHeading></MeshHeadingList></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC999</ArticleId>
</ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article><Abstract>
<AbstractText><b>Background</b>No label here.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_parse_efetch_abstracts():
    recs = parse_efetch_xml(EFETCH_XML)
    a = recs["111"]
    assert (
        a["abstract"]
        == "Background: Endometriosis is common.\nResults: Pain fell by 30 %."
    )
    assert a["mesh_terms"] == ["Endometriosis"] and a["pmcid"] == "PMC999"
    assert a["publication_types"] == ["Randomized Controlled Trial"]
    assert (
        recs["222"]["abstract"] == "Background No label here."
    )  # inline heading spaced


def test_openalex_inverted_index():
    inv = {"Pain": [0], "fell": [1], "sharply": [2], "again": [4], "and": [3]}
    assert el.abstract_from_inverted_index(inv) == "Pain fell sharply and again"
    assert el.abstract_from_inverted_index(None) == ""


def test_enrich_library_fills_only_blanks(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "uid": "p1",
                "source": "pubmed",
                "source_id": "111",
                "abstract": None,
                "doi": "10.1/a",
                "mesh_terms": None,
            },
            {
                "uid": "p2",
                "source": "pubmed",
                "source_id": "333",
                "abstract": "Kept",
                "doi": None,
                "mesh_terms": None,
            },
            {
                "uid": "c1",
                "source": "crossref",
                "source_id": "x",
                "abstract": "",
                "doi": "https://doi.org/10.1/B",
                "mesh_terms": None,
            },
        ]
    )
    monkeypatch.setattr(
        el.pubmed,
        "fetch_abstracts",
        lambda ids: {
            "111": {
                "abstract": "New abstract",
                "mesh_terms": ["M"],
                "publication_types": ["Review"],
                "pmcid": "PMC1",
            },
            "333": {
                "abstract": "Would overwrite",
                "mesh_terms": [],
                "publication_types": [],
                "pmcid": "",
            },
        },
    )
    monkeypatch.setattr(
        el,
        "fetch_openalex_by_doi",
        lambda dois: {
            "10.1/a": {
                "cited_by_count": 7,
                "is_oa": True,
                "oa_url": "u",
                "abstract": "OA abstract",
            },
            "10.1/b": {
                "cited_by_count": 2,
                "is_oa": False,
                "oa_url": "",
                "abstract": "Crossref filled",
            },
        },
    )
    changed = el.enrich(df, log=lambda *a: None)
    assert set(changed) == {"p1", "c1"}
    assert df.loc[0, "abstract"] == "New abstract"  # PubMed first, not OpenAlex
    assert df.loc[1, "abstract"] == "Kept"  # existing abstracts never overwritten
    assert df.loc[2, "abstract"] == "Crossref filled"
    assert df.loc[0, "cited_by_count"] == 7 and df.loc[0, "pmcid"] == "PMC1"
    assert df.loc[0, "mesh_terms"] == ["M"]


JATS = """<article><front><article-meta><abstract><p>Short abstract.</p></abstract>
</article-meta></front><body>
<sec><title>Methods</title><p>We enrolled 42 people.</p></sec>
<sec><title>Results</title><p>Sensitivity was 90%.</p></sec>
<sec><title>Discussion</title><p>This is promising.</p>
<p>Our study has several limitations, including a small sample.</p>
<sec><title>Strengths and limitations</title><p>Single centre.</p></sec></sec>
</body></article>"""


def test_parse_jats_and_excerpt():
    secs = ft.parse_jats(JATS)
    assert secs["abstract"] == "Short abstract."
    assert secs["methods"] == "We enrolled 42 people."
    assert "Sensitivity was 90%." in secs["results"]
    assert (
        "small sample" in secs["limitations"]
        and "Single centre." in secs["limitations"]
    )
    text = ft.excerpt({"sections": secs}, 2000)
    assert (
        text.index("[Results]") < text.index("[Limitations]") < text.index("[Methods]")
    )


def test_article_ids():
    assert ft.article_ids({"url": "https://pubmed.ncbi.nlm.nih.gov/123/"}) == {
        "pmid": "123"
    }
    assert ft.article_ids({"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC77/"}) == {
        "pmcid": "PMC77"
    }
    assert ft.article_ids({"url": "https://doi.org/10.1/x.y"})["doi"] == "10.1/x.y"
    assert ft.article_ids(
        {"url": "https://a.com/x", "citation": {"doi": "10.2/z"}}
    ) == {"doi": "10.2/z"}
    assert (
        ft.article_ids({"source": "pubmed", "source_id": "55", "url": ""})["pmid"]
        == "55"
    )


def test_enrich_articles_full_text_and_abstract(monkeypatch):
    def fake_full(a):
        if a["title"] == "OA":
            return {
                "pmcid": "PMC1",
                "license": "CC BY",
                "source": "Europe PMC (open access)",
                "sections": {"abstract": "A" * 500, "results": "R", "limitations": "L"},
            }
        return None

    monkeypatch.setattr(ft, "full_text", fake_full)
    monkeypatch.setattr(ft, "best_abstract", lambda a: "Abstract from Europe PMC" * 5)
    arts = [
        {"title": "OA", "snippet": "short"},
        {"title": "Paywalled", "snippet": "x"},
        {"title": "Trial", "origin": "clinical_trial", "snippet": "keep"},
    ]
    n = ft.enrich_articles(arts, [3000, 3000, 3000])
    assert n == 1
    assert "[Results] R" in arts[0]["full_text_excerpt"]
    assert arts[0]["full_text_license"] == "CC BY" and arts[0]["snippet"] == "A" * 500
    assert arts[1]["snippet"].startswith("Abstract from Europe PMC")
    assert "full_text_excerpt" not in arts[1] and arts[2]["snippet"] == "keep"


def test_full_text_reaches_prompt_and_chat():
    from src.ui.newsletter_from_briefing import _article_block, sources_block

    a = {
        "title": "T",
        "url": "https://x",
        "source": "pubmed",
        "snippet": "S",
        "full_text_excerpt": "[Results] Pain fell.\n[Limitations] Small.",
        "full_text_license": "CC BY",
        "full_text_source": "Europe PMC, PMC1",
    }
    block = _article_block(1, a)
    assert "Full text excerpt (open access, CC BY, Europe PMC, PMC1)" in block
    assert "      [Limitations] Small." in block
    src = [
        {
            "role": "featured",
            "title": "T",
            "source": "",
            "date": "",
            "url": "",
            "summary": "S",
            "full_text": "[Results] Pain fell.",
            "full_text_license": "CC BY",
        }
    ]
    assert "Full text excerpt (open access, CC BY):" in sources_block(src)
