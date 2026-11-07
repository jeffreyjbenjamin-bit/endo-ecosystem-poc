"""Press releases are searched for, tagged, and kept among the briefing's sources."""

from src.pipelines.ai_search import (
    BRIEFING_SOURCES,
    MIN_PRESS_RELEASES,
    SEARCH_CATEGORIES,
    generate_search_queries,
    is_press_release,
    score_results,
)


def _r(title, host, url=None, snippet="", date=None):
    return {
        "title": title,
        "url": url or f"https://{host}/{abs(hash(title))}",
        "snippet": snippet,
        "source_name": host,
        "published_date": date,
    }


def test_new_category_added_without_losing_existing_ones():
    for cat in [
        "Clinical Trials",
        "Treatments & Drug Pipeline",
        "Diagnostics",
        "Therapeutics",
        "Epidemiology",
        "Basic Research",
        "Regulatory & Policy",
        "Surgery",
        "Patient Outcomes",
        "Industry & Press Releases",
    ]:
        assert cat in SEARCH_CATEGORIES
    all_q = generate_search_queries("endometriosis", 7, [])
    assert any("site:businesswire.com" in q for q in all_q)
    assert any("diagnostic biomarker" in q for q in all_q)  # existing queries kept
    only_press = generate_search_queries(
        "endometriosis", 7, ["Industry & Press Releases"]
    )
    assert any("phase 1 OR phase 2 OR phase 3" in q for q in only_press)


def test_press_release_detection():
    assert is_press_release(_r("X announces data", "businesswire.com"))
    assert is_press_release(
        _r(
            "X announces data",
            "ir.example.com",
            url="https://ir.example.com/news-releases/x",
        )
    )
    assert is_press_release(_r("Acme Press Release: endometriosis trial", "acme.com"))
    assert not is_press_release(
        _r("Cohort study of endometriosis", "pubmed.ncbi.nlm.nih.gov")
    )
    # Society / agency press offices are not company press releases
    assert not is_press_release(
        _r(
            "New study on IVF access",
            "asrm.org",
            url="https://www.asrm.org/news-and-events/asrm-news/press-releases/x",
        )
    )


def test_on_topic_press_releases_reach_the_briefing():
    journals = [
        _r(
            f"Endometriosis study {i}",
            "pubmed.ncbi.nlm.nih.gov",
            snippet="endometriosis randomized controlled trial biomarker",
        )
        for i in range(25)
    ]
    press = [
        _r(
            "Acme announces Phase 2 topline results in endometriosis",
            "businesswire.com",
            snippet="endometriosis phase 2",
        ),
        _r(
            "Beta Dx launches endometriosis blood test",
            "prnewswire.com",
            snippet="diagnostic test for endometriosis",
        ),
        _r(
            "Gamma doses first patient in Phase 1 endometriosis trial",
            "globenewswire.com",
            snippet="first patient dosed endometriosis",
        ),
        _r(
            "Delta reports quarterly earnings",
            "businesswire.com",
            snippet="revenue guidance",
        ),  # off-topic: must not be promoted
    ]
    ranked = score_results(journals + press, "endometriosis", 7)
    top = ranked[:BRIEFING_SOURCES]
    top_press = [r for r in top if r["is_press_release"]]
    assert len(top_press) == MIN_PRESS_RELEASES
    assert all("endometriosis" in r["title"].lower() for r in top_press)
    assert len(ranked) == len(journals) + len(press)  # nothing lost
    assert len({id(r) for r in ranked}) == len(ranked)  # nothing duplicated


def test_no_promotion_when_press_releases_already_rank():
    press = [
        _r(
            f"Acme endometriosis Phase 3 update {i}",
            "businesswire.com",
            snippet="endometriosis phase 3 fda approval",
        )
        for i in range(5)
    ]
    ranked = score_results(press, "endometriosis", 7)
    assert [r["title"] for r in ranked] == sorted(
        [r["title"] for r in ranked],
        key=lambda t: -next(x["composite_score"] for x in ranked if x["title"] == t),
    )
