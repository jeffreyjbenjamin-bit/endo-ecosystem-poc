"""Clinical Trials tab: parsing and the two filters (no network)."""

from src.pipelines.clinical_trials import (
    confirmed_diagnosis_phrase,
    filter_trials,
    parse_study,
)


def _study(nct, sponsor_class="INDUSTRY", criteria="", countries=("United States",)):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}"},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "primaryCompletionDateStruct": {"date": "2027-06", "type": "ESTIMATED"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Acme Bio", "class": sponsor_class}
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2", "PHASE3"],
                "enrollmentInfo": {"count": 120, "type": "ESTIMATED"},
            },
            "eligibilityModule": {"eligibilityCriteria": criteria},
            "contactsLocationsModule": {
                "locations": [{"country": c} for c in countries]
            },
        }
    }


def test_parse_study_columns():
    t = parse_study(
        _study("NCT1", countries=("Canada", "United States", "United States"))
    )
    assert t["url"] == "https://clinicaltrials.gov/study/NCT1"
    assert t["source"] == "ClinicalTrials.gov"
    assert t["study_phase"] == "Interventional · Phase 2/Phase 3"
    assert t["status"] == "Recruiting"
    assert t["primary_completion"] == "2027-06"
    assert t["enrollment"] == 120
    assert t["countries"] == ["United States", "Canada"]  # US first
    assert t["us_sites"] == 2


def test_confirmed_diagnosis_matches():
    for line in [
        "Surgically and pathologically confirmed endometriosis",
        "Operative confirmation of endometriosis",
        "Histologically confirmed diagnosis of endometriosis via surgical pathology",
        "Endometriosis diagnosed by prior laparoscopy within 10 years",
    ]:
        crit = f"Inclusion Criteria:\n\n* Age 18-45\n* {line}\n\nExclusion Criteria:\n\n* None"
        assert confirmed_diagnosis_phrase(crit), line


def test_confirmed_diagnosis_rejects():
    for crit in [
        # surgery only planned
        "Inclusion Criteria:\n* Scheduled to undergo laparoscopy for diagnosis of endometriosis",
        # imaging / clinical diagnosis accepted instead
        "Inclusion Criteria:\n* Surgical, imaging, or clinical diagnosis of endometriosis",
        "Inclusion Criteria:\n* Laparoscopic surgery with biopsy-proven endometriosis, OR\n"
        "* Endometrioma on ultrasound/MRI",
        # only in the exclusion criteria
        "Inclusion Criteria:\n* Healthy women\n\nExclusion Criteria:\n"
        "* Surgically confirmed endometriosis",
        # confirmation about something else
        "Inclusion Criteria:\n* Histologically confirmed breast cancer",
    ]:
        assert confirmed_diagnosis_phrase(crit) is None, crit


def test_filters():
    confirmed = "Inclusion Criteria:\n* Operative confirmation of endometriosis"
    trials = [
        parse_study(_study("A", "INDUSTRY", confirmed)),
        parse_study(_study("B", "OTHER", confirmed)),
        parse_study(_study("C", "NIH")),
        parse_study(_study("D", "INDUSTRY")),
    ]
    ids = lambda ts: [t["id"] for t in ts]  # noqa: E731
    assert ids(filter_trials(trials)) == ["A", "B", "C", "D"]
    assert ids(filter_trials(trials, companies_only=True)) == ["A", "D"]
    assert ids(filter_trials(trials, confirmed_only=True)) == ["A", "B"]
    assert ids(filter_trials(trials, True, True)) == ["A"]


# ── New trials, briefing section, newsletter flags ────────────────────────────

import pytest  # noqa: E402

from src.pipelines import clinical_trials as ct  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "state_path", lambda name: str(tmp_path / name))
    return tmp_path


def _fake_fetch(monkeypatch, ids, sponsor="INDUSTRY"):
    trials = [parse_study(_study(i, sponsor)) for i in ids]
    monkeypatch.setattr(ct, "fetch_open_us_trials", lambda: trials)


def test_first_refresh_is_baseline_then_new_trials_flagged(cache, monkeypatch):
    _fake_fetch(monkeypatch, ["A", "B"])
    data = ct.refresh_trials()
    assert ct.unbriefed_trials(data) == []
    assert not any(t["is_new"] for t in data["trials"])

    _fake_fetch(monkeypatch, ["A", "B", "C"])
    data = ct.refresh_trials()
    assert [t["id"] for t in ct.unbriefed_trials(data)] == ["C"]
    assert [t["id"] for t in data["trials"] if t["is_new"]] == ["C"]
    # display-only fields are not saved
    raw = (cache / ct.CACHE_FILE).read_text(encoding="utf-8")
    assert "is_new" not in raw and "in_newsletter" not in raw


def test_briefing_section_and_mark_briefed(cache, monkeypatch):
    _fake_fetch(monkeypatch, ["A"])
    ct.refresh_trials()
    _fake_fetch(monkeypatch, ["A", "NCT999"])
    ct.refresh_trials()
    new = ct.unbriefed_trials()
    md = "## Intelligence Summary\nText [1]\n\n## References\n[1] Ref."
    out = ct.add_trials_to_briefing(md, new)
    assert "## New Clinical Trials" in out
    assert out.index("## New Clinical Trials") < out.index("## References")
    assert "https://clinicaltrials.gov/study/NCT999" in out
    assert ct.add_trials_to_briefing(md, []) == md  # nothing new: unchanged

    assert ct.mark_briefed(out) == ["NCT999"]
    assert ct.unbriefed_trials() == []
    # a later refresh does not make it new again
    ct.refresh_trials()
    assert ct.unbriefed_trials() == []


def test_newsletter_flags_survive_refresh(cache, monkeypatch):
    _fake_fetch(monkeypatch, ["A", "B"])
    ct.refresh_trials()
    ct.set_in_newsletter("B", True)
    ct.set_in_newsletter("A", True)
    assert [t["id"] for t in ct.newsletter_trials()] == ["B", "A"]
    ct.set_in_newsletter("B", False)
    assert [t["id"] for t in ct.newsletter_trials()] == ["A"]
    ct.refresh_trials()
    assert ct.load_trials()["trials"][0]["in_newsletter"] is True
    assert "Clinical trial A" in ct.trial_snippet(ct.newsletter_trials()[0])
    # a trial that closes drops off the newsletter list
    _fake_fetch(monkeypatch, ["B"])
    ct.refresh_trials()
    assert ct.newsletter_trials() == []
