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
