"""Open endometriosis trials with a US site, from ClinicalTrials.gov.

The Clinical Trials tab calls :func:`refresh_trials` (Refresh button) and
:func:`load_trials` (on every page load). Results are cached as JSON in the app
state directory so the table survives restarts without hitting the API.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.common.paths import state_path

CTGOV_API = os.getenv("CTGOV_BASE", "https://clinicaltrials.gov/api/v2/studies")
SOURCE_NAME = "ClinicalTrials.gov"
CACHE_FILE = "clinical_trials.json"

# "Open" = still taking (or about to take) participants.
OPEN_STATUSES = ("RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION")
US = "United States"

_FIELDS = "|".join(
    [
        "IdentificationModule",
        "StatusModule",
        "SponsorCollaboratorsModule",
        "DesignModule",
        "EligibilityModule",
        "ContactsLocationsModule",
    ]
)

# Inclusion-criteria wording that requires a surgically / pathologically
# confirmed endometriosis diagnosis.
_CONFIRMED_PATTERNS = [
    r"surgical(?:ly)?\s*(?:and\s*/?\s*or|and|or|/|,)?\s*(?:histo)?patholog\w*",
    r"(?:surgical(?:ly)?|laparoscopic(?:ally)?|histolog\w*|histopatholog\w*|"
    r"patholog\w*|biopsy|operative(?:ly)?|visual(?:ly)?)[\w\s,/-]{0,40}?"
    r"(?:confirm\w*|prove[nd]?|diagnos\w*|document\w*|verifi\w*)",
    r"(?:confirm\w*|prove[nd]?|diagnos\w*|document\w*|verifi\w*)[\w\s,/-]{0,30}?"
    r"(?:by|via|through|at|during|on)\s+(?:a\s+|prior\s+|previous\s+)?"
    r"(?:surgery|surgical|laparoscop\w*|laparotomy|histolog\w*|patholog\w*|biopsy|"
    r"operative|operation)",
    r"operative\s+confirmation",
]
_CONFIRMED_RE = re.compile("|".join(f"(?:{p})" for p in _CONFIRMED_PATTERNS), re.I)
_EXCLUSION_HEADER = re.compile(r"^\s*\W*\s*exclusion\s+criteria", re.I | re.M)

_STATUS_LABELS = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
}
_SPONSOR_LABELS = {
    "INDUSTRY": "Industry",
    "NIH": "NIH",
    "FED": "U.S. federal",
    "OTHER_GOV": "Government",
    "NETWORK": "Network",
    "INDIV": "Individual",
    "OTHER": "Academic / other",
    "AMBIG": "Unknown",
    "UNKNOWN": "Unknown",
}


def _titlecase(code: str | None) -> str:
    return (code or "").replace("_", " ").title()


def inclusion_text(criteria: str | None) -> str:
    """The inclusion part of an eligibility-criteria block (before Exclusion)."""
    text = criteria or ""
    m = _EXCLUSION_HEADER.search(text)
    return text[: m.start()] if m else text


# A confirmation line does not count when surgery is only planned, or when a
# non-surgical diagnosis (imaging, clinical) is an accepted alternative.
_NOT_CONFIRMED_RE = re.compile(
    r"\b(?:scheduled|planned|plan(?:ning)? to|undergo(?:ing)?|suspect\w*|"
    r"imaging|ultrasound|sonograph\w*|mri|clinical(?:ly)?\s+(?:diagnos\w*|suspect\w*)|"
    r"clinical\s+diagnosis|or\s+clinical)\b",
    re.I,
)


def confirmed_diagnosis_phrase(criteria: str | None) -> str | None:
    """The inclusion-criteria line requiring a confirmed diagnosis, or None.

    Only lines about endometriosis count. Lines where surgery is still to come
    ("scheduled to undergo laparoscopy") or where imaging or a clinical
    diagnosis is accepted instead are not a confirmed-diagnosis requirement.
    """
    lines = [ln for ln in re.split(r"\n+", inclusion_text(criteria)) if ln.strip()]
    for i, line in enumerate(lines):
        if not re.search(r"endometrio|lesion|deep infiltrating", line, re.I):
            continue
        if not _CONFIRMED_RE.search(line):
            continue
        scope = line
        if re.search(r"\bor[\s,;:]*$", line, re.I) and i + 1 < len(lines):
            scope += " " + lines[i + 1]  # "biopsy-proven, OR <alternative>"
        if _NOT_CONFIRMED_RE.search(scope):
            continue
        line = re.sub(r"\\([<>*_])", r"\1", line)  # CT.gov markdown escapes
        return re.sub(r"^\s*[\*\-\u2022\d.)]+\s*", "", line).strip()
    return None


def _phase(design: dict) -> str:
    study_type = _titlecase(design.get("studyType")) or "—"
    phases = [p for p in design.get("phases") or [] if p != "NA"]
    if not phases:
        return study_type
    label = "/".join(p.replace("EARLY_PHASE1", "Early Phase 1") for p in phases)
    label = re.sub(r"PHASE(\d)", r"Phase \1", label)
    return f"{study_type} · {label}"


def parse_study(study: dict) -> dict:
    sec = study.get("protocolSection", {})
    ident = sec.get("identificationModule", {})
    stat = sec.get("statusModule", {})
    sponsor = sec.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    design = sec.get("designModule", {})
    elig = sec.get("eligibilityModule", {})
    locs = sec.get("contactsLocationsModule", {}).get("locations") or []
    nct = ident.get("nctId", "")

    countries = sorted({loc.get("country") for loc in locs if loc.get("country")})
    if US in countries:  # US first, then the rest alphabetically
        countries = [US] + [c for c in countries if c != US]
    us_sites = sum(1 for loc in locs if loc.get("country") == US)

    pcd = stat.get("primaryCompletionDateStruct") or {}
    enroll = design.get("enrollmentInfo") or {}
    criteria = elig.get("eligibilityCriteria") or ""
    phrase = confirmed_diagnosis_phrase(criteria)

    return {
        "id": nct,
        "url": f"https://clinicaltrials.gov/study/{nct}",
        "source": SOURCE_NAME,
        "title": ident.get("briefTitle") or ident.get("officialTitle") or "",
        "lead_sponsor": sponsor.get("name") or "",
        "sponsor_class": sponsor.get("class") or "UNKNOWN",
        "sponsor_type": _SPONSOR_LABELS.get(sponsor.get("class"), "Unknown"),
        "is_company": sponsor.get("class") == "INDUSTRY",
        "study_phase": _phase(design),
        "status": _STATUS_LABELS.get(
            stat.get("overallStatus"), _titlecase(stat.get("overallStatus"))
        ),
        "primary_completion": pcd.get("date") or "",
        "primary_completion_type": _titlecase(pcd.get("type")),
        "enrollment": enroll.get("count"),
        "enrollment_type": _titlecase(enroll.get("type")),
        "countries": countries,
        "us_sites": us_sites,
        "confirmed_dx": phrase is not None,
        "confirmed_dx_phrase": phrase or "",
        "last_update": (stat.get("lastUpdatePostDateStruct") or {}).get("date", ""),
    }


def fetch_open_us_trials(
    condition: str = "endometriosis", timeout: int = 60
) -> list[dict]:
    """All open trials for the condition with at least one US site."""
    params = {
        "query.cond": condition,
        "query.locn": US,
        "filter.overallStatus": ",".join(OPEN_STATUSES),
        "fields": _FIELDS,
        "pageSize": 1000,
    }
    studies: list[dict] = []
    while True:
        r = requests.get(CTGOV_API, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        studies.extend(data.get("studies", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token

    trials = [parse_study(s) for s in studies]
    # query.locn is a text match; keep only trials with an actual US site.
    trials = [t for t in trials if US in t["countries"]]
    trials.sort(key=lambda t: (t["primary_completion"] or "9999", t["id"]))
    return trials


def refresh_trials() -> dict:
    """Run the search, cache it, and return {"refreshed_at", "trials"}."""
    payload = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_NAME,
        "trials": fetch_open_us_trials(),
    }
    path = Path(state_path(CACHE_FILE))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return payload


def load_trials() -> dict | None:
    """The last cached search, or None if Refresh has never run."""
    try:
        return json.loads(Path(state_path(CACHE_FILE)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def filter_trials(
    trials: list[dict], companies_only: bool = False, confirmed_only: bool = False
) -> list[dict]:
    out = trials
    if companies_only:
        out = [t for t in out if t["is_company"]]
    if confirmed_only:
        out = [t for t in out if t["confirmed_dx"]]
    return out
