"""Open endometriosis trials with a US site, from ClinicalTrials.gov.

The Clinical Trials tab calls :func:`refresh_trials` (Refresh button) and
:func:`load_trials` (on every page load). Results are cached as JSON in the app
state directory so the table survives restarts without hitting the API.

scheduler.py refreshes the list once a day. Trials that appear after the first
search are "new": every Intelligence Briefing gets a New Clinical Trials section
for them until a daily email has carried them (:func:`mark_briefed`). Trials
flagged "Include in newsletter" are offered in Build Newsletter from Intel
Briefing.
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
        "DescriptionModule",
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
        "first_posted": (stat.get("studyFirstPostDateStruct") or {}).get("date", ""),
        "summary": " ".join(
            (sec.get("descriptionModule", {}).get("briefSummary") or "").split()
        )[:1500],
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


# ── Cache ─────────────────────────────────────────────────────────────────────
# clinical_trials.json holds:
#   refreshed_at    last successful search (UTC ISO)
#   trials          the current open trials (parse_study rows)
#   history         {nct: {"first_seen", "baseline", "briefed"}} for every trial
#                   ever listed, so a trial is announced as new only once
#   newsletter_ids  trials flagged "Include in newsletter"

NEW_BADGE_DAYS = 14


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cache_path() -> Path:
    return Path(state_path(CACHE_FILE))


def _save(payload: dict) -> None:
    path = _cache_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_trials() -> dict | None:
    """The last cached search, or None if Refresh has never run."""
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data.setdefault("history", {})
    data.setdefault("newsletter_ids", [])
    _annotate(data)
    return data


def _annotate(data: dict) -> None:
    """Add is_new / in_newsletter to each trial row (not saved)."""
    hist = data.get("history") or {}
    nl = set(data.get("newsletter_ids") or [])
    cutoff = datetime.now(timezone.utc).timestamp() - NEW_BADGE_DAYS * 86400
    for t in data.get("trials") or []:
        h = hist.get(t["id"]) or {}
        first = h.get("first_seen") or ""
        t["first_seen"] = first
        try:
            recent = datetime.fromisoformat(first).timestamp() >= cutoff
        except ValueError:
            recent = False
        t["is_new"] = bool(h) and not h.get("baseline") and recent
        t["in_newsletter"] = t["id"] in nl


def merge_refresh(old: dict | None, trials: list[dict], now: str) -> dict:
    """Combine a fresh search with the previous cache.

    The first search ever is the baseline: those trials are not "new". After
    that, a trial not seen before gets first_seen=now and briefed=False until a
    briefing includes it. Newsletter flags are kept for trials still listed.
    """
    old = old or {}
    baseline = not old.get("trials") and not old.get("history")
    hist = dict(old.get("history") or {})
    # Caches written before history existed: their trials are the baseline.
    for t in old.get("trials") or []:
        hist.setdefault(
            t["id"],
            {
                "first_seen": old.get("refreshed_at", now),
                "baseline": True,
                "briefed": True,
            },
        )
    for t in trials:
        if t["id"] not in hist:
            hist[t["id"]] = {
                "first_seen": now,
                "baseline": baseline,
                "briefed": baseline,
            }
    ids = {t["id"] for t in trials}
    return {
        "refreshed_at": now,
        "source": SOURCE_NAME,
        "trials": trials,
        "history": hist,
        "newsletter_ids": [i for i in old.get("newsletter_ids") or [] if i in ids],
    }


def refresh_trials() -> dict:
    """Run the search, merge it into the cache, and return the cache."""
    trials = fetch_open_us_trials()
    payload = merge_refresh(load_trials(), trials, _now_iso())
    _save_clean(payload)
    _annotate(payload)
    return payload


def refreshed_today(tz_name: str | None = None) -> bool:
    """True if the cache was refreshed today (in the send time zone)."""
    data = load_trials()
    if not data or not data.get("refreshed_at"):
        return False
    tz = _tz(tz_name)
    then = datetime.fromisoformat(data["refreshed_at"]).astimezone(tz)
    return then.date() == datetime.now(tz).date()


def _tz(tz_name: str | None):
    from zoneinfo import ZoneInfo

    name = tz_name or os.getenv("SEND_TIMEZONE", "").strip() or "America/New_York"
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


# ── New trials → Intelligence Briefing ───────────────────────────────────────

BRIEFING_SECTION = "New Clinical Trials"


def unbriefed_trials(data: dict | None = None) -> list[dict]:
    """Open trials first listed since the baseline that no briefing has included."""
    data = data if data is not None else load_trials()
    if not data:
        return []
    hist = data.get("history") or {}
    return [
        t
        for t in data.get("trials") or []
        if not (hist.get(t["id"]) or {}).get("briefed", True)
    ]


def briefing_section_md(trials: list[dict]) -> str:
    """Markdown for the briefing's New Clinical Trials section (card layout)."""
    if not trials:
        return ""
    parts = [f"## {BRIEFING_SECTION}", ""]
    for t in trials:
        title = t["title"].replace("[", "(").replace("]", ")")
        enroll = (
            f"; enrollment {t['enrollment']:,} ({t['enrollment_type'].lower()})"
            if t.get("enrollment") is not None
            else ""
        )
        pcd = (
            f"; primary completion {t['primary_completion']}"
            f" ({t['primary_completion_type'].lower()})"
            if t.get("primary_completion")
            else ""
        )
        sites = f"; {t['us_sites']} US site{'s' if t['us_sites'] != 1 else ''}"
        dx = (
            " Inclusion requires a surgically or pathologically confirmed diagnosis."
            if t.get("confirmed_dx")
            else ""
        )
        parts += [
            f"**{title}** Sponsored by {t['lead_sponsor']} ({t['sponsor_type']}). "
            f"{t['study_phase']}; {t['status'].lower()}{enroll}{pcd}{sites}.{dx}"
            f" Newly listed on {SOURCE_NAME}.",
            f"*→ [{t['id']} on {SOURCE_NAME}]({t['url']})*",
            "",
        ]
    return "\n".join(parts).rstrip() + "\n"


def add_trials_to_briefing(briefing_md: str, trials: list[dict]) -> str:
    """Insert the New Clinical Trials section before References (or at the end)."""
    section = briefing_section_md(trials)
    if not section:
        return briefing_md
    m = re.search(r"^##\s+References\b", briefing_md, re.M)
    if m:
        return briefing_md[: m.start()] + section + "\n" + briefing_md[m.start() :]
    return briefing_md.rstrip() + "\n\n" + section


def mark_briefed(briefing_md: str) -> list[str]:
    """Mark new trials that appear in an emailed briefing so they are not repeated."""
    data = load_trials()
    if not data or not briefing_md:
        return []
    done = [t["id"] for t in unbriefed_trials(data) if t["id"] in briefing_md]
    for i in done:
        data["history"][i]["briefed"] = True
    if done:
        _save_clean(data)
    return done


def _save_clean(data: dict) -> None:
    """Save without the display-only fields _annotate adds."""
    clean = dict(data)
    clean["trials"] = [
        {
            k: v
            for k, v in t.items()
            if k not in ("first_seen", "is_new", "in_newsletter")
        }
        for t in data.get("trials") or []
    ]
    _save(clean)


# ── Newsletter flags ──────────────────────────────────────────────────────────


def set_in_newsletter(nct_id: str, include: bool) -> None:
    data = load_trials()
    if not data:
        return
    ids = [i for i in data["newsletter_ids"] if i != nct_id]
    if include:
        ids.append(nct_id)
    data["newsletter_ids"] = ids
    _save_clean(data)


def newsletter_trials() -> list[dict]:
    """Trials flagged for the newsletter, in the order they were flagged."""
    data = load_trials()
    if not data:
        return []
    by_id = {t["id"]: t for t in data["trials"]}
    return [by_id[i] for i in data["newsletter_ids"] if i in by_id]


def trial_snippet(t: dict) -> str:
    """Plain-text description of a trial for the newsletter writer."""
    bits = [
        f"Clinical trial {t['id']}, sponsored by {t['lead_sponsor']} ({t['sponsor_type']}).",
        f"{t['study_phase']}; status {t['status'].lower()}.",
    ]
    if t.get("enrollment") is not None:
        bits.append(f"Enrollment {t['enrollment']:,} ({t['enrollment_type'].lower()}).")
    if t.get("primary_completion"):
        bits.append(
            f"Primary completion {t['primary_completion']} "
            f"({t['primary_completion_type'].lower()})."
        )
    bits.append(f"Sites: {', '.join(t.get('countries') or [])}.")
    if t.get("confirmed_dx"):
        bits.append(f"Inclusion criteria: {t['confirmed_dx_phrase']}.")
    if t.get("summary"):
        bits.append(f"Summary: {t['summary']}")
    return " ".join(bits)


def filter_trials(
    trials: list[dict], companies_only: bool = False, confirmed_only: bool = False
) -> list[dict]:
    out = trials
    if companies_only:
        out = [t for t in out if t["is_company"]]
    if confirmed_only:
        out = [t for t in out if t["confirmed_dx"]]
    return out
