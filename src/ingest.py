from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import requests
from urllib.parse import urlencode
import sys
import json

__all__ = ["http_get_json", "ping", "ctgov_search_endometriosis"]


def http_get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Basic GET helper that prefers JSON and falls back to text.
    Raises for non-2xx responses.
    """
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "EndoEcosystemPoC/0.1 (+github)"},
    )
    resp.raise_for_status()
    try:
        return resp.json()  # type: ignore[no-any-return]
    except ValueError:
        return {
            "_raw": resp.text,
            "_content_type": resp.headers.get("Content-Type", ""),
            "_status_code": resp.status_code,
        }


def ping(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Lightweight health check: returns ok + status_code only.
    Does NOT raise for non-2xx — just reports.
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "EndoEcosystemPoC/0.1 (+github)"},
        )
        return {"ok": resp.ok, "status_code": resp.status_code}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


class Trial(TypedDict, total=False):
    id: str
    title: str
    status: Optional[str]


def _normalize_v2_trial(item: Dict[str, Any]) -> Trial:
    """
    ClinicalTrials.gov v2 'studies' format → {id, title, status}
    Access defensively; schema may evolve.
    """
    try:
        ps = item.get("protocolSection", {}) if isinstance(item, dict) else {}
        ident = ps.get("identificationModule", {}) if isinstance(ps, dict) else {}
        status_mod = ps.get("statusModule", {}) if isinstance(ps, dict) else {}

        nct_id = ident.get("nctId") or item.get("nctId") or item.get("id")
        title = (
            ident.get("officialTitle")
            or ident.get("briefTitle")
            or item.get("officialTitle")
            or item.get("briefTitle")
        )
        status = status_mod.get("overallStatus")
        trial: Trial = {}
        if nct_id:
            trial["id"] = str(nct_id)
        if title:
            trial["title"] = str(title)
        if status:
            trial["status"] = str(status)
        return trial
    except Exception:
        return {}


def _normalize_v1_trial(item: Dict[str, Any]) -> Trial:
    """
    ClinicalTrials.gov v1 'study_fields' → {id, title, status}
    """

    def first(lst: Any) -> Optional[str]:
        return lst[0] if isinstance(lst, list) and lst else None

    trial: Trial = {}
    nct = first(item.get("NCTId"))
    title = first(item.get("BriefTitle")) or first(item.get("OfficialTitle"))
    status = first(item.get("OverallStatus"))
    if nct:
        trial["id"] = str(nct)
    if title:
        trial["title"] = str(title)
    if status:
        trial["status"] = str(status)
    return trial


def ctgov_search_endometriosis(
    base_v2: str, limit: int = 5, timeout: float = 10.0
) -> List[Trial]:
    """
    Try ClinicalTrials.gov v2 first; if it fails or is unexpected, fall back to v1.
    Returns a small normalized list of trials.
    """
    # --- v2 attempt
    try:
        base = base_v2.rstrip("/")
        # Official v2 docs: https://clinicaltrials.gov/data-api/api
        # /studies?query.term=...&pageSize=...
        v2_url = f"{base}/studies?{urlencode({'query.term': 'endometriosis', 'pageSize': int(limit)})}"
        resp = requests.get(
            v2_url,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "EndoEcosystemPoC/0.1",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies") if isinstance(data, dict) else None
        if isinstance(studies, list) and studies:
            out: List[Trial] = []
            for item in studies[:limit]:
                t = _normalize_v2_trial(item if isinstance(item, dict) else {})
                if t:
                    out.append(t)
            if out:
                return out
        # If schema not as expected, fall through to v1
    except Exception:
        pass

    # --- v1 fallback
    try:
        fields = ["NCTId", "BriefTitle", "OfficialTitle", "OverallStatus"]
        v1_url = (
            "https://clinicaltrials.gov/api/query/study_fields?"
            f"expr=endometriosis&fields={','.join(fields)}&min_rnk=1&max_rnk={int(limit)}&fmt=json"
        )
        resp = requests.get(
            v1_url,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "EndoEcosystemPoC/0.1",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        sfr = data.get("StudyFieldsResponse", {})
        items = sfr.get("StudyFields", []) if isinstance(sfr, dict) else []
        out: List[Trial] = []
        for item in items:
            t = _normalize_v1_trial(item if isinstance(item, dict) else {})
            if t:
                out.append(t)
        return out
    except Exception as exc:
        return [
            {
                "id": "ERROR",
                "title": f"ClinicalTrials fetch failed: {exc}",
                "status": None,
            }
        ]


if __name__ == "__main__":
    # Quick local sanity run:
    base_v2 = "https://clinicaltrials.gov/api/v2"
    results = ctgov_search_endometriosis(base_v2, limit=5)
    print(json.dumps(results, indent=2))
    sys.exit(0)
