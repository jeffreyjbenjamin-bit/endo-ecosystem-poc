from __future__ import annotations

from typing import Any, Dict, List
import ingest


class _DummyResp:
    def __init__(self, status_code: int, json_obj: Dict[str, Any] | None = None):
        self.status_code = status_code
        self._json_obj = json_obj
        self.ok = 200 <= status_code < 300
        self.headers = {"Content-Type": "application/json"}
        self.text = "x"

    def json(self) -> Dict[str, Any]:
        if self._json_obj is None:
            raise ValueError("no json")
        return self._json_obj

    def raise_for_status(self) -> None:
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}")


def test_ctgov_v2_happy_path(monkeypatch):
    # Simulate a minimal v2 shape
    v2_payload = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000001",
                        "officialTitle": "Endometriosis Study A",
                    },
                    "statusModule": {"overallStatus": "Recruiting"},
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000002",
                        "briefTitle": "Endo Study B",
                    },
                    "statusModule": {"overallStatus": "Completed"},
                }
            },
        ]
    }

    def _fake_get(url, *args, **kwargs):
        assert "/studies" in url
        return _DummyResp(200, v2_payload)

    monkeypatch.setattr("requests.get", _fake_get)
    trials = ingest.ctgov_search_endometriosis(
        "https://clinicaltrials.gov/api/v2", limit=2
    )
    assert isinstance(trials, list) and len(trials) == 2
    assert trials[0]["id"] == "NCT00000001"
    assert "title" in trials[0]
    assert trials[0]["status"] == "Recruiting"


def test_ctgov_v2_falls_back_to_v1(monkeypatch):
    # First call (v2) fails; second call (v1) succeeds
    calls: List[str] = []

    def _fake_get(url, *args, **kwargs):
        calls.append(url)
        if "/studies" in url:
            return _DummyResp(500, None)  # force v2 failure
        # v1 response structure
        v1_payload = {
            "StudyFieldsResponse": {
                "StudyFields": [
                    {
                        "NCTId": ["NCT123"],
                        "BriefTitle": ["Endo Trial X"],
                        "OverallStatus": ["Recruiting"],
                    }
                ]
            }
        }
        return _DummyResp(200, v1_payload)

    monkeypatch.setattr("requests.get", _fake_get)
    trials = ingest.ctgov_search_endometriosis(
        "https://clinicaltrials.gov/api/v2", limit=1
    )
    assert len(trials) == 1
    assert trials[0]["id"] == "NCT123"
    # Ensure we tried v2 first, then v1
    assert any("/studies" in u for u in calls)
    assert any("/api/query/study_fields" in u for u in calls)
