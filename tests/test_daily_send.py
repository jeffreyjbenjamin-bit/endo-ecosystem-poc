"""Tests for the daily send schedule in send_daily_briefing.py (no real email is sent)."""

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import send_daily_briefing as sdb

ET = ZoneInfo("America/New_York")


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point the script at throwaway databases with one briefing subscriber."""
    ml = tmp_path / "mailing_list.sqlite"
    conn = sqlite3.connect(ml)
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE recipients (id INTEGER PRIMARY KEY, name TEXT, email TEXT, "
        "added_ts TEXT, last_sent_ts TEXT, subscription_type TEXT)"
    )
    conn.execute(
        "INSERT INTO recipients VALUES (1,'Test','t@example.test','',NULL,'briefing')"
    )
    conn.execute("INSERT INTO config VALUES ('daily_send_time','07:00')")
    conn.commit()
    conn.close()

    ai = tmp_path / "ai_search_runs.sqlite"
    conn = sqlite3.connect(ai)
    conn.execute(
        "CREATE TABLE ai_search_runs (id INTEGER PRIMARY KEY, run_ts TEXT, topic TEXT, "
        "briefing_text TEXT, sources_json TEXT)"
    )
    conn.execute("INSERT INTO ai_search_runs VALUES (1,'','t','## Brief\nText','[]')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(sdb, "MAILING_LIST_DB", str(ml))
    monkeypatch.setattr(sdb, "AI_SEARCH_DB", str(ai))
    monkeypatch.setattr(sdb, "NEWSLETTERS_DB", str(tmp_path / "newsletters.sqlite"))
    sent = []
    monkeypatch.setattr(sdb, "send_briefing", lambda **kw: sent.append(kw["to_addr"]))
    monkeypatch.setattr(sdb, "smtp_configured", lambda: True)
    monkeypatch.setattr(sdb, "_run_ai_search", lambda: True)
    return sent


def at(monkeypatch, hh, mm):
    monkeypatch.setattr(sdb, "_now", lambda: datetime(2026, 11, 9, hh, mm, tzinfo=ET))


@pytest.mark.parametrize(
    "hh,mm,due",
    [(6, 59, False), (7, 0, True), (7, 45, True), (9, 0, True), (9, 1, False)],
)
def test_send_window(state, monkeypatch, hh, mm, due):
    at(monkeypatch, hh, mm)
    assert sdb.send_status()[0] is due


def test_sends_once_per_day(state, monkeypatch):
    at(monkeypatch, 7, 0)
    sdb.main(auto_search=True)
    assert state == ["t@example.test"]
    at(monkeypatch, 7, 5)
    sdb.main(auto_search=True)  # second check the same day: no resend
    assert state == ["t@example.test"]
    assert sdb._get_config("last_daily_send_date") == "2026-11-09"


def test_not_due_does_not_send(state, monkeypatch):
    at(monkeypatch, 6, 0)
    sdb.main(auto_search=True)
    assert state == []
    assert sdb._get_config("last_daily_send_date") == ""


def test_force_ignores_schedule_and_guard(state, monkeypatch):
    at(monkeypatch, 3, 0)
    sdb.main(force=True)
    assert state == ["t@example.test"]
    assert sdb._get_config("last_daily_send_date") == ""  # manual sends don't count


def test_send_timezone_is_used(monkeypatch):
    monkeypatch.setenv("SEND_TIMEZONE", "America/New_York")
    assert sdb._now().tzinfo == ET
