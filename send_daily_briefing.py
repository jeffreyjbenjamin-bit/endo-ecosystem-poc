"""
send_daily_briefing.py — CLI script for automated daily briefing and newsletter dispatch.

On Azure this runs automatically: scheduler.py (started by deploy/start.sh in the
Docker image) checks every minute and runs this script with --auto-search when due.
Elsewhere, run it from a system scheduler (Windows Task Scheduler or cron).

A scheduled run sends once per day, from the time configured in Settings (⚙️)
until 2 hours after it. The time is read in SEND_TIMEZONE (e.g. America/New_York),
or the system's local time zone if that isn't set.

Usage:
    python send_daily_briefing.py              # sends if due and not yet sent today
    python send_daily_briefing.py --force      # send immediately, ignore time check
    python send_daily_briefing.py --auto-search  # run AI search first, then send

Scheduling examples
-------------------
Windows Task Scheduler:
    Action: python C:\\path\\to\\repo\\send_daily_briefing.py --auto-search
    Trigger: Daily, ~30 min before your configured send time

macOS / Linux cron (runs every 5 minutes, script self-gates on time):
    */5 * * * * /path/to/venv/bin/python /path/to/repo/send_daily_briefing.py --auto-search

Two recipients types are supported:
  - subscription_type='briefing' or 'both' → receives the daily AI Search briefing
  - subscription_type='newsletter' or 'both' → receives any queued Newsletter
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.email_briefing import send_briefing, smtp_configured  # noqa: E402
from src.common.paths import STATE_DIR  # noqa: E402

DATA_DIR = STATE_DIR
MAILING_LIST_DB = str(DATA_DIR / "mailing_list.sqlite")
AI_SEARCH_DB = str(DATA_DIR / "ai_search_runs.sqlite")
NEWSLETTERS_DB = str(DATA_DIR / "newsletters.sqlite")

# A scheduled (non --force) run sends once per day, starting at the configured time
# and up to this many minutes after it (covers restarts and slow schedulers).
_CATCH_UP_MINUTES = 120
_LAST_SEND_KEY = "last_daily_send_date"


def _send_tz():
    """Time zone the configured send time is in (SEND_TIMEZONE, else system local)."""
    name = os.getenv("SEND_TIMEZONE", "").strip()
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        print(f"WARNING: unknown SEND_TIMEZONE {name!r}; using system local time.")
        return None


def _now() -> datetime:
    tz = _send_tz()
    return datetime.now(tz) if tz else datetime.now()


# ── Config helpers ─────────────────────────────────────────────────────────────


def _get_config(key: str, default: str = "") -> str:
    if not Path(MAILING_LIST_DB).exists():
        return default
    conn = sqlite3.connect(MAILING_LIST_DB)
    try:
        cur = conn.execute("SELECT value FROM config WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default
    finally:
        conn.close()


def _set_config(key: str, value: str) -> None:
    conn = sqlite3.connect(MAILING_LIST_DB)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT INTO config (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


# ── Recipient helpers ──────────────────────────────────────────────────────────


def _load_recipients(subscription_filter: tuple = ("briefing", "both")):
    if not Path(MAILING_LIST_DB).exists():
        return []
    conn = sqlite3.connect(MAILING_LIST_DB)
    try:
        placeholders = ",".join("?" for _ in subscription_filter)
        cur = conn.execute(
            f"SELECT id, name, email, subscription_type FROM recipients "
            f"WHERE subscription_type IN ({placeholders}) ORDER BY name COLLATE NOCASE",
            subscription_filter,
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _mark_sent(recipient_id: int) -> None:
    conn = sqlite3.connect(MAILING_LIST_DB)
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE recipients SET last_sent_ts=? WHERE id=?", (ts, recipient_id))
    conn.commit()
    conn.close()


# ── AI Search run helpers ──────────────────────────────────────────────────────


def _load_latest_ai_runs(limit: int = 2):
    if not Path(AI_SEARCH_DB).exists():
        return []
    conn = sqlite3.connect(AI_SEARCH_DB)
    cur = conn.execute(
        "SELECT id, run_ts, topic, briefing_text, sources_json "
        "FROM ai_search_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def _compute_new_items(current_run: dict, previous_run: dict | None) -> list:
    if not previous_run:
        return []
    try:
        cur_sources = json.loads(current_run.get("sources_json") or "[]")
        prev_sources = json.loads(previous_run.get("sources_json") or "[]")
        prev_urls = {
            s.get("url", "").lower().rstrip("/") for s in prev_sources if s.get("url")
        }
        return [
            {"title": s.get("title", "Untitled"), "url": s.get("url", "")}
            for s in cur_sources
            if s.get("url") and s["url"].lower().rstrip("/") not in prev_urls
        ]
    except Exception:
        return []


# ── Newsletter helpers ─────────────────────────────────────────────────────────


def _load_newsletter(newsletter_id: int) -> dict | None:
    if not Path(NEWSLETTERS_DB).exists():
        return None
    conn = sqlite3.connect(NEWSLETTERS_DB)
    try:
        cur = conn.execute(
            "SELECT id, title, content_md FROM newsletters WHERE id=?",
            (newsletter_id,),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None
    except Exception:
        return None
    finally:
        conn.close()


# ── Timing helper ──────────────────────────────────────────────────────────────


def _is_send_time(configured_time: str, catch_up_minutes: int) -> bool:
    """True from the configured time until `catch_up_minutes` after it (send time zone)."""
    try:
        cfg_h, cfg_m = [int(x) for x in configured_time.split(":")]
    except Exception:
        return False
    now = _now()
    minutes_past = (now.hour * 60 + now.minute) - (cfg_h * 60 + cfg_m)
    return 0 <= minutes_past <= catch_up_minutes


def send_status() -> tuple[bool, str]:
    """(due, reason) for a scheduled send right now. Used by this script and scheduler.py."""
    configured_time = _get_config("daily_send_time", "07:00")
    today = _now().strftime("%Y-%m-%d")
    tz_label = os.getenv("SEND_TIMEZONE", "") or "local"
    if _get_config(_LAST_SEND_KEY, "") == today:
        return False, f"Already sent today ({today})."
    if not _is_send_time(configured_time, _CATCH_UP_MINUTES):
        return False, (
            f"Not send time yet (configured: {configured_time} {tz_label}, "
            f"now: {_now().strftime('%H:%M')})."
        )
    return True, f"Due: configured {configured_time} {tz_label}."


# ── Auto AI search ─────────────────────────────────────────────────────────────


def _refresh_trials() -> None:
    """Refresh the Clinical Trials list (failures never stop the send)."""
    try:
        from src.pipelines.clinical_trials import refresh_trials, unbriefed_trials

        data = refresh_trials()
        print(
            f"  Clinical trials refreshed: {len(data['trials'])} open, "
            f"{len(unbriefed_trials(data))} new for the briefing."
        )
    except Exception as exc:
        print(f"  Clinical trials refresh failed: {exc}")


def _with_new_trials(briefing_text: str) -> str:
    """Add the New Clinical Trials section for trials not yet emailed."""
    try:
        from src.pipelines.clinical_trials import (
            add_trials_to_briefing,
            unbriefed_trials,
        )

        return add_trials_to_briefing(briefing_text, unbriefed_trials())
    except Exception as exc:
        print(f"  Could not add new clinical trials: {exc}")
        return briefing_text


def _mark_trials_briefed(briefing_text: str) -> None:
    try:
        from src.pipelines.clinical_trials import mark_briefed

        done = mark_briefed(briefing_text)
        if done:
            print(f"  New clinical trials included: {', '.join(done)}")
    except Exception as exc:
        print(f"  Could not record emailed clinical trials: {exc}")


def _prepare_home_page() -> None:
    """Refresh the Home page's AI trend analysis and article preview images."""
    try:
        from src.pipelines.trends import generate_trends

        data = generate_trends()
        print(f"  Trend analysis updated ({data['counts']}).")
    except Exception as exc:
        print(f"  Trend analysis failed: {exc}")
    try:
        from src.common.page_images import images_for

        runs = _load_latest_ai_runs(limit=1)
        sources = json.loads(runs[0].get("sources_json") or "[]") if runs else []
        found = images_for(s.get("url", "") for s in sources if not s.get("image"))
        print(
            f"  Preview images: {sum(1 for v in found.values() if v)} of {len(found)}."
        )
    except Exception as exc:
        print(f"  Preview images failed: {exc}")


def _run_ai_search() -> bool:
    """Run the AI Article Search pipeline and save the result. Returns True on success."""
    print("Running AI Search pipeline…")
    try:
        from src.pipelines.ai_search import (
            generate_search_queries,
            run_search_provider,
            normalize_results,
            deduplicate_results,
            score_results,
            synthesize_briefing,
            save_ai_search_run,
            CHAT_DEPLOYMENT_5,
        )
        from src.connectors.search_provider import get_provider

        topic = "endometriosis"
        date_range_days = 90
        categories: list = []

        _refresh_trials()

        print("  Generating search queries…")
        queries = generate_search_queries(topic, date_range_days, categories)

        print(f"  Running {len(queries)} queries…")
        provider = get_provider()
        raw, errs = run_search_provider(
            queries, provider, num_per_query=10, date_range_days=date_range_days
        )
        if errs:
            for e in errs:
                print(f"  Search warning: {e}")
        if not raw:
            print("  ERROR: No search results returned.")
            return False

        print(f"  Normalizing and scoring {len(raw)} results…")
        normed = normalize_results(raw)
        deduped = deduplicate_results(normed)
        scored = score_results(deduped, topic, date_range_days)

        print("  Synthesizing briefing via Azure GPT…")
        briefing_text, perf = synthesize_briefing(
            scored, topic, date_range_days, CHAT_DEPLOYMENT_5
        )
        briefing_text = _with_new_trials(briefing_text)

        save_ai_search_run(
            topic=topic,
            date_range_days=date_range_days,
            categories=categories,
            queries=queries,
            n_raw=len(raw),
            scored_results=scored,
            provider_name=provider.name,
            model_used=CHAT_DEPLOYMENT_5,
            latency_ms=perf.get("latency_ms", 0),
            briefing_text=briefing_text,
        )
        print(
            f"  AI Search complete ({len(scored)} scored articles, {perf.get('latency_ms',0):.0f}ms)."
        )
        return True
    except Exception as exc:
        print(f"  AI Search failed: {exc}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────


def main(force: bool = False, auto_search: bool = False) -> None:
    if not force:
        due, reason = send_status()
        if not due:
            print(f"{reason} Exiting.")
            return

    if not smtp_configured():
        print(
            "ERROR: SMTP is not configured. Set EMAIL_SMTP_HOST, EMAIL_FROM, EMAIL_PASSWORD in .env"
        )
        sys.exit(1)

    if not force:
        # Record today's send before sending, so a crash or partial failure
        # can never cause the same day's emails to go out twice.
        _set_config(_LAST_SEND_KEY, _now().strftime("%Y-%m-%d"))

    # Optionally run AI search to get fresh content
    if auto_search:
        _run_ai_search()
        _prepare_home_page()

    today = _now().strftime("%B %d, %Y")
    sent_utc = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")

    # ── Send AI Briefing to briefing subscribers ───────────────
    runs = _load_latest_ai_runs(limit=2)
    if runs:
        current_run = runs[0]
        previous_run = runs[1] if len(runs) > 1 else None
        briefing_text = (current_run.get("briefing_text") or "").strip()

        if briefing_text:
            briefing_recips = _load_recipients(("briefing", "both"))
            if briefing_recips:
                new_items = _compute_new_items(current_run, previous_run)
                subject = f"Evidence Gap — Intelligence Briefing · {today}"
                meta = f"Endometriosis Intelligence · Sent {sent_utc}"
                ok, fail = [], []
                print(f"\nSending AI Briefing to {len(briefing_recips)} subscriber(s)…")
                for recip in briefing_recips:
                    try:
                        send_briefing(
                            to_addr=recip["email"],
                            subject=subject,
                            title=subject,
                            meta=meta,
                            briefing_md=briefing_text,
                            new_items=new_items or None,
                        )
                        _mark_sent(recip["id"])
                        ok.append(f"{recip['name']} <{recip['email']}>")
                        print(f"  ✓ {recip['name']} <{recip['email']}>")
                    except Exception as exc:
                        fail.append(f"{recip['name']} — {exc}")
                        print(f"  ✗ {recip['name']} — {exc}")
                print(
                    f"  Briefing: {len(ok)} sent, {len(fail)} failed. {len(new_items)} new sources."
                )
                if ok:
                    _mark_trials_briefed(briefing_text)
            else:
                print("No recipients subscribed to AI Briefing.")
        else:
            print(
                "WARNING: Most recent AI Search run has no briefing text. Skipping briefing send."
            )
    else:
        print("WARNING: No AI Search runs found. Skipping briefing send.")

    # ── Send queued Newsletter to newsletter subscribers ────────
    pending_id_str = _get_config("pending_newsletter_id", "")
    if pending_id_str:
        try:
            pending_id = int(pending_id_str)
        except ValueError:
            pending_id = None

        if pending_id:
            nl = _load_newsletter(pending_id)
            if nl:
                nl_text = (nl.get("content_md") or "").strip()
                nl_title = nl.get("title") or "Endometriosis Newsletter"
                nl_recips = _load_recipients(("newsletter", "both"))
                if nl_recips and nl_text:
                    nl_subject = f"Evidence Gap — {nl_title}"
                    nl_meta = f"Endometriosis Newsletter · Sent {sent_utc}"
                    ok_nl, fail_nl = [], []
                    print(
                        f"\nSending Newsletter #{pending_id} to {len(nl_recips)} subscriber(s)…"
                    )
                    for recip in nl_recips:
                        try:
                            send_briefing(
                                to_addr=recip["email"],
                                subject=nl_subject,
                                title=nl_title,
                                meta=nl_meta,
                                briefing_md=nl_text,
                            )
                            _mark_sent(recip["id"])
                            ok_nl.append(f"{recip['name']} <{recip['email']}>")
                            print(f"  ✓ {recip['name']} <{recip['email']}>")
                        except Exception as exc:
                            fail_nl.append(f"{recip['name']} — {exc}")
                            print(f"  ✗ {recip['name']} — {exc}")
                    print(f"  Newsletter: {len(ok_nl)} sent, {len(fail_nl)} failed.")
                    # Clear the queue
                    _set_config("pending_newsletter_id", "")
                else:
                    print(
                        f"Newsletter #{pending_id}: no content or no subscribers. Skipping."
                    )
            else:
                print(f"Newsletter #{pending_id} not found in DB. Clearing queue.")
                _set_config("pending_newsletter_id", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send daily Intelligence Briefing and/or Newsletter"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send immediately regardless of configured send time",
    )
    parser.add_argument(
        "--auto-search",
        action="store_true",
        dest="auto_search",
        help="Run the AI Article Search pipeline before sending to get fresh content",
    )
    args = parser.parse_args()
    main(force=args.force, auto_search=args.auto_search)
