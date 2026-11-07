"""
scheduler.py — runs the daily Intelligence Briefing / Newsletter send on a schedule.

Started next to the Streamlit app by deploy/start.sh in the Docker image. Every minute
it checks whether the daily send is due (time set in the Report Config tab, read in
SEND_TIMEZONE) and, if so, runs `send_daily_briefing.py --auto-search` in a child
process. send_daily_briefing.py records each day's send, so it never sends twice.

Output goes to stdout (Azure log stream) and <state dir>/daily_send.log.
Set DAILY_SEND_ENABLED=0 to switch it off.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from send_daily_briefing import send_status  # noqa: E402
from src.common.paths import STATE_DIR  # noqa: E402

CHECK_EVERY_SECONDS = 60
LOG_PATH = STATE_DIR / "daily_send.log"


def log(msg: str) -> None:
    line = f"[scheduler {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run_send() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "send_daily_briefing.py"), "--auto-search"]
    log("Starting daily send: " + " ".join(cmd[1:]))
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45 * 60,
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        log("  " + line)
    log(f"Daily send finished (exit code {proc.returncode}).")


def main() -> None:
    if os.getenv("DAILY_SEND_ENABLED", "1").strip().lower() in ("0", "false", "no"):
        log("DAILY_SEND_ENABLED is off; scheduler not running.")
        return
    tz = os.getenv("SEND_TIMEZONE", "") or "system local time"
    log(f"Scheduler started (time zone: {tz}). {send_status()[1]}")
    while True:
        try:
            due, _reason = send_status()
            if due:
                run_send()
        except Exception as exc:  # keep the loop alive no matter what
            log(f"ERROR: {exc!r}")
        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()
