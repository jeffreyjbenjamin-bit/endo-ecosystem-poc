from __future__ import annotations

import argparse
import datetime
import json
import platform

from config import (
    OPENAI_API_KEY,
    CLINICAL_TRIALS_API_BASE,
    LOG_LEVEL,
)
from ingest import ping, ctgov_search_endometriosis


def print_banner() -> None:
    print("=" * 60)
    print("Hello from the Endo Ecosystem PoC!")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Timestamp: {datetime.datetime.now()}")
    print(f"Env: LOG_LEVEL={LOG_LEVEL}")
    print(f"Env: CLINICAL_TRIALS_API_BASE={CLINICAL_TRIALS_API_BASE}")
    # Never print secrets; only indicate presence
    print(f"Env: OPENAI_API_KEY set? {'yes' if OPENAI_API_KEY else 'no'}")
    print("=" * 60)


def main(limit: int = 5, json_out: bool = False) -> None:
    """
    Entry point for PoC CLI.

    Args:
        limit: Max number of trials to fetch/display.
        json_out: If True, print trials as compact JSON.
    """
    print_banner()

    if CLINICAL_TRIALS_API_BASE:
        print("Checking connectivity to ClinicalTrials API base...")
        ping_result = ping(CLINICAL_TRIALS_API_BASE)
        print(f"Ping: {ping_result}")

        print(f"Fetching sample endometriosis trials (up to {limit})...")
        trials = ctgov_search_endometriosis(CLINICAL_TRIALS_API_BASE, limit=limit)

        if json_out:
            print(json.dumps(trials, ensure_ascii=False))
        else:
            if not trials:
                print("No trials returned.")
            else:
                for i, t in enumerate(trials, start=1):
                    tid = t.get("id", "N/A")
                    title = t.get("title", "N/A")
                    status = t.get("status", "N/A")
                    print(f"{i:02d}. {tid} | {status} | {title}")
    else:
        print("No API base configured; skipping ping/search.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Endo Ecosystem PoC runner: prints env info and fetches a few Endometriosis trials."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of trials to fetch (default: 5)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Print trials as JSON instead of pretty text",
    )
    args = parser.parse_args()
    main(limit=args.limit, json_out=args.json_out)
