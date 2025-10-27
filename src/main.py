from __future__ import annotations
import datetime
import platform
from config import OPENAI_API_KEY, CLINICAL_TRIALS_API_BASE, LOG_LEVEL


def main():
    print("=" * 60)
    print("Hello from the Endo Ecosystem PoC!")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Timestamp: {datetime.datetime.now()}")
    # Do NOT print secrets; show whether they exist:
    print(f"Env: LOG_LEVEL={LOG_LEVEL}")
    print(f"Env: CLINICAL_TRIALS_API_BASE={CLINICAL_TRIALS_API_BASE}")
    print(f"Env: OPENAI_API_KEY set? {'yes' if OPENAI_API_KEY else 'no'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
