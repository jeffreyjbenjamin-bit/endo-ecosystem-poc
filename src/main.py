"""
Endo Ecosystem PoC – Starter Script

This simple placeholder validates that the environment and CI pipeline
can execute Python code correctly.
"""

import datetime
import platform


def main():
    print("=" * 60)
    print("Hello from the Endo Ecosystem PoC!")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Timestamp: {datetime.datetime.now()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
