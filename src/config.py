from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root if present.
# This is safe: if .env doesn't exist (e.g., in CI), nothing happens.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


def get_env(
    name: str, default: str | None = None, required: bool = False
) -> str | None:
    """Fetch an environment variable, optionally enforcing presence."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Required environment variable '{name}' is not set")
    return value


# Example typed accessors (extend as needed)
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
CLINICAL_TRIALS_API_BASE = get_env(
    "CLINICAL_TRIALS_API_BASE", "https://clinicaltrials.gov/api/v2"
)
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
