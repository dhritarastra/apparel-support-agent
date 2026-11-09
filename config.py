"""Paths, the business clock and the model name.

The clock is a fixed demo date (not the wall clock) so the 30-day-window cases in
data/ stay in/out of window no matter when the code is run or graded.
Override with AS_OF_DATE=YYYY-MM-DD.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_STATE_PATH = ROOT / "state" / "state.json"

DEFAULT_AS_OF = date(2026, 11, 9)
DEFAULT_MODEL = "claude-sonnet-5-5"


def get_as_of() -> date:
    raw = os.environ.get("AS_OF_DATE", "").strip()
    if not raw:
        return DEFAULT_AS_OF
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"AS_OF_DATE must be YYYY-MM-DD, got {raw!r}") from None


def get_model() -> str:
    return os.environ.get("MODEL", "").strip() or DEFAULT_MODEL


def get_state_path() -> Path:
    """state/state.json, or STATE_FILE if set (evals point this at a temp file)."""
    raw = os.environ.get("STATE_FILE", "").strip()
    return Path(raw) if raw else DEFAULT_STATE_PATH
