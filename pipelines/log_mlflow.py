"""Backfill MLflow with the registered champion (run once, idempotent-ish).

Usage: uv run python pipelines/log_mlflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.tracking.mlflow import log_champion, tracking_available  # noqa: E402


def main() -> int:
    if not tracking_available():
        print("mlflow not installed; skipping")
        return 0
    run_id = log_champion()
    print("logged champion run:", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())