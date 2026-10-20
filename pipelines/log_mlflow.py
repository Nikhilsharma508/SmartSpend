"""Backfill MLflow with every registered catalog artifact (run once).

Usage: uv run python pipelines/log_mlflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.tracking.mlflow import (  # noqa: E402
    log_registered_models,
    tracking_available,
)


def main() -> int:
    if not tracking_available():
        print("mlflow not installed; skipping")
        return 0
    runs = log_registered_models()
    print(f"logged {len(runs)} registered model run(s):")
    for run_id in runs:
        print(" -", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
