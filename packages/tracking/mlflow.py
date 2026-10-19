"""MLflow tracking helpers.

MLflow is an optional, local-only dependency: if it is not installed or the
tracking URI is unreachable, every helper degrades to a no-op so the API and
pipelines keep working. The joblib + provenance sidecar remains the load-path
code of record; MLflow is tracking/audit, never the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def tracking_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _tracking_uri() -> str:
    try:
        from app.core.config import settings

        return settings.mlflow_tracking_uri
    except Exception:
        return "file:./mlruns"


def log_champion() -> str | None:
    """Log the registered champion as an MLflow run; returns run id or None."""
    if not tracking_available():
        return None
    import mlflow

    meta_path = REPO_ROOT / "models" / "champion.meta.json"
    joblib_path = REPO_ROOT / "models" / "champion.joblib"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    metrics = meta.get("metrics", {}) or {}
    assumptions = meta.get("assumptions", {}) or {}

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("smartspend")
    with mlflow.start_run(run_name="champion-response-seed42") as run:
        mlflow.log_params(
            {
                "learner": meta.get("learner"),
                "target": meta.get("target"),
                "seed": meta.get("random_state"),
                "k": meta.get("k"),
                "git_commit": meta.get("git_commit"),
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        for k, v in assumptions.items():
            mlflow.log_param(f"assumption_{k}", v)
        if joblib_path.exists():
            mlflow.log_artifact(str(joblib_path), artifact_path="model")
        mlflow.log_artifact(str(meta_path), artifact_path="model")
        mlflow.set_tag("role", "champion")
        return run.info.run_id


def log_decision_run(
    tool: str,
    params: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> str | None:
    """Log one deterministic decision-tool invocation as an MLflow run."""
    if not tracking_available():
        return None
    import mlflow

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("smartspend")
    with mlflow.start_run(run_name=f"decision-{tool}") as run:
        mlflow.log_params(
            {
                k: (v if not isinstance(v, (dict, list)) else json.dumps(v))
                for k, v in params.items()
            }
        )
        if metrics:
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
            )
        mlflow.set_tag("kind", "decision")
        mlflow.set_tag("tool", tool)
        return run.info.run_id