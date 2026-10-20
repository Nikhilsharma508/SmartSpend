"""MLflow tracking helpers.

MLflow is an optional, local-only dependency: if it is not installed or the
tracking URI is unreachable, every helper degrades to a no-op so the API and
pipelines keep working. The joblib + provenance sidecar remains the load-path
code of record; MLflow is tracking/audit, never the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Written next to the tracking store after the champion is logged, so a
#: uvicorn `--reload` (or any restart) never spams duplicate champion runs.
_CHAMPION_STAMP = ".champion_logged.json"


def tracking_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _tracking_uri() -> str:
    """Same env var pydantic-settings reads; no import of the API package."""
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


def _store_dir() -> Path:
    uri = _tracking_uri()
    if uri.startswith("file:"):
        path = Path(uri[len("file:") :])
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / "mlruns"


def _mlflow_uri() -> str:
    """Resolve a relative file store against the repo root, not the CWD."""
    uri = _tracking_uri()
    return f"file:{_store_dir()}" if uri.startswith("file:") else uri


def log_champion() -> str | None:
    """Log the registered champion once per artifact; returns run id or None."""
    meta_path = REPO_ROOT / "models" / "champion.meta.json"
    joblib_path = REPO_ROOT / "models" / "champion.joblib"
    if not meta_path.exists():
        return None

    digest = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    stamp = _store_dir() / _CHAMPION_STAMP
    if stamp.exists():
        try:
            logged = json.loads(stamp.read_text())
            if logged.get("digest") == digest:
                return logged.get("run_id")
        except (OSError, json.JSONDecodeError):
            pass

    if not tracking_available():
        return None
    import mlflow

    meta = json.loads(meta_path.read_text())
    metrics = meta.get("metrics", {}) or {}
    assumptions = meta.get("assumptions", {}) or {}

    try:
        mlflow.set_tracking_uri(_mlflow_uri())
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
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            for k, v in assumptions.items():
                mlflow.log_param(f"assumption_{k}", v)
            if joblib_path.exists():
                mlflow.log_artifact(str(joblib_path), artifact_path="model")
            mlflow.log_artifact(str(meta_path), artifact_path="model")
            mlflow.set_tag("role", "champion")
            run_id = run.info.run_id
    except Exception:  # noqa: BLE001 - tracking must never break a boot
        return None

    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps({"digest": digest, "run_id": run_id}) + "\n")
    except OSError:
        pass
    return run_id


def log_decision_run(
    tool: str,
    params: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> str | None:
    """Log one deterministic decision-tool invocation as an MLflow run."""
    if not tracking_available():
        return None
    import mlflow

    try:
        mlflow.set_tracking_uri(_mlflow_uri())
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
    except Exception:  # noqa: BLE001 - tracking must never break a request
        return None
