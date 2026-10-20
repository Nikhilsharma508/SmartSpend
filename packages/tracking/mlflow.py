"""MLflow tracking helpers.

MLflow is an optional, local-only dependency: if it is not installed or the
tracking URI is unreachable, every helper degrades to a no-op so the API and
pipelines keep working. Tracking is metadata-and-trace only: runs, params,
metrics, tags and GenAI spans live in ``data/mlflow/mlflow.db``. No artifacts are logged;
the joblib + provenance sidecar remains the load-path code of record.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Silence MLflow 3's CLI agent hint before the package can print it.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: True only inside a :func:`request_trace` block, so stage spans are no-ops
#: for tests, evals and library callers that did not ask for tracing.
_TRACE_ACTIVE: ContextVar[bool] = ContextVar("smartspend_mlflow_trace", default=False)


def tracking_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _tracking_uri() -> str:
    """Same env var pydantic-settings reads; no import of the API package."""
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///data/mlflow/mlflow.db")


def _store_dir() -> Path:
    uri = _tracking_uri()
    if uri.startswith("file:"):
        path = Path(uri[len("file:") :])
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / "data" / "mlflow" / "mlruns"


def _mlflow_uri() -> str:
    """Resolve relative stores against the repo root, not the CWD."""
    uri = _tracking_uri()
    if uri.startswith("file:"):
        return f"file:{_store_dir()}"
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel = uri[len("sqlite:///") :]
        return f"sqlite:///{(REPO_ROOT / rel).resolve()}"
    return uri


def _init_tracking() -> bool:
    """Point MLflow at the local store and select the SmartSpend experiment."""
    if not tracking_available():
        return False
    import mlflow

    try:
        uri = _mlflow_uri()
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        elif uri.startswith("file:"):
            Path(uri.removeprefix("file:")).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("smartspend")
        return True
    except Exception:  # noqa: BLE001 - tracking must never break a caller
        return False


def tracing_enabled() -> bool:
    """MLFLOW_TRACING=0 turns GenAI traces off (tests, evals, offline runs)."""
    return os.environ.get("MLFLOW_TRACING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _set_inputs(span: Any, inputs: Any) -> None:
    if span is None or inputs is None:
        return
    with suppress(Exception):
        span.set_inputs(inputs)


def _set_outputs(span: Any, outputs: Any) -> None:
    if span is None or outputs is None:
        return
    with suppress(Exception):
        span.set_outputs(outputs)


def _set_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
    if span is None or not attributes:
        return
    with suppress(Exception):
        span.set_attributes(
            {
                key: value if isinstance(value, (str, int, float, bool)) else str(value)
                for key, value in attributes.items()
            }
        )


@contextmanager
def request_trace(
    question: str,
    provider: str | None = None,
) -> Iterator[Any | None]:
    """Root GenAI trace for one copilot request; yields the live span or None."""
    if not tracing_enabled() or not _init_tracking():
        yield None
        return

    import mlflow

    try:
        span_cm = mlflow.start_span(name="copilot", span_type="CHAIN")
        span = span_cm.__enter__()
    except Exception:  # noqa: BLE001 - tracing must never break a request
        yield None
        return

    _set_inputs(span, {"question": question, "provider": provider or "config"})
    token = _TRACE_ACTIVE.set(True)
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield span
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        _TRACE_ACTIVE.reset(token)
        with suppress(Exception):
            span_cm.__exit__(*exc_info)


@contextmanager
def trace_span(
    name: str,
    span_type: str = "UNKNOWN",
    *,
    inputs: Any = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Child span inside :func:`request_trace`; a no-op elsewhere."""
    if not _TRACE_ACTIVE.get():
        yield None
        return

    import mlflow

    try:
        span_cm = mlflow.start_span(name=name, span_type=span_type)
        span = span_cm.__enter__()
    except Exception:  # noqa: BLE001
        yield None
        return

    _set_inputs(span, inputs)
    _set_attributes(span, attributes)
    try:
        yield span
    finally:
        with suppress(Exception):
            span_cm.__exit__(None, None, None)


def set_span_outputs(span: Any | None, outputs: Any) -> None:
    """Public helper so callers do not import MLflow themselves."""
    _set_outputs(span, outputs)


def annotate_trace(
    *,
    trace_id: str,
    request_preview: str,
    response_preview: str,
    tags: dict[str, str] | None = None,
) -> None:
    """Attach the app trace id and readable previews to the active GenAI trace."""
    if not _TRACE_ACTIVE.get():
        return
    try:
        import mlflow

        mlflow.update_current_trace(
            client_request_id=trace_id,
            request_preview=request_preview[:1000],
            response_preview=response_preview[:1000],
            tags={"smartspend.trace_id": trace_id, **(tags or {})},
        )
    except Exception:  # noqa: BLE001 - tracing is best-effort
        pass


def log_registered_model(name: str = "champion") -> str | None:
    """Log one registered artifact once per metadata digest; None when absent/off.

    Every registered model gets a run so the MLflow compare view can line the
    six learners up against each other; tags carry the catalog semantics
    (``tier``, ``score_kind``, ``research.mode``) so the UI can filter. No
    artifacts are ever logged — the sidecar remains the provenance record.
    """
    meta_path = REPO_ROOT / "models" / f"{name}.meta.json"
    if not meta_path.exists():
        return None

    digest = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    if not _init_tracking():
        return None
    import mlflow

    meta = json.loads(meta_path.read_text())
    metrics = meta.get("metrics", {}) or {}
    assumptions = meta.get("assumptions", {}) or {}

    try:
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name("smartspend")
        if experiment is not None:
            existing = client.search_runs(
                [experiment.experiment_id],
                filter_string=(f"tags.meta_digest = '{digest}' and tags.model_name = '{name}'"),
                max_results=1,
            )
            if existing:
                return existing[0].info.run_id

        learner = str(meta.get("learner", name))
        tier = str(meta.get("tier", "research"))
        run_name = f"{name}-{learner}-seed{meta.get('random_state', '?')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "learner": meta.get("learner"),
                    "model_name": name,
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
            mlflow.set_tag("role", tier)
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("tier", tier)
            mlflow.set_tag("score_kind", str(meta.get("score_kind", "unknown")))
            mlflow.set_tag("research.mode", (
                "on" if str(tier) == "research" else "off"
            ))
            mlflow.set_tag("meta_digest", digest)
            return run.info.run_id
    except Exception:  # noqa: BLE001 - tracking must never break a boot
        return None


def log_registered_models() -> list[str]:
    """Log every registered catalog artifact; returns the run ids that were logged."""
    from packages.causal.catalog import MODEL_SPECS
    from packages.causal.registry import is_registered

    logged: list[str] = []
    for spec in MODEL_SPECS:
        if not is_registered(spec.name):
            continue
        run_id = log_registered_model(spec.name)
        if run_id:
            logged.append(run_id)
    return logged


def log_champion() -> str | None:
    """Backwards-compatible alias: log the champion artifact only."""
    return log_registered_model("champion")
