"""Lightweight tracking helpers (MLflow) for SmartSpend runs and GenAI traces."""

from packages.tracking.mlflow import (  # noqa: F401
    annotate_trace,
    log_champion,
    request_trace,
    set_span_outputs,
    trace_span,
    tracing_enabled,
    tracking_available,
)
