"""Trace identifiers for the copilot pipeline.

Every request gets a short id; the MLflow GenAI trace stores it as
``client_request_id`` and a ``smartspend.trace_id`` tag so the API response,
the audit log and the tracing UI can be correlated.
"""

from __future__ import annotations

import uuid


def new_trace_id() -> str:
    """A short random id — enough to correlate requests, not a security token."""
    return uuid.uuid4().hex[:12]
