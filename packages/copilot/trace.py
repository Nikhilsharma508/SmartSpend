"""The visibility layer — one JSONL row per stage, per request.

Every request gets a short ``trace_id``; every stage (gate, plan, write,
validate) appends a :class:`~packages.copilot.schemas.TraceEvent` to
``traces/<YYYY-MM-DD>.jsonl`` (git-ignored — request traces are local data,
and Day 6's eval runner reads this same format instead of inventing a second
telemetry system).

A disabled logger (:class:`NullTrace`) is a no-op, so tests and library
callers never have to think about the filesystem.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.copilot.schemas import TraceEvent

#: Repo-root directory the traces land in (git-ignored).
TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"


def new_trace_id() -> str:
    """A short random id — enough to correlate rows, not a security token."""
    return uuid.uuid4().hex[:12]


class TraceLogger:
    """Appends events to ``traces/<date>.jsonl``; one file per UTC day."""

    def __init__(self, trace_id: str, directory: Path | None = None) -> None:
        self.trace_id = trace_id
        self.directory = Path(directory) if directory is not None else TRACES_DIR

    def log(self, stage: str, event: str, **data: Any) -> TraceEvent:
        entry = TraceEvent(
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            trace_id=self.trace_id,
            stage=stage,
            event=event,
            data=data,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{datetime.now(UTC).date().isoformat()}.jsonl"
        with path.open("a") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry


class NullTrace:
    """No-op logger for tests, library use and disabled configs."""

    trace_id: str = ""

    def log(self, stage: str, event: str, **data: Any) -> TraceEvent:
        return TraceEvent(ts="", trace_id=self.trace_id, stage=stage, event=event, data=data)


def make_logger(
    trace_id: str, enabled: bool, directory: str | None = None
) -> TraceLogger | NullTrace:
    """A real logger when enabled, the no-op otherwise (one call site)."""
    if not enabled:
        return NullTrace()
    return TraceLogger(trace_id, Path(directory) if directory else None)
