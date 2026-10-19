"""Copilot endpoint — runs the grounded pipeline and persists the brief."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from packages.copilot import run_request

from ...db.session import get_db
from ...models.audit_log import AuditLog
from ...models.brief import Brief
from ...models.scenario import Scenario

router = APIRouter()

DbDep = Annotated[Session, Depends(get_db)]


class CopilotRequest(BaseModel):
    question: str


@router.post("")
def answer(payload: CopilotRequest, db: DbDep) -> dict:
    pkg = run_request(payload.question)

    # Persist scenario + brief (best-effort; never blocks the response).
    scenario = Scenario(
        question=payload.question,
        intent=str(getattr(pkg, "intent", "plan")),
    )
    db.add(scenario)
    db.flush()

    validation = getattr(pkg, "validation", None)
    validation_json = validation.model_dump_json() if validation is not None else None
    citations = getattr(pkg, "citations", None)
    citations_json = json.dumps(
        [c.model_dump() if hasattr(c, "model_dump") else c for c in citations]
    ) if citations is not None else None

    brief = Brief(
        scenario_id=scenario.id,
        prose=pkg.prose,
        validation_json=validation_json,
        citations_json=citations_json,
    )
    db.add(brief)

    db.add(
        AuditLog(
            kind="copilot",
            name="run_request",
            payload=json.dumps({"question": payload.question, "trace_id": pkg.trace_id}),
        )
    )
    db.commit()

    # Best-effort MLflow tracking of the deterministic decision run (never
    # blocks or fails the response).
    try:
        from packages.tracking.mlflow import log_decision_run

        log_decision_run(
            "copilot",
            {"question": payload.question, "trace_id": pkg.trace_id, "fast_path": pkg.fast_path},
            {"draft_attempts": getattr(pkg, "draft_attempts", 1)},
        )
    except Exception:  # noqa: BLE001
        pass

    return pkg.model_dump()