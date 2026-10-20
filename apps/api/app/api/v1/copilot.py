"""Copilot endpoint — runs the grounded pipeline and persists the brief."""

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from packages.copilot import run_request
from packages.copilot.providers import ProviderError

from ...core.config import settings
from ...db.session import get_db
from ...models.audit_log import AuditLog
from ...models.brief import Brief
from ...models.scenario import Scenario

router = APIRouter()

DbDep = Annotated[Session, Depends(get_db)]

WriterProvider = Literal["template", "ollama", "neurometric"]


class CopilotRequest(BaseModel):
    question: str
    #: Writer override for the ablation toggle in the UI; None uses configs/copilot.yaml.
    provider: WriterProvider | None = None
    #: Pin a ranking model ("use this model" in the Model Lab). None lets the
    #: gate choose for plan/what-if; an invalid pin is a 422, never a swap.
    model: str | None = None
    #: Research Mode widens the allow-list to S/T/X/forest (locked otherwise).
    research_mode: bool = False


@router.post("")
def answer(payload: CopilotRequest, db: DbDep) -> dict:
    try:
        pkg = run_request(
            payload.question,
            writer_provider=payload.provider,
            model=payload.model,
            research_mode=payload.research_mode,
            mlflow_trace=settings.mlflow_tracing,
        )
    except ProviderError as err:
        raise HTTPException(
            status_code=503,
            detail=f"writer provider unavailable: {err}",
        ) from err
    except ValueError as err:
        # A pinned model that is unknown, unregistered, or research-only
        # without Research Mode — the message names exactly why.
        raise HTTPException(status_code=422, detail=str(err)) from err

    # Persist scenario + brief (best-effort; never blocks the response).
    scenario = Scenario(
        question=payload.question,
        intent=str(getattr(pkg, "intent", "plan")),
        model_name=pkg.model_name,
        model_selector=pkg.model_selector,
        score_kind=pkg.score_kind,
        research_mode=payload.research_mode,
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
            payload=json.dumps(
                {
                    "question": payload.question,
                    "trace_id": pkg.trace_id,
                    "model": pkg.model_name,
                    "model_selector": pkg.model_selector,
                    "research_mode": payload.research_mode,
                }
            ),
        )
    )
    db.commit()

    return pkg.model_dump()
