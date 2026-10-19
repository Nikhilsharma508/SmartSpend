"""Scenario persistence endpoints."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...models.scenario import Scenario

router = APIRouter()

DbDep = Annotated[Session, Depends(get_db)]


class ScenarioCreate(BaseModel):
    question: str
    intent: str = "plan"
    budget_usd: float | None = None
    cost_per_contact_usd: float | None = None
    depth: float | None = None
    assumptions_used: list[str] = []


class ScenarioRead(BaseModel):
    id: int
    question: str
    intent: str
    budget_usd: float | None = None
    cost_per_contact_usd: float | None = None
    depth: float | None = None
    assumptions_used: list[str] = []

    model_config = {"from_attributes": True}


def _to_read(s: Scenario) -> ScenarioRead:
    try:
        assumptions = json.loads(s.assumptions_used) if s.assumptions_used else []
    except json.JSONDecodeError:
        assumptions = []
    return ScenarioRead(
        id=s.id,
        question=s.question,
        intent=s.intent,
        budget_usd=s.budget_usd,
        cost_per_contact_usd=s.cost_per_contact_usd,
        depth=s.depth,
        assumptions_used=assumptions,
    )


@router.post("", response_model=ScenarioRead)
def create_scenario(payload: ScenarioCreate, db: DbDep) -> ScenarioRead:
    s = Scenario(
        question=payload.question,
        intent=payload.intent,
        budget_usd=payload.budget_usd,
        cost_per_contact_usd=payload.cost_per_contact_usd,
        depth=payload.depth,
        assumptions_used=json.dumps(payload.assumptions_used),
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return _to_read(s)


@router.get("/{scenario_id}", response_model=ScenarioRead)
def get_scenario(scenario_id: int, db: DbDep) -> ScenarioRead:
    s = db.get(Scenario, scenario_id)
    if s is None:
        raise HTTPException(status_code=404, detail="scenario not found")
    return _to_read(s)