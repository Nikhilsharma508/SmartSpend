"""A/B test design endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from packages.decision.power import design_ab_test

router = APIRouter()


class AbTestRequest(BaseModel):
    baseline_rate: float
    mde: float
    alpha: float = 0.05
    power: float = 0.8
    daily_eligible: int = 1000


@router.post("/design")
def design(payload: AbTestRequest) -> dict:
    result = design_ab_test(
        payload.baseline_rate,
        payload.mde,
        alpha=payload.alpha,
        power=payload.power,
        daily_eligible=payload.daily_eligible,
    )
    return result.model_dump()