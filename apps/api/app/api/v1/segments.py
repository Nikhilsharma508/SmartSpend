"""Segment and model-card endpoints (run on the registered champion)."""

from fastapi import APIRouter

from packages.copilot.planner import get_context

router = APIRouter()


@router.get("/profile")
def profile() -> dict:
    ctx = get_context()
    return ctx.segment_profile().model_dump()


@router.get("/model-card")
def model_card() -> dict:
    ctx = get_context()
    return ctx.model_card().model_dump()