"""Segment and model-card endpoints — run on the requested registered model.

The segment tree is shared across models (same split, same policy fit), while
the model card is per artifact; both accept the ``model`` + ``research_mode``
query pair so the pages stay in sync with the model the user selected. The
validation helper lives in :mod:`.overview` — the same 404/422 contract.
"""

from fastapi import APIRouter, Query

from packages.copilot.planner import get_context

from .overview import _validate_model_param

router = APIRouter()


@router.get("/profile")
def profile(
    model: str = Query(default="champion"),
    research_mode: bool = Query(default=False),
) -> dict:
    _validate_model_param(model, research_mode)
    ctx = get_context(model)
    return ctx.segment_profile().model_dump()


@router.get("/model-card")
def model_card(
    model: str = Query(default="champion"),
    research_mode: bool = Query(default=False),
) -> dict:
    _validate_model_param(model, research_mode)
    ctx = get_context(model)
    return ctx.model_card().model_dump()
