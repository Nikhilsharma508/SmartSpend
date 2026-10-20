"""Aggregated v1 router."""

from fastapi import APIRouter

from . import ab_tests, copilot, overview, scenarios, segments

api_router = APIRouter()
api_router.include_router(overview.router, prefix="/overview", tags=["overview"])
api_router.include_router(scenarios.router, prefix="/scenarios", tags=["scenarios"])
api_router.include_router(segments.router, prefix="/segments", tags=["segments"])
api_router.include_router(ab_tests.router, prefix="/ab-tests", tags=["ab-tests"])
api_router.include_router(copilot.router, prefix="/copilot", tags=["copilot"])