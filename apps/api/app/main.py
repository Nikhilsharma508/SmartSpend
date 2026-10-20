"""FastAPI application entrypoint."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.v1.router import api_router
from .core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The policy-tree fit is ~20 s; do it off the event loop so /health answers
    # immediately and the first page view does not pay for it.
    from .api.v1.overview import warm_overview

    threading.Thread(target=warm_overview, daemon=True, name="overview-warmup").start()

    # Best-effort: register the champion in MLflow on boot (no-op once logged).
    try:
        from packages.tracking.mlflow import log_champion

        log_champion()
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
