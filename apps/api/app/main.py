"""FastAPI application entrypoint."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .api.v1.router import api_router
from .core.config import settings

FIXTURES_PATH = Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "api" / "mock" / "fixtures.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from packages.tracking.mlflow import log_champion

        log_champion()
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/api/fixtures")
def fixtures() -> dict:
    return json.loads(FIXTURES_PATH.read_text())


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}