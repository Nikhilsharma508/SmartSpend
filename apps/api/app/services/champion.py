"""Cached loader for the registered champion model."""

import json
from functools import lru_cache
from pathlib import Path

import joblib

from ..core.config import settings


@lru_cache(maxsize=1)
def load_champion() -> tuple[object, dict]:
    model = joblib.load(settings.champion_path)
    sidecar = json.loads(Path(settings.champion_sidecar).read_text())
    return model, sidecar