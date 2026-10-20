"""Model lab endpoints: the catalog snapshot, one model card, the comparison.

- ``GET /api/v1/models``            — all six artifacts + registration status.
  ``?research_mode=true`` flips the ``selectable`` flags; research-only models
  are always *listed* (locked, with their benchmark warning), never hidden.
- ``GET /api/v1/models/compare``    — every registered artifact scored once on
  the shared seed-42 test split (the model lab's table). Cached per process.
- ``GET /api/v1/models/{name}``     — the full per-model card.

No retraining and no LLM here: the lab reads registry sidecars and the shared
Day-3 scoring math, so listing is cheap and comparison is deterministic.
"""

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.causal import catalog
from packages.causal.registry import is_registered, load_meta
from packages.decision import get_model_card
from packages.decision.model_compare import compare_registered_models

router = APIRouter()


def _descriptor(name: str, spec: catalog.ModelSpec, *, research_mode: bool) -> dict[str, Any]:
    """One catalog row: semantics from the catalog, metrics from the sidecar."""
    registered = is_registered(spec.name)
    metrics: dict[str, Any] | None = None
    trained_at: str | None = None
    meta_error: str | None = None
    if registered:
        try:
            meta = load_meta(spec.name)
            metrics = dict(meta.get("metrics", {}))
            trained_at = str(meta.get("created_at"))
        except (FileNotFoundError, ValueError) as err:
            registered = False
            meta_error = f"{type(err).__name__}: {err}"
    return {
        "name": spec.name,
        "label": spec.label,
        "learner": spec.learner,
        "tier": spec.tier,
        "score_kind": spec.score_kind,
        "supports_intervals": spec.supports_intervals,
        "research_only": spec.research_only,
        "use_case": spec.use_case,
        "warning": spec.warning,
        "registered": registered,
        "selectable": catalog.is_allowed(spec.name, research_mode),
        "added_to_allow_list_by_mode": research_mode and spec.research_only,
        "trained_at": trained_at,
        "seed42_metrics": metrics,
        "meta_error": meta_error,
    }


@router.get("")
def catalog_list(
    research_mode: bool = Query(default=False),
) -> dict[str, Any]:
    """The six registered artifacts as the model lab lists them."""
    return {
        "default": catalog.DEFAULT_MODEL,
        "research_mode": research_mode,
        "models": [
            _descriptor(spec.name, spec, research_mode=research_mode)
            for spec in catalog.MODEL_SPECS
        ],
        "score_meanings": dict(catalog.SCORE_MEANING),
    }


@router.get("/compare")
def compare() -> dict[str, Any]:
    """Every registered artifact scored once on the same test rows."""
    return {
        "rows": _comparison().model_dump()["rows"],
        "note": _comparison().model_dump()["note"],
        "score_meanings": dict(catalog.SCORE_MEANING),
    }


@lru_cache(maxsize=1)
def _comparison():
    """The comparison is deterministic per process; artifacts change only on retrain."""
    return compare_registered_models()


@router.get("/{name}")
def model_detail(name: str) -> dict[str, Any]:
    """The full model card for one registered artifact."""
    if name not in {spec.name for spec in catalog.MODEL_SPECS}:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model {name!r}: registered models are "
            f"{[s.name for s in catalog.MODEL_SPECS]}",
        )
    try:
        card = get_model_card(name)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return card.model_dump()
