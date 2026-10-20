"""The app's dashboard payload — computed live from the registered model.

The web planner used to replay a recorded snapshot while `apps/api` was being
built. Now every number on every page comes from these calls: the profit curve,
scenario comparison, segment tree and power grid are the same deterministic
Day-4 tools the copilot quotes, run against the selected model's seed-42
holdout.

The API warms the one expensive piece (a ~20 s DRPolicyTree fit) in a
background thread at boot and caches the payload per model for the process; a
retrained model is picked up by restarting the API.

Research-only models (S/T/X/forest) are refused here unless
``research_mode=true`` — normal pages never serve a weak benchmark learner.
"""

import contextlib
import threading
from datetime import date
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.causal import catalog
from packages.copilot.planner import get_context
from packages.decision import compare_scenarios, design_ab_test, optimize_budget

router = APIRouter()

#: The MDE grid the test-design page renders (absolute lifts on the visit rate).
MDE_GRID = (0.005, 0.01, 0.02, 0.03)
DAILY_ELIGIBLE = 2000


#: Serialise the one expensive fit so a request arriving during warm-up waits
#: for the same computation instead of starting a second policy-tree fit.
_WARM_LOCK = threading.Lock()


def _validate_model_param(model: str, research_mode: bool) -> None:
    """Unknown names 404; research-only names without the mode 422."""
    try:
        catalog.get_spec(model)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    if not catalog.is_allowed(model, research_mode):
        raise HTTPException(
            status_code=422,
            detail=(
                f"model {model!r} is research-only: enable Research Mode to view it "
                f"(normal mode allows {list(catalog.allowed_models(research_mode))})"
            ),
        )


@lru_cache(maxsize=8)
def _compute_overview(model: str) -> dict[str, Any]:
    ctx = get_context(model)
    a = ctx.assumptions

    curve = optimize_budget(ctx.y_money_te, ctx.score, ctx.t_te, assumptions=a)
    depth = curve.recommended_depth if curve.recommended_depth is not None else 0.30
    scenarios = compare_scenarios(
        ctx.y_money_te,
        ctx.score,
        ctx.t_te,
        assumptions=a,
        depth=depth,
        rank_label=ctx.label or ctx.model_name,
    )
    designs = [
        design_ab_test(ctx.baseline_visit_rate, mde, daily_eligible=DAILY_ELIGIBLE)
        for mde in MDE_GRID
    ]

    return {
        "meta": {
            "generated": date.today().isoformat(),
            "generator": (
                "Computed live by apps/api: packages.decision tools over the registered "
                f"{model}'s recorded seed holdout (packages.copilot.planner.build_context). "
                "No recorded snapshot is served."
            ),
        },
        "context": {
            "n_test": ctx.n_test,
            "baseline_visit_rate": ctx.baseline_visit_rate,
            "feature_names": list(ctx.feature_names),
            "model_name": ctx.model_name,
            "model_label": ctx.model_card().label,
            "learner": ctx.learner,
            "score_kind": ctx.score_kind,
            "tier": ctx.tier,
            "supports_intervals": ctx.supports_intervals,
            "research_only": ctx.research_only,
            "model_random_state": ctx.meta.get("random_state"),
            "data_sha256": ctx.meta.get("data_sha256"),
        },
        "assumptions": curve.assumptions.model_dump(),
        "profit_curve": curve.model_dump(),
        "scenarios": scenarios.model_dump(),
        "segment_profile": ctx.segment_profile().model_dump(),
        "ab_designs": [d.model_dump() for d in designs],
        "model_card": ctx.model_card().model_dump(),
    }


def build_overview(model: str = "champion") -> dict[str, Any]:
    with _WARM_LOCK:
        return _compute_overview(model)


def warm_overview() -> None:
    """Fit the policy tree at boot so the first page view is instant."""
    with contextlib.suppress(Exception):  # warm-up must never take the API down
        build_overview()


@router.get("")
def overview(
    model: str = Query(default="champion"),
    research_mode: bool = Query(default=False),
) -> dict[str, Any]:
    """Everything the planner, segments and test-design pages need."""
    _validate_model_param(model, research_mode)
    return build_overview(model)
