"""The app's dashboard payload — computed live from the registered champion.

The web planner used to replay a recorded snapshot while `apps/api` was being
built. Now every number on every page comes from these calls: the profit curve,
scenario comparison, segment tree and power grid are the same deterministic
Day-4 tools the copilot quotes, run against the champion's seed-42 holdout.

The API warms the one expensive piece (a ~20 s DRPolicyTree fit) in a
background thread at boot and caches the whole payload for the process; a
retrained champion is picked up by restarting the API.
"""

import contextlib
import threading
from datetime import date
from functools import lru_cache
from typing import Any

from fastapi import APIRouter

from packages.copilot.planner import get_context
from packages.decision import compare_scenarios, design_ab_test, optimize_budget

router = APIRouter()

#: The MDE grid the test-design page renders (absolute lifts on the visit rate).
MDE_GRID = (0.005, 0.01, 0.02, 0.03)
DAILY_ELIGIBLE = 2000


#: Serialise the one expensive fit so a request arriving during warm-up waits
#: for the same computation instead of starting a second policy-tree fit.
_WARM_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _compute_overview() -> dict[str, Any]:
    ctx = get_context()
    a = ctx.assumptions

    curve = optimize_budget(ctx.y_money_te, ctx.score, ctx.t_te, assumptions=a)
    depth = curve.recommended_depth if curve.recommended_depth is not None else 0.30
    scenarios = compare_scenarios(
        ctx.y_money_te, ctx.score, ctx.t_te, assumptions=a, depth=depth
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
                "champion's seed-42 holdout (packages.copilot.planner.build_context). "
                "No recorded snapshot is served."
            ),
        },
        "context": {
            "n_test": ctx.n_test,
            "baseline_visit_rate": ctx.baseline_visit_rate,
            "feature_names": list(ctx.feature_names),
            "champion_random_state": ctx.meta.get("random_state"),
            "data_sha256": ctx.meta.get("data_sha256"),
        },
        "assumptions": curve.assumptions.model_dump(),
        "profit_curve": curve.model_dump(),
        "scenarios": scenarios.model_dump(),
        "segment_profile": ctx.segment_profile().model_dump(),
        "ab_designs": [d.model_dump() for d in designs],
        "model_card": ctx.model_card().model_dump(),
    }


def build_overview() -> dict[str, Any]:
    with _WARM_LOCK:
        return _compute_overview()


def warm_overview() -> None:
    """Fit the policy tree at boot so the first page view is instant."""
    with contextlib.suppress(Exception):  # warm-up must never take the API down
        build_overview()


@router.get("")
def overview() -> dict[str, Any]:
    """Everything the planner, segments and test-design pages need."""
    return build_overview()
