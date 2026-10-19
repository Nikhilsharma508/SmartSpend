"""Profit curves on the holdout: what a ranked list earns, depth by depth.

The benchmark (Day 3) answers "is the ranking any better than random at one
depth". This module answers the decision question: **at what depth does
targeting stop paying**, and where does profit peak. The construction is the
same one the ledger records — incremental conversions among the top-k,
treated-minus-control, valued on the *money* outcome — plus the piece the
benchmark deliberately skipped: a normal-approximation confidence interval per
depth, so the curve cannot be read as more certain than it is.

Profit per point is the decided formula from
``configs/business_assumptions.yaml``::

    profit(depth) = value_per_conversion * incremental_conversions(depth)
                    - cost_per_contact * n_targeted(depth)
"""

from __future__ import annotations

import numpy as np

from packages.causal.benchmark import BusinessAssumptions
from packages.decision.schemas import (
    AssumptionsSchema,
    ProfitCurve,
    ProfitPoint,
)

#: Standard normal quantile for the reported 95% intervals.
Z_95 = 1.959963984540054


def _assumptions(assumptions: BusinessAssumptions | None) -> BusinessAssumptions:
    return assumptions if assumptions is not None else BusinessAssumptions.from_yaml()


def _assumptions_schema(a: BusinessAssumptions) -> AssumptionsSchema:
    return AssumptionsSchema(
        margin_rate=a.margin_rate,
        cost_per_contact_usd=a.cost_per_contact_usd,
        value_per_conversion_usd=a.value_per_conversion_usd,
    )


def binary_diff_with_se(
    y_treated: np.ndarray, y_control: np.ndarray
) -> tuple[float, float]:
    """Treated-minus-control mean difference of a binary outcome, with its SE.

    The workhorse behind every holdout uplift number in this package: the
    policy value at a depth, a segment's effect, a fairness group's effect.
    """
    y_t = np.asarray(y_treated, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    if len(y_t) == 0 or len(y_c) == 0:
        return float("nan"), float("nan")
    p_t, p_c = y_t.mean(), y_c.mean()
    se = float(np.sqrt(p_t * (1 - p_t) / len(y_t) + p_c * (1 - p_c) / len(y_c)))
    return float(p_t - p_c), se


def profit_point(
    y_money: np.ndarray,
    score: np.ndarray,
    t: np.ndarray,
    depth: float,
    assumptions: BusinessAssumptions,
) -> ProfitPoint:
    """Profit from targeting the top-``depth`` fraction, with a 95% interval.

    ``y_money`` must be the money outcome (conversions when ranking on
    ``visit``) — the same Day-3 rule the benchmark enforces. The interval uses
    the normal approximation of the treated-control difference *within the
    targeted set*, scaled to ``n_targeted``; it widens fast at shallow depths
    because few converters land there.
    """
    y = np.asarray(y_money, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float))
    n_k = max(int(round(depth * len(y))), 1)
    y_k, t_k = y[order][:n_k], np.asarray(t)[order][:n_k]

    treated = t_k == 1
    if treated.sum() == 0 or (~treated).sum() == 0:
        diff, se_rate = float("nan"), float("nan")
    else:
        diff, se_rate = binary_diff_with_se(y_k[treated], y_k[~treated])
    incr = diff * n_k
    se = se_rate * n_k

    v, c = assumptions.value_per_conversion_usd, assumptions.cost_per_contact_usd
    profit = v * incr - c * n_k
    return ProfitPoint(
        depth=round(float(depth), 4),
        n_targeted=n_k,
        incremental_conversions=round(float(incr), 2),
        incremental_conversions_se=round(float(se), 2),
        profit_usd=round(float(profit), 2),
        profit_ci_low_usd=round(float(v * (incr - Z_95 * se) - c * n_k), 2),
        profit_ci_high_usd=round(float(v * (incr + Z_95 * se) - c * n_k), 2),
    )


def optimize_budget(
    y_money: np.ndarray,
    score: np.ndarray,
    t: np.ndarray,
    *,
    assumptions: BusinessAssumptions | None = None,
    depths: np.ndarray | None = None,
    k_max: float = 0.5,
    n_depths: int = 25,
) -> ProfitCurve:
    """Sweep targeting depth and locate break-even and the profit optimum.

    Parameters
    ----------
    y_money, score, t:
        Money outcome, ranking score and treatment indicator, all aligned on
        the **holdout** rows — this evaluates a policy, it never fits one.
    depths:
        Explicit grid; defaults to ``n_depths`` points on ``[0.02, k_max]``.
    """
    a = _assumptions(assumptions)
    if depths is None:
        depths = np.linspace(0.02, k_max, n_depths)
    points = [profit_point(y_money, score, t, float(k), a) for k in depths]

    break_even = next(
        (p.depth for p in points if np.isfinite(p.profit_usd) and p.profit_usd >= 0),
        None,
    )
    finite = [p for p in points if np.isfinite(p.profit_usd)]
    best = max(finite, key=lambda p: p.profit_usd) if finite else None

    return ProfitCurve(
        points=points,
        break_even_depth=break_even,
        recommended_depth=best.depth if best else None,
        recommended_profit_usd=best.profit_usd if best else None,
        assumptions=_assumptions_schema(a),
        note=(
            "Profit valued on the money outcome (conversions), treated-minus-control "
            "within the targeted set; 95% intervals are normal-approximation and "
            "widen sharply at shallow depths. Depths beyond the holdout's own "
            "converter counts are estimates, not measurements — read the interval, "
            "not just the point."
        ),
    )
