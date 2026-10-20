"""Scenario comparison: the champion's money against honest alternatives.

A ranked list is only worth something relative to what you would do without
it. :func:`compare_scenarios` evaluates every strategy at the *same budget* on
the holdout — the champion's ranking, random targeting (the null), targeting
everyone, and doing nothing — using the same treated-minus-control construction
the ledger records, so its numbers are comparable with the Day-3 benchmark by
construction rather than by coincidence.

:func:`policy_value_ipw` is the multi-arm sibling: "which email, if any?" can
only be scored honestly with inverse-propensity weighting, because a customer
randomized to Womens E-Mail tells you nothing directly about what Mens E-Mail
would have done.
"""

from __future__ import annotations

import numpy as np

from packages.causal.benchmark import BusinessAssumptions
from packages.decision.profit import Z_95, _assumptions, binary_diff_with_se
from packages.decision.schemas import (
    PolicyValue,
    Scenario,
    ScenarioComparison,
)


def compare_scenarios(
    y_money: np.ndarray,
    score: np.ndarray,
    t: np.ndarray,
    *,
    assumptions: BusinessAssumptions | None = None,
    depth: float = 0.3,
    budget_usd: float | None = None,
    rank_label: str | None = None,
) -> ScenarioComparison:
    """Evaluate four strategies at one budget on the holdout.

    Parameters
    ----------
    y_money, score, t:
        Money outcome, ranking score, treatment — holdout rows.
    depth:
        Targeting depth for the budget-matched strategies; the budget defaults
        to what that depth costs (``n_targeted * cost_per_contact``).
    budget_usd:
        Explicit budget override. ``target_all`` is marked infeasible when the
        budget cannot cover contacting everyone.
    rank_label:
        Name for the ranked strategy ("DR uplift learner ranking"); None keeps
        the historical "champion ranking" label. The *strategy name* is a
        label only — the mathematics is identical for any ranking.
    """
    a = _assumptions(assumptions)
    y = np.asarray(y_money, dtype=float)
    t_arr = np.asarray(t)
    n = len(y)
    n_k = max(int(round(depth * n)), 1)
    budget = float(budget_usd) if budget_usd is not None else n_k * a.cost_per_contact_usd
    ranked_name = f"{rank_label} ranking" if rank_label else "champion ranking"

    def _point(n_targeted: int, incr: float, se: float) -> tuple[float, float, float]:
        profit = a.value_per_conversion_usd * incr - a.cost_per_contact_usd * n_targeted
        lo = a.value_per_conversion_usd * (incr - Z_95 * se) - a.cost_per_contact_usd * n_targeted
        hi = a.value_per_conversion_usd * (incr + Z_95 * se) - a.cost_per_contact_usd * n_targeted
        return float(profit), float(lo), float(hi)

    # Ranked strategy: top-n_k by score, treated-minus-control inside the set.
    # The strategy *name* follows the ranking model in play; the arithmetic is
    # unchanged — comparing the ranking against the honest alternatives.
    order = np.argsort(-np.asarray(score, dtype=float))
    y_k, t_k = y[order][:n_k], t_arr[order][:n_k]
    treated_k = t_k == 1
    diff, se_rate = binary_diff_with_se(y_k[treated_k], y_k[~treated_k])
    incr_champ = diff * n_k
    p, lo, hi = _point(n_k, incr_champ, se_rate * n_k)
    champion = Scenario(
        strategy=ranked_name,
        n_targeted=n_k,
        incremental_conversions=round(float(incr_champ), 2),
        profit_usd=round(p, 2),
        profit_ci_low_usd=round(lo, 2),
        profit_ci_high_usd=round(hi, 2),
        feasible=True,
    )

    # Random targeting: selecting n_k at random has expected incremental
    # conversions n_k * overall ATE; the SE scales the same ATE uncertainty.
    treated_all = t_arr == 1
    diff_all, se_all = binary_diff_with_se(y[treated_all], y[~treated_all])
    incr_random = diff_all * n_k
    p, lo, hi = _point(n_k, incr_random, se_all * n_k)
    random = Scenario(
        strategy="random targeting (null)",
        n_targeted=n_k,
        incremental_conversions=round(float(incr_random), 2),
        profit_usd=round(p, 2),
        profit_ci_low_usd=round(lo, 2),
        profit_ci_high_usd=round(hi, 2),
        feasible=True,
    )

    # Target everyone: the untargeted-email economics.
    incr_all = diff_all * n
    p, lo, hi = _point(n, incr_all, se_all * n)
    target_all = Scenario(
        strategy="target all",
        n_targeted=n,
        incremental_conversions=round(float(incr_all), 2),
        profit_usd=round(p, 2),
        profit_ci_low_usd=round(lo, 2),
        profit_ci_high_usd=round(hi, 2),
        feasible=budget >= n * a.cost_per_contact_usd,
    )

    # Do nothing: zero risk, zero reward — the true baseline.
    nothing = Scenario(
        strategy="no targeting",
        n_targeted=0,
        incremental_conversions=0.0,
        profit_usd=0.0,
        profit_ci_low_usd=0.0,
        profit_ci_high_usd=0.0,
        feasible=True,
    )

    alternatives = [s for s in (random, target_all, nothing) if s.feasible]
    best_alt = max((s.profit_usd for s in alternatives), default=0.0)
    return ScenarioComparison(
        depth=depth,
        budget_usd=round(budget, 2),
        scenarios=[champion, random, target_all, nothing],
        champion_wins=bool(champion.profit_usd > best_alt),
        note=(
            "All strategies priced with the same profit formula on the same "
            "holdout rows; 'random' is the unbiased expectation of sampling n_k "
            "customers at random. CI overlap between champion and random is the "
            "honest read at this sample size — a mean win inside the interval "
            "width is not a win."
        ),
    )


def policy_value_ipw(
    arm_assigned: np.ndarray,
    y: np.ndarray,
    policy_arm: np.ndarray,
    *,
    control_arm: int = 0,
    n_arms: int = 3,
    strategy: str = "multi-arm policy",
    assumptions: BusinessAssumptions | None = None,
) -> PolicyValue:
    """Inverse-propensity value of an assignment policy, on the holdout.

    Under uniform randomization each customer's arm has propensity ``1/A``, so
    the policy value is the mean of ``A * y * 1{actual == policy}`` — customers
    whose randomized arm happens to match what the policy would send, upweighted
    to undo the randomization. The incremental rate vs no-email comes from the
    paired per-row contribution ``A * y * (1{match policy} - 1{match control})``,
    whose sample SE is reported; IPW inflates variance by ~the number of arms,
    which is the honest price of evaluating "which email, if any?" from a
    single randomized experiment.

    ``policy_arm`` must never name an arm that ``arm_assigned`` cannot contain —
    a policy proposing emails the experiment never sent cannot be scored.
    """
    a = _assumptions(assumptions)
    arm = np.asarray(arm_assigned, dtype=int)
    y_arr = np.asarray(y, dtype=float)
    pol = np.asarray(policy_arm, dtype=int)
    if len(arm) != len(y_arr) or len(pol) != len(arm):
        raise ValueError("arm_assigned, y and policy_arm must have equal length")
    if pol.min() < 0 or pol.max() >= n_arms:
        raise ValueError(f"policy_arm names an arm outside [0, {n_arms})")
    if control_arm not in set(np.unique(arm)):
        raise ValueError("control_arm never appears in arm_assigned — cannot anchor")

    contrib = n_arms * y_arr * ((arm == pol).astype(float) - (arm == control_arm).astype(float))
    incr_rate = float(contrib.mean())
    incr_se = float(contrib.std(ddof=1) / np.sqrt(len(contrib)))
    value_rate = float(n_arms * (y_arr * (arm == pol)).mean())

    contacted = int((pol != control_arm).sum())
    profit = a.value_per_conversion_usd * incr_rate * len(pol) - a.cost_per_contact_usd * contacted
    return PolicyValue(
        strategy=strategy,
        n_customers=len(pol),
        value_rate=round(value_rate, 5),
        value_rate_se=round(float(n_arms * y_arr.std(ddof=1) / np.sqrt(len(y_arr))), 5),
        incremental_rate_vs_no_email=round(incr_rate, 5),
        incremental_conversions=round(incr_rate * len(pol), 2),
        incremental_ci_low=round(incr_rate - Z_95 * incr_se, 5),
        incremental_ci_high=round(incr_rate + Z_95 * incr_se, 5),
        profit_usd=round(float(profit), 2),
        note=(
            "IPW under uniform randomization (propensity 1/n_arms); the paired "
            "contribution SE is reported. 'Incremental' is versus sending no "
            "email at all."
        ),
    )
