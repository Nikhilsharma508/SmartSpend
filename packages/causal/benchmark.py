"""Day-3 stability runner: repeat every model across seeds and report spread.

One split can flatter a model. A ranking difference of 0.003 in Qini AUC — the
kind of gap Day 2 saw between the X-learner and the response baseline — is well
inside single-split noise, so a champion picked from one run is a coin flip with
good PR. This module repeats the whole train/score/evaluate loop over many seeds
and reports **mean ± spread**, which is the only way to tell a real edge from
luck.

It also computes **profit at k** under the decided business assumptions, because
the project's stated rule is that the champion is chosen on Qini **and** profit,
never on Brier. Profit here is the conversion-route formula from
``configs/business_assumptions.yaml``:

    profit(k) = v * incremental_conversions(k) - c * n_targeted(k)

The incremental conversions at depth k are estimated the same way the Qini curve
does it: among the top-k targeted, the treated-minus-control outcome difference
scaled to the number targeted.

**Profit is valued on the money outcome, not the ranking outcome.** A model
trained to rank by ``visit`` is still paid out in ``conversion`` dollars: ranking
visits and counting them as $34.91 each would overstate profit by an order of
magnitude (Day 1 measured visits at ~11x the conversion rate). ``make_fold`` can
therefore return a 7th array — the money outcome on the test rows, aligned by
construction since it comes from the same splitter. When it is omitted, profit
falls back to the ranking outcome, which is only correct when that outcome *is*
the money (e.g. ranking and paying on ``spend`` itself).

Notebook 04 drives this over the real Hillstrom data; the tests here use a small
synthetic set so the suite stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from packages.causal.learners import LEARNER_NAMES, LearnerName, make_learner
from packages.causal.metrics import qini_auc, uplift_at_k

#: Default seeds for the loop. Kept as an explicit tuple so a benchmark run is
#: fully described by its inputs and reproducible from them alone.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True)
class BusinessAssumptions:
    """The profit constants the benchmark optimises under.

    Loaded from ``configs/business_assumptions.yaml`` — the single source the
    copilot also reads — so a benchmark profit figure and a copilot brief figure
    can never drift apart.
    """

    margin_rate: float
    cost_per_contact_usd: float
    value_per_conversion_usd: float

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> BusinessAssumptions:
        path = path or (
            Path(__file__).resolve().parents[2] / "configs" / "business_assumptions.yaml"
        )
        raw = yaml.safe_load(path.read_text())["hillstrom"]
        return cls(
            margin_rate=float(raw["margin_rate"]),
            cost_per_contact_usd=float(raw["cost_per_contact_usd"]),
            value_per_conversion_usd=float(raw["value_per_conversion_usd"]),
        )


def incremental_conversions_at_k(
    y: np.ndarray, uplift_score: np.ndarray, treatment: np.ndarray, k: float
) -> float:
    """Incremental conversions from targeting the top-``k`` fraction.

    Estimated as the treated-minus-control outcome difference among the targeted
    set, scaled by the number targeted. This is the same construction the Qini
    curve uses, evaluated at one depth.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    order = np.argsort(-np.asarray(uplift_score))
    n_k = max(int(round(k * len(y))), 1)
    y_k, t_k = y[order][:n_k], t[order][:n_k]

    treated = t_k == 1
    if treated.sum() == 0 or (~treated).sum() == 0:
        return float("nan")
    uplift_rate = y_k[treated].mean() - y_k[~treated].mean()
    return float(uplift_rate * n_k)


def profit_at_k(
    y: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    k: float,
    assumptions: BusinessAssumptions,
) -> float:
    """Net profit from targeting the top-``k`` fraction, in USD.

    Uses the conversion route: incremental conversions times value per
    conversion, minus the contact cost for everyone targeted. ``y`` must be the
    **money outcome** (conversions or spend) — not the ranking proxy — because
    the value-per-conversion constant only pays out actual conversions. Returns
    USD for the whole targeted segment (not per customer); divide by ``k * n``
    for a per-contact figure.
    """
    n_k = max(int(round(k * len(y))), 1)
    incr = incremental_conversions_at_k(y, uplift_score, treatment, k)
    revenue = incr * assumptions.value_per_conversion_usd
    cost = n_k * assumptions.cost_per_contact_usd
    return float(revenue - cost)


@dataclass
class BenchmarkResult:
    """Tidy long-form results plus the pivot tables the notebook displays.

    ``raw`` keeps every (seed, model) row so the spread is auditable; the
    ``summary`` property pivots it to model x metric with mean and std.
    """

    raw: pd.DataFrame
    assumptions: BusinessAssumptions
    seeds: tuple[int, ...] = field(default=DEFAULT_SEEDS)
    k: float = 0.3

    @property
    def summary(self) -> pd.DataFrame:
        """Mean ± std per model, for each metric, sorted by Qini AUC mean."""
        metrics = ["qini_auc", "uplift_at_k", "profit_at_k"]
        grouped = self.raw.groupby("model")[metrics]
        out = grouped.agg(["mean", "std"])
        # Flatten the multiindex columns: qini_auc_mean, qini_auc_std, ...
        out.columns = [f"{m}_{s}" for m, s in out.columns]
        return out.sort_values("qini_auc_mean", ascending=False).round(4)

    def winners(self) -> pd.DataFrame:
        """The summary plus a stability flag: does the best model win by a
        margin larger than its own spread?

        A mean edge smaller than one standard deviation is not a result worth
        acting on; this surfaces that honestly rather than declaring a champion
        from noise.
        """
        s = self.summary.copy()
        top = s.iloc[0]
        second = s.iloc[1] if len(s) > 1 else None
        s["qini_edge_over_second"] = np.nan
        if second is not None:
            s.iloc[0, s.columns.get_loc("qini_edge_over_second")] = round(
                top["qini_auc_mean"] - second["qini_auc_mean"], 4
            )
        # Is the edge larger than the top model's own spread?
        s["is_clear_winner"] = False
        if second is not None:
            edge = top["qini_auc_mean"] - second["qini_auc_mean"]
            s.iloc[0, s.columns.get_loc("is_clear_winner")] = bool(
                edge > top["qini_auc_std"]
            )
        return s


def run_benchmark(
    make_fold,
    *,
    target: str = "visit",
    learners: tuple[LearnerName, ...] = LEARNER_NAMES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    k: float = 0.3,
    assumptions: BusinessAssumptions | None = None,
) -> BenchmarkResult:
    """Fit every learner on every seed's split and collect the metrics.

    Parameters
    ----------
    make_fold:
        Callable ``make_fold(seed) -> (X_train, t_train, y_train, X_test,
        t_test, y_test)`` or ``... , y_money_test)``. Keeping the split logic
        outside this function means the benchmark can run on any dataset
        (Hillstrom now, a synthetic set in tests) without this module knowing
        about either. When the 7th array is given it is the **money outcome on
        the test rows** (``conversion`` when ranking on ``visit``): Qini and
        uplift are measured on ``y_test``, profit on ``y_money_test``, so a
        model can rank visits but be paid in conversion dollars. Without it,
        profit uses ``y_test`` — correct only when that outcome is itself
        money.
    target:
        Outcome being modelled; passed to the learner factory so ``spend``
        switches base learners to regressors.
    learners, seeds, k:
        What to fit, over how many splits, and at what targeting depth to
        evaluate profit.
    assumptions:
        Profit constants; defaults to the YAML config.

    Returns
    -------
    BenchmarkResult
    """
    assumptions = assumptions or BusinessAssumptions.from_yaml()
    rows: list[dict] = []

    for seed in seeds:
        fold = make_fold(seed)
        X_tr, t_tr, y_tr, X_te, t_te, y_te = fold[:6]
        # Money outcome for profit: separate from the ranking outcome when the
        # fold provides one; otherwise the ranking outcome is the money.
        y_money_te = fold[6] if len(fold) == 7 else y_te
        for name in learners:
            model = make_learner(name, target=target, random_state=seed)
            model.fit(X_tr, t_tr, y_tr)
            score = model.predict(X_te)
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "qini_auc": qini_auc(y_te, score, t_te),
                    "uplift_at_k": uplift_at_k(y_te, score, t_te, k=k),
                    "profit_at_k": profit_at_k(
                        y_money_te, score, t_te, k, assumptions
                    ),
                }
            )

    return BenchmarkResult(
        raw=pd.DataFrame(rows), assumptions=assumptions, seeds=seeds, k=k
    )
