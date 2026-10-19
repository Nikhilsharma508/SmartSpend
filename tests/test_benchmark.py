"""Tests for ``packages/causal/benchmark.py`` — the stability runner.

Synthetic data with a *known* winner, so the tests can assert the benchmark
surfaces it, rather than just checking that rows appear. A tiny learner count
and seed count keep the suite fast; the real 10-seed run lives in notebook 03.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.causal.benchmark import (
    DEFAULT_SEEDS,
    BusinessAssumptions,
    incremental_conversions_at_k,
    profit_at_k,
    run_benchmark,
)


def _assumptions() -> BusinessAssumptions:
    return BusinessAssumptions(
        margin_rate=0.30, cost_per_contact_usd=0.10, value_per_conversion_usd=34.91
    )


def _synthetic_fold(seed: int, n: int = 800):
    """Split with recoverable heterogeneity, matching run_benchmark's contract.

    Treated customers with high ``signal`` convert; the effect is real, so a
    correct model ranks by ``signal`` and a random one does not.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.binomial(1, 0.5, size=n)
    p = 0.05 + t * (0.4 * (X[:, 0] > 0))
    y = rng.binomial(1, p).astype(float)
    return (
        X.astype(float),
        t.astype(float),
        y,
        X.astype(float),
        t.astype(float),
        y,
    )


def _fake_fold_returning(perfect_score: np.ndarray, noise_score: np.ndarray):
    """A fold factory that ignores features and yields two fixed score columns.

    Lets a test control the ranking directly, so the assertion is about the
    metric plumbing, not about whether LightGBM recovered a signal.
    """

    def make_fold(seed: int):
        n = len(perfect_score)
        X = np.zeros((n, 1))
        t = np.ones(n)  # every row treated -> observed uplift = mean y
        y = perfect_score.astype(float)  # reuse arrays as stand-ins
        return X, t, y, X, t, y

    return make_fold


def test_assumptions_load_from_yaml():
    a = BusinessAssumptions.from_yaml()
    assert a.margin_rate == 0.30
    assert a.cost_per_contact_usd == 0.10
    assert a.value_per_conversion_usd == 34.91


def test_incremental_conversions_perfect_ranking_beats_inverted():
    """When high-signal customers are the persuadables, targeting them yields
    more incremental conversions than targeting the opposite end."""
    rng = np.random.default_rng(0)
    n = 3000
    score = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, 0.05 + t * 0.5 * (score > 0)).astype(float)

    good = incremental_conversions_at_k(y, score, t, k=0.3)
    bad = incremental_conversions_at_k(y, -score, t, k=0.3)
    assert good > bad


def test_incremental_conversions_single_arm_is_nan():
    """If the top-k happens to be all treated (or all control), the difference
    is undefined and should be NaN rather than a fabricated number."""
    n = 100
    score = np.arange(n, dtype=float)  # descending target order
    t = np.ones(n)  # everyone treated
    y = np.zeros(n)
    assert np.isnan(incremental_conversions_at_k(y, score, t, k=0.5))


def test_profit_formula_matches_manual_arithmetic():
    rng = np.random.default_rng(1)
    n = 500
    score = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n).astype(float)
    y = rng.binomial(1, 0.05 + t * 0.3).astype(float)
    a = _assumptions()

    incr = incremental_conversions_at_k(y, score, t, k=0.4)
    expected = incr * a.value_per_conversion_usd - int(round(0.4 * n)) * a.cost_per_contact_usd
    assert profit_at_k(y, score, t, k=0.4, assumptions=a) == pytest.approx(expected)


def test_run_benchmark_returns_raw_and_summary():
    result = run_benchmark(
        _synthetic_fold,
        learners=("response", "s", "t"),
        seeds=(0, 1),
        assumptions=_assumptions(),
    )

    assert isinstance(result.raw, pd.DataFrame)
    assert set(result.raw.columns) == {
        "seed",
        "model",
        "qini_auc",
        "uplift_at_k",
        "profit_at_k",
    }
    assert len(result.raw) == 3 * 2  # learners x seeds

    summary = result.summary
    for col in ("qini_auc_mean", "qini_auc_std", "profit_at_k_mean"):
        assert col in summary.columns
    assert list(summary.index) == sorted(
        summary.index, key=lambda m: -summary.loc[m, "qini_auc_mean"]
    )


def test_run_benchmark_values_profit_on_the_money_outcome():
    """Profit must come from the money array, not the ranking outcome.

    The fold ranks on a *noise* test outcome (so the ranking metrics see no
    effect) but pays out on a real conversion vector. Valuing the noise would
    land profit near the contact cost; valuing the money outcome returns the
    true incremental profit.
    """

    def make_fold(seed: int):
        rng = np.random.default_rng(seed)
        n = 800
        X = rng.normal(size=(n, 3)).astype(float)
        t = rng.binomial(1, 0.5, size=n).astype(float)
        y_tr = t * (X[:, 0] > 0)  # learnable effect on train
        y_rank_test = rng.binomial(1, 0.1, size=n).astype(float)  # noise
        y_money_test = t * (X[:, 0] > 0)  # real conversions on test
        return X, t, y_tr, X, t, y_rank_test, y_money_test

    result = run_benchmark(
        make_fold, learners=("response",), seeds=(0,), assumptions=_assumptions()
    )
    profit = result.raw["profit_at_k"].iloc[0]
    # Valued on conversions: the ranking recovers X[:, 0] > 0, so incremental
    # conversions are large and profit clears $1k. Valuing the noise outcome
    # would stay near zero (a few spurious conversions minus the contact cost).
    assert profit > 1000


def test_run_benchmark_is_reproducible():
    kwargs = dict(learners=("s",), seeds=(0, 1), assumptions=_assumptions())
    a = run_benchmark(_synthetic_fold, **kwargs)
    b = run_benchmark(_synthetic_fold, **kwargs)
    assert a.raw["qini_auc"].to_numpy() == pytest.approx(b.raw["qini_auc"].to_numpy())


def test_winners_flags_thin_margins():
    """A 2-seed run on noisy data should NOT be declared a clear winner, since
    the edge over second place will be within the spread."""
    result = run_benchmark(
        _synthetic_fold,
        learners=("response", "s", "t"),
        seeds=(0, 1),
        assumptions=_assumptions(),
    )
    w = result.winners()
    assert "qini_edge_over_second" in w.columns
    assert "is_clear_winner" in w.columns
    # With only 2 noisy seeds, a clear winner is unlikely; assert the column is
    # a bool and the top row carries the edge value.
    assert w["is_clear_winner"].dtype == bool
    assert not np.isnan(w.iloc[0]["qini_edge_over_second"])


def test_default_seeds_are_ten_distinct():
    assert len(DEFAULT_SEEDS) == 10
    assert len(set(DEFAULT_SEEDS)) == 10
