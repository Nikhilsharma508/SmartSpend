"""Tests for ``packages/causal/metrics.py`` on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from packages.causal.metrics import (
    DEFAULT_KS,
    qini_auc,
    random_uplift,
    ranking_table,
    uplift_at_k,
    uplift_auc,
)


def _perfect_case(n: int = 2000, seed: int = 0):
    """Synthetic data where uplift is perfectly recoverable from ``signal``.

    Treated customers with a high ``signal`` convert; control ones do not, so a
    score proportional to ``signal`` is the ideal ranking. That makes the Qini
    AUC of ``signal`` high and that of ``-signal`` negative, which is the
    property the tests assert.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n)
    base = 0.05
    lift = 0.30 * (signal > 0)
    p = base + t * lift
    y = rng.binomial(1, p)
    return y, t, signal


def test_qini_auc_orders_perfect_vs_inverted():
    y, t, signal = _perfect_case()
    good = qini_auc(y, signal, t)
    bad = qini_auc(y, -signal, t)

    assert good > 0.3
    assert bad < 0
    assert good > bad


def test_uplift_auc_is_finite_and_ordered():
    y, t, signal = _perfect_case()
    assert np.isfinite(uplift_auc(y, signal, t))
    assert uplift_auc(y, signal, t) > uplift_auc(y, -signal, t)


def test_uplift_at_k_decreases_as_k_shrinks():
    y, t, signal = _perfect_case()
    # A perfect ranking front-loads the effect, so the top 10% should show at
    # least as much per-customer lift as the top 30%.
    at10 = uplift_at_k(y, signal, t, k=0.1)
    at30 = uplift_at_k(y, signal, t, k=0.3)
    assert at10 >= at30


def test_random_uplift_is_near_zero_and_reproducible():
    y, t, _ = _perfect_case()
    r1 = random_uplift(y, t, random_state=7)
    r2 = random_uplift(y, t, random_state=7)

    assert np.array_equal(r1, r2)
    # Averaged over many permutations the ranking carries no real signal, so
    # its Qini AUC should land close to zero.
    assert abs(qini_auc(y, r1, t)) < 0.05


def test_ranking_table_shape_and_sorting():
    y, t, signal = _perfect_case()
    table = ranking_table(
        y,
        t,
        {
            "perfect": signal,
            "inverted": -signal,
            "random": random_uplift(y, t),
        },
    )

    # Sorted by Qini AUC descending: the perfect ranking leads, and the
    # inverted one (which actively ranks persuadables last) is worse than
    # random, so it comes last.
    assert list(table.index) == ["perfect", "random", "inverted"]
    assert "qini_auc" in table.columns and "uplift_auc" in table.columns
    for k in DEFAULT_KS:
        assert f"uplift_at_{int(k * 100)}pct" in table.columns
    assert table.index.name == "model"
    # All values rounded to 4dp for display.
    assert np.allclose(table.to_numpy(), np.round(table.to_numpy(), 4))


def test_random_uplift_length_matches_input():
    y, t, _ = _perfect_case(n=500)
    assert len(random_uplift(y, t)) == 500


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
