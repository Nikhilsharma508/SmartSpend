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


def test_uplift_at_k_handles_continuous_outcome():
    """sklift's uplift_at_k requires a binary y_true and raises on a continuous
    one (e.g. spend). Ours must not: the arithmetic is a plain difference of
    means, valid for any numeric outcome. Regression test for the spend path."""
    rng = np.random.default_rng(11)
    n = 3000
    score = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n)
    # Continuous outcome whose mean rises with the score only for the treated.
    y = 1.0 + t * (3.0 * (score > 0)) + rng.normal(scale=0.1, size=n)

    val = uplift_at_k(y, score, t, k=0.3)
    assert np.isfinite(val)
    # A good ranking front-loads the effect, so the top 30% should show at
    # least as much lift as a random ranking would.
    rnd = uplift_at_k(y, random_uplift(y, t), t, k=0.3)
    assert val > rnd


def test_uplift_at_k_matches_sklearn_free_reference():
    """Hand-checked reference for the 'overall' strategy: sort by score
    descending, take the top k fraction, and difference the two arm means.

    Six rows, k=0.5 -> top 3 (indices 0,1,2). Their t=[1,1,0], y=[1,1,0], so
    treated mean = 1.0 and control mean = 0.0 -> uplift 1.0."""
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    t = np.array([1, 1, 0, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])  # already descending

    assert uplift_at_k(y, score, t, k=0.5) == pytest.approx(1.0)


def test_uplift_at_k_by_group_scales_each_arm_separately():
    """A float k in the 'by_group' strategy scales *each arm's own count*, so
    the two subsets differ in size. Regression test: reusing one shared size
    silently diverges from scikit-uplift's definition."""
    rng = np.random.default_rng(12)
    n = 2000
    # Highly unbalanced arms, so a shared vs per-arm size clearly differs.
    t = rng.binomial(1, 0.2, size=n)
    score = rng.normal(size=n)
    y = rng.binomial(1, 0.1 + t * 0.5 * (score > 0)).astype(float)

    val = uplift_at_k(y, score, t, k=0.5, strategy="by_group")
    assert np.isfinite(val)

    # Recompute the expected value independently: top 50% of each arm.
    order = np.argsort(score, kind="mergesort")[::-1]
    ctrl_order, trmnt_order = order[t[order] == 0], order[t[order] == 1]
    n_ctrl, n_trmnt = int((t == 0).sum() * 0.5), int((t == 1).sum() * 0.5)
    expected = y[trmnt_order][:n_trmnt].mean() - y[ctrl_order][:n_ctrl].mean()
    assert val == pytest.approx(expected)


def test_uplift_at_k_rejects_bad_strategy():
    y, t, signal = _perfect_case(n=200)
    with pytest.raises(ValueError, match="strategy"):
        uplift_at_k(y, signal, t, k=0.3, strategy="treated")


def test_uplift_at_k_rejects_bad_k():
    y, t, signal = _perfect_case(n=200)
    with pytest.raises(ValueError, match="k"):
        uplift_at_k(y, signal, t, k=1.5)


def test_ranking_table_continuous_outcome_drops_binary_only_columns():
    """For a continuous target, Qini AUC / AUUC are undefined (sklift raises),
    so the table must omit them and keep only the incremental-at-k columns."""
    rng = np.random.default_rng(13)
    n = 1500
    score = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n)
    y = 1.0 + t * (2.0 * (score > 0)) + rng.normal(scale=0.1, size=n)

    table = ranking_table(y, t, {"good": score, "random": random_uplift(y, t)})
    assert "qini_auc" not in table.columns and "uplift_auc" not in table.columns
    assert any(c.startswith("uplift_at_") for c in table.columns)
    # Sorted by the first uplift_at_* column, descending.
    first_col = next(c for c in table.columns if c.startswith("uplift_at_"))
    assert list(table.index) == list(
        table.sort_values(first_col, ascending=False).index
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
