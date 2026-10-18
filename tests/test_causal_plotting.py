"""Tests for ``packages/causal/plotting.py`` — the Qini curve construction."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no window during tests

import numpy as np
import pytest

from packages.causal.plotting import (
    cumulative_uplift_curve,
    perfect_qini,
    plot_qini,
)


def _perfect_case(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n)
    p = 0.05 + t * (0.30 * (signal > 0))
    y = rng.binomial(1, p)
    return y, t, signal


def test_curve_ascending_depth_and_length():
    y, t, signal = _perfect_case()
    depth, lift = cumulative_uplift_curve(y, signal, t, n_points=50)

    assert len(depth) == len(lift) == 50
    assert depth[0] > 0 and depth[-1] == pytest.approx(1.0)
    assert np.all(np.diff(depth) >= 0)


def test_good_ranking_beats_inverted_over_most_of_the_curve():
    """A good ranking keeps cumulative lift above the inverted one.

    Both curves start at 0 (the very top of a ranking can contain no control
    customers yet) and converge at full depth (ordering stops mattering once
    everyone is targeted), so the meaningful property is that the good curve
    dominates across the middle of the range.
    """
    y, t, signal = _perfect_case()
    _, good = cumulative_uplift_curve(y, signal, t, n_points=100)
    _, bad = cumulative_uplift_curve(y, -signal, t, n_points=100)

    assert (good > bad).mean() > 0.9


def test_perfect_qini_is_concave_and_peaks_above_models():
    """The perfect curve must front-load effect: steep rise, then a peak above
    the real curves, returning to the common endpoint at full depth."""
    y, t, signal = _perfect_case()
    depth, curve = perfect_qini(y, t, n_points=100)
    _, good = cumulative_uplift_curve(y, signal, t, n_points=100)

    assert depth[0] > 0 and depth[-1] == pytest.approx(1.0)
    # Concave: the peak sits well above the final value (unlike a straight
    # diagonal, which would peak exactly at the end).
    assert curve.max() > curve[-1] * 1.05
    # And it dominates a real (merely good) ranking for most of the range.
    assert (curve >= good - 1e-9).mean() > 0.9


def test_perfect_qini_ends_at_same_total_as_any_ranking():
    """At 100% targeted every ranking converges, so the perfect curve's endpoint
    must match a real curve's endpoint."""
    y, t, signal = _perfect_case()
    _, perfect = perfect_qini(y, t, n_points=20)
    _, real = cumulative_uplift_curve(y, signal, t, n_points=20)

    assert perfect[-1] == pytest.approx(real[-1])


def test_plot_qini_saves_figure(tmp_path, monkeypatch):
    monkeypatch.setattr("packages.plotting.FIGURES_DIR", tmp_path)
    y, t, signal = _perfect_case()

    fig = plot_qini(
        y,
        t,
        {"perfect signal": signal, "random": np.random.default_rng(1).normal(size=len(y))},
        name="unit_test_qini",
    )
    assert (tmp_path / "unit_test_qini.png").exists()
    assert len(fig.axes) == 1
    # perfect reference + 2 model lines + the axhline(0) baseline
    assert len(fig.axes[0].lines) == 4


def test_plot_qini_skips_saving_when_name_is_none():
    y, t, signal = _perfect_case()
    fig = plot_qini(y, t, {"m": signal}, name=None, show_perfect=False)
    # One model line + the axhline(0) baseline; no perfect reference.
    assert len(fig.axes[0].lines) == 2
