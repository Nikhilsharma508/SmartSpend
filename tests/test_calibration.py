"""Tests for ``packages/causal/calibration.py``.

Synthetic data with *known* calibration behaviour, so each test can assert the
right answer rather than just "it ran". A well-calibrated model is one whose
predicted probability equals the observed frequency; a deliberately
over-confident one is off the diagonal in a predictable direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.causal.calibration import (
    brier_by_arm,
    cate_calibration_by_decile,
    fit_recalibrator,
    reliability_by_arm,
)


def _well_calibrated(n: int = 4000, seed: int = 0):
    """Outcomes drawn at exactly the predicted probability -> perfectly calibrated."""
    rng = np.random.default_rng(seed)
    t = rng.binomial(1, 0.5, size=n)
    # Different base rates per arm so the two arms are distinguishable.
    p = np.where(t == 1, 0.30, 0.10)
    y = rng.binomial(1, p)
    return y, p, t


def _over_confident(n: int = 4000, seed: int = 1):
    """Predictions pushed to the extremes; true rate sits in the middle."""
    rng = np.random.default_rng(seed)
    t = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, 0.5, size=n)  # true rate 0.5 in both arms
    p = np.where(rng.random(n) < 0.5, 0.02, 0.98)  # extreme, uninformative
    return y, p, t


def test_brier_by_arm_perfect_calibration_has_zero_skill_gap():
    y, p, t = _well_calibrated()
    table = brier_by_arm(y, p, t)

    assert list(table.index) == ["treated", "control"]
    assert set(table.columns) >= {"n", "base_rate", "brier_model", "skill_vs_base"}
    # A perfectly calibrated model should be close to the base-rate predictor
    # in Brier terms, so skill is near 0 (not strongly positive, not negative).
    assert table["skill_vs_base"].between(-0.05, 0.05).all()
    # Treated base rate ~0.30, control ~0.10 by construction.
    assert 0.25 < table.loc["treated", "base_rate"] < 0.35
    assert 0.05 < table.loc["control", "base_rate"] < 0.15


def test_brier_skill_is_negative_for_over_confident_model():
    """An uninformative-but-extreme model is worse than predicting the base rate."""
    y, p, t = _over_confident()
    table = brier_by_arm(y, p, t)
    assert (table["skill_vs_base"] < 0).all()


def test_reliability_diagonal_for_well_calibrated():
    y, p, t = _well_calibrated()
    rel = reliability_by_arm(y, p, t)

    assert set(rel["arm"]) == {"treated", "control"}
    # Every bin's mean observed rate should sit near its mean predicted rate.
    gap = (rel["mean_predicted"] - rel["mean_observed"]).abs()
    assert gap.mean() < 0.03


def test_reliability_extremes_for_over_confident():
    y, p, t = _over_confident()
    rel = reliability_by_arm(y, p, t)

    # Predictions of 0.02 and 0.98 against a true 0.5 -> large diagonal gap.
    gap = (rel["mean_predicted"] - rel["mean_observed"]).abs()
    assert gap.max() > 0.3


def test_fit_recalibrator_isotonic_fixes_over_confidence():
    """After isotonic on validation, the recalibrated predictions should be
    closer to the true rate than the raw ones."""
    rng = np.random.default_rng(2)
    y_val = rng.binomial(1, 0.5, size=2000)
    p_val = np.where(rng.random(2000) < 0.5, 0.02, 0.98)

    fn = fit_recalibrator(p_val, y_val, method="isotonic")
    # Score a fresh set drawn the same way.
    y_test = rng.binomial(1, 0.5, size=2000)
    p_test = np.where(rng.random(2000) < 0.5, 0.02, 0.98)
    recal = fn(p_test)

    raw_gap = abs(p_test.mean() - y_test.mean())
    new_gap = abs(recal.mean() - y_test.mean())
    assert new_gap < raw_gap
    assert recal.min() >= 0 and recal.max() <= 1


def test_fit_recalibrator_platt_is_monotone():
    """Platt scaling fits a logistic in the score, so its output must be
    non-decreasing in the input. The data needs a genuine positive
    relationship: on pure noise the fitted coefficient can come out negative,
    which is a data artefact, not a bug in Platt."""
    rng = np.random.default_rng(3)
    # Probability rises with p, so the fitted logistic has a positive slope.
    p = rng.uniform(0.1, 0.9, size=1500)
    y = rng.binomial(1, np.clip(0.2 + 0.6 * p, 0, 1))

    fn = fit_recalibrator(p, y, method="platt")
    grid = np.linspace(0.05, 0.95, 50)
    out = fn(grid)
    assert np.all(np.diff(out) >= -1e-9)  # Platt is monotone in p


def test_fit_recalibrator_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown recalibration"):
        fit_recalibrator(np.array([0.1, 0.9]), np.array([0, 1]), method="beta")


def test_cate_calibration_monotonic_for_true_signal():
    """When uplift really rises with the score, observed uplift should rise
    across deciles, so the monotonicity diagnostics should be positive."""
    rng = np.random.default_rng(4)
    n = 6000
    score = rng.uniform(size=n)  # the "predicted" uplift ranking
    t = rng.binomial(1, 0.5, size=n)
    # True effect rises with score: 0.0 at score 0 to 0.4 at score 1.
    true_effect = 0.4 * score
    y = rng.binomial(1, 0.1 + t * true_effect)

    table = cate_calibration_by_decile(y, score, t, n_bins=10)

    assert len(table) == 10
    assert table["n"].sum() == n
    # Observed uplift should generally rise down the deciles.
    assert table.attrs["frac_deciles_monotonic"] > 0.6
    # And it should correlate positively with the prediction.
    assert table.attrs["spearman_pred_vs_obs"] > 0.7


def test_cate_calibration_flat_for_noise_score():
    """A pure-noise ranking shows no trend: observed uplift stays near the
    constant ATE across deciles, so rank correlation is low."""
    rng = np.random.default_rng(5)
    n = 6000
    score = rng.uniform(size=n)  # pure noise, unrelated to effect
    t = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, 0.1 + t * 0.2)  # constant effect for everyone

    table = cate_calibration_by_decile(y, score, t, n_bins=10)
    assert abs(table.attrs["spearman_pred_vs_obs"]) < 0.6


def test_cate_calibration_single_arm_bin_is_nan():
    """If a decile contains only treated (or only control) customers, the
    observed difference is undefined and should be NaN, not a crash."""
    rng = np.random.default_rng(6)
    n = 200
    score = np.linspace(0, 1, n)
    # Force the top half to be all treated, bottom half all control.
    t = (score > 0.5).astype(int)
    y = rng.binomial(1, 0.2, size=n)

    table = cate_calibration_by_decile(y, score, t, n_bins=4)
    assert table["observed_uplift"].isna().any()
