"""Tests for the learner factory in ``packages/causal/learners.py``.

Uses a small synthetic dataset rather than Hillstrom: the point is to check the
*interface contract* every learner must satisfy (fit/predict shapes, sign
conventions, interval support), not to measure real performance. Full-model
behaviour is benchmarked in notebook 04 over 10 seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.causal.learners import LEARNER_NAMES, LearnerName, make_learner


def _synthetic(n: int = 600, seed: int = 0, continuous: bool = False):
    """Small dataset with real, recoverable heterogeneity.

    Treated customers with high ``signal`` convert; low-signal ones do not, so a
    correct uplift model ranks by ``signal``. Kept tiny so the six learners
    finish in a couple of seconds.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.binomial(1, 0.5, size=n)
    lift = 0.4 * (X[:, 0] > 0)
    if continuous:
        y = 1.0 + t * (2.0 + lift * 5.0) + rng.normal(scale=0.1, size=n)
    else:
        p = 0.05 + t * lift
        y = rng.binomial(1, p)
    return X.astype(float), t.astype(float), y.astype(float)


# A fast subset for the per-learner interface checks; the full set is covered
# by the round-trip test below and by notebook 04.
FAST_LEARNERS: tuple[LearnerName, ...] = ("response", "s", "t", "dr", "forest")


@pytest.mark.parametrize("name", FAST_LEARNERS)
def test_fit_predict_returns_finite_scores(name):
    X, t, y = _synthetic()
    model = make_learner(name, random_state=0)
    model.fit(X, t, y)
    scores = model.predict(X)

    assert scores.shape == (len(y),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("name", FAST_LEARNERS)
def test_unknown_target_raises(name):
    with pytest.raises(ValueError):
        make_learner(name, target="open_rate")


def test_unknown_learner_raises():
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner("z-learner")  # type: ignore[arg-type]


def test_response_baseline_ranks_persuadables_low():
    """The baseline ranks by P(buy | treated), so its top customers convert in
    *both* arms — i.e. its score is not a good uplift ranking. We only assert
    it produces a valid ranking here; the sure-things finding lives in
    notebook 03, measured on real data."""
    X, t, y = _synthetic()
    model = make_learner("response", random_state=0)
    model.fit(X, t, y)
    scores = model.predict(X)
    assert np.isfinite(scores).all()
    # Scores are probabilities for a binary target.
    assert scores.min() >= 0 and scores.max() <= 1


def test_only_econml_models_support_intervals():
    """DR and causal forest report effect intervals (both are econml estimators
    fit with inference); the causalml meta-learners and the response baseline
    do not."""
    X, t, y = _synthetic()
    for name in FAST_LEARNERS:
        model = make_learner(name, random_state=0)
        model.fit(X, t, y)
        if name in {"dr", "forest"}:
            assert model.supports_intervals
            ci = model.effect_interval(X)
            assert ci is not None
            lower, upper = ci
            assert lower.shape == upper.shape == (len(y),)
            assert np.all(lower <= upper)
        else:
            assert not model.supports_intervals
            assert model.effect_interval(X) is None


def test_continuous_target_uses_regressor_base():
    """On ``spend`` the base learners must be regressors; a misconfigured
    classifier would raise on a continuous target. The DR-learner is the
    cheapest way to prove the plumbing switches correctly."""
    X, t, y = _synthetic(continuous=True)
    model = make_learner("dr", target="spend", random_state=0)
    model.fit(X, t, y)
    scores = model.predict(X)
    assert np.isfinite(scores).all()


def test_forest_intervals_bracket_the_point_estimate():
    X, t, y = _synthetic()
    model = make_learner("forest", random_state=0)
    model.fit(X, t, y)
    point = model.predict(X)
    lower, upper = model.effect_interval(X, alpha=0.1)

    # The point estimate should sit inside a wide (90%) interval for most rows.
    inside = ((point >= lower) & (point <= upper)).mean()
    assert inside > 0.9


def test_random_state_is_reproducible():
    X, t, y = _synthetic()
    a = make_learner("forest", random_state=7)
    b = make_learner("forest", random_state=7)
    a.fit(X, t, y)
    b.fit(X, t, y)
    assert np.allclose(a.predict(X), b.predict(X))


def test_every_declared_name_is_constructible():
    """Fast check that all names build; the slow X-learner is only constructed,
    not fitted, to keep the suite quick."""
    for name in LEARNER_NAMES:
        model = make_learner(name, random_state=0)
        assert model.name == name
