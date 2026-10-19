"""Tests for the Day-4 decision layer in ``packages/decision/``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.causal.benchmark import (
    BusinessAssumptions,
    incremental_conversions_at_k,
    profit_at_k,
)
from packages.causal.learners import make_learner
from packages.causal.registry import REQUIRED_META_FIELDS, save_model
from packages.decision import (
    compare_scenarios,
    design_ab_test,
    get_model_card,
    get_segment_profile,
    optimize_budget,
    policy_value_ipw,
    profit_point,
)


def _register_dummy_champion(tmp_path):
    """Minimal registered artifact: tiny fitted response model + full meta."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    t = pd.Series(rng.integers(0, 2, 200))
    y = pd.Series(rng.integers(0, 2, 200))
    model = make_learner("response", random_state=0)
    model.fit(X.to_numpy(), t.to_numpy(), y.to_numpy())
    meta = {f: "x" for f in REQUIRED_META_FIELDS}
    meta.update(
        {
            "learner": "response",
            "target": "visit",
            "train_rows": 200,
            "metrics": {"qini_auc": 0.02},
            "assumptions": {
                "margin_rate": 0.3,
                "cost_per_contact_usd": 0.1,
                "value_per_conversion_usd": 34.91,
            },
        }
    )
    save_model(model, "champion", meta=meta, models_dir=tmp_path)


def _synthetic(n: int = 2000, seed: int = 0, effect: float = 0.10):
    """Ranking score + treatment + money outcome with a planted uplift on x0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.10 + t * (X[:, 0] > 0.5) * effect)
    score = X[:, 0] + rng.normal(0, 0.3, n)
    return score, t, y.astype(float)


# ---------------------------------------------------------------- optimize_budget


def test_optimize_budget_curve_shape_and_intervals():
    score, t, y = _synthetic()
    curve = optimize_budget(y, score, t)

    depths = [p.depth for p in curve.points]
    assert depths == sorted(depths)
    assert [p.n_targeted for p in curve.points] == sorted(p.n_targeted for p in curve.points)
    for p in curve.points:
        assert p.profit_ci_low_usd <= p.profit_usd <= p.profit_ci_high_usd
    assert curve.recommended_depth in depths
    assert curve.recommended_profit_usd == max(p.profit_usd for p in curve.points)
    assert "money outcome" in curve.note


def test_optimize_budget_finds_planted_effect():
    score, t, y = _synthetic(n=4000, seed=3, effect=0.30)
    curve = optimize_budget(y, score, t, k_max=0.6, n_depths=15)

    assert curve.break_even_depth is not None  # strong signal must clear zero
    assert curve.recommended_profit_usd > 0


def test_profit_point_matches_benchmark_formula():
    """The decision layer prices the same construction as the ledger — pin it."""
    score, t, y = _synthetic(n=1500, seed=9)
    a = BusinessAssumptions.from_yaml()
    k = 0.3

    point = profit_point(y, score, t, k, a)

    assert point.profit_usd == pytest.approx(profit_at_k(y, score, t, k, a), abs=0.01)
    assert point.incremental_conversions == pytest.approx(
        incremental_conversions_at_k(y, score, t, k), abs=0.01
    )


# ------------------------------------------------------------------- scenarios


def test_compare_scenarios_structure():
    score, t, y = _synthetic()
    comp = compare_scenarios(y, score, t, depth=0.3)

    assert [s.strategy for s in comp.scenarios] == [
        "champion ranking",
        "random targeting (null)",
        "target all",
        "no targeting",
    ]
    champion, random, target_all, nothing = comp.scenarios
    assert champion.feasible and random.feasible and nothing.feasible
    assert nothing.profit_usd == 0.0
    assert champion.n_targeted == random.n_targeted
    assert comp.budget_usd == pytest.approx(champion.n_targeted * 0.1, abs=0.5)
    assert isinstance(comp.champion_wins, bool)


def test_compare_scenarios_champion_beats_null_on_clear_signal():
    score, t, y = _synthetic(n=4000, seed=5, effect=0.30)
    comp = compare_scenarios(y, score, t)
    champion, random, _, _ = comp.scenarios
    assert champion.profit_usd > random.profit_usd
    assert comp.champion_wins


def test_compare_scenarios_target_all_feasibility():
    score, t, y = _synthetic(n=500, seed=1)
    comp = compare_scenarios(y, score, t, budget_usd=1.0)
    target_all = comp.scenarios[2]
    assert not target_all.feasible


# --------------------------------------------------------------------- IPW


def test_policy_value_ipw_null_policy_is_exactly_zero():
    rng = np.random.default_rng(0)
    arm = rng.integers(0, 3, 500)
    y = rng.binomial(1, 0.2, 500).astype(float)
    value = policy_value_ipw(arm, y, np.zeros(500, dtype=int))

    assert value.incremental_rate_vs_no_email == 0.0
    assert value.profit_usd == 0.0
    assert value.n_customers == 500


def test_policy_value_ipw_perfect_policy_is_positive():
    rng = np.random.default_rng(1)
    arm = rng.integers(0, 3, 2000)
    y = rng.binomial(1, 0.1 + 0.15 * (arm > 0), 2000).astype(float)  # emails work
    value = policy_value_ipw(arm, y, arm.copy(), strategy="oracle")

    assert value.incremental_rate_vs_no_email > 0
    assert value.incremental_ci_low < value.incremental_rate_vs_no_email
    assert value.profit_usd > 0


def test_policy_value_ipw_validates_inputs():
    arm = np.array([0, 1, 2, 0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="equal length"):
        policy_value_ipw(arm, y[:3], np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="outside"):
        policy_value_ipw(arm, y, np.array([0, 1, 9, 0]))
    with pytest.raises(ValueError, match="control_arm"):
        policy_value_ipw(np.array([1, 1, 2, 2]), np.zeros(4), np.zeros(4, dtype=int))


# --------------------------------------------------------------------- power


def test_design_ab_test_matches_two_proportion_formula():
    from scipy.stats import norm

    baseline, mde = 0.15, 0.01
    design = design_ab_test(baseline, mde, daily_eligible=1000)

    p1, p2 = baseline, baseline + mde
    p_bar = (p1 + p2) / 2
    expected = np.ceil(
        (
            norm.ppf(0.975) * np.sqrt(2 * p_bar * (1 - p_bar))
            + norm.ppf(0.8) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        ** 2
        / mde**2
    )
    assert design.n_per_arm == int(expected)
    assert design.n_total == 2 * design.n_per_arm
    assert design.duration_days == int(np.ceil(design.n_total / 1000))
    assert design.mde_relative == round(mde / baseline, 4)


def test_design_ab_test_rejects_bad_inputs():
    for kwargs in (
        {"baseline_rate": 1.5, "mde": 0.01},
        {"baseline_rate": 0.15, "mde": -0.01},
        {"baseline_rate": 0.95, "mde": 0.10},
        {"baseline_rate": 0.15, "mde": 0.01, "alpha": 2.0},
        {"baseline_rate": 0.15, "mde": 0.01, "daily_eligible": 0},
    ):
        with pytest.raises(ValueError):
            design_ab_test(**kwargs)


# ----------------------------------------------------------------- model card


def test_get_model_card_reads_registry(tmp_path):
    _register_dummy_champion(tmp_path)
    card = get_model_card("champion", models_dir=tmp_path)

    assert card.learner == "response"
    assert card.target == "visit"
    assert "qini_auc" in card.metrics
    assert card.assumptions.value_per_conversion_usd == 34.91
    assert card.when_not_to_use and card.limitations
    assert "zip_code" in card.fairness_note


def test_get_model_card_unknown_name_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_model_card("ghost", models_dir=tmp_path)


# ------------------------------------------------------------------- segments


@pytest.fixture(scope="module")
def segment_profile():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.10 + t * (X[:, 0] > 0.5) * 0.30).astype(float)
    money = rng.binomial(1, 0.01 + t * (X[:, 0] > 0.5) * 0.02).astype(float)
    groups = np.where(X[:, 1] > 0, "urban", "rural")
    return get_segment_profile(
        X[:3000], t[:3000], y[:3000],
        X[3000:], t[3000:], y[3000:],
        feature_names=["effect_feature", "group_feature", "noise_feature"],
        y_eval_money=money[3000:],
        fairness_col=groups[3000:],
        max_depth=2,
        min_samples_leaf=100,
    )


def test_segment_profile_recovers_planted_effect(segment_profile):
    profile = segment_profile

    assert profile.segments
    assert sum(s.n_eval for s in profile.segments) == 1000
    assert sum(s.share for s in profile.segments) == pytest.approx(1.0, abs=1e-6)
    uplinks = [s.uplift_visit for s in profile.segments]
    assert max(uplinks) > 0.15  # the planted +0.30 segment must surface
    assert min(uplinks) < 0.05
    for s in profile.segments:
        assert s.uplift_visit_ci_low <= s.uplift_visit <= s.uplift_visit_ci_high
        assert s.uplift_conversion is not None  # money column requested


def test_segment_profile_rules_are_readable(segment_profile):
    profile = segment_profile

    flat = [r for s in profile.segments for r in s.rules]
    assert flat, "expected at least one split rule"
    assert all("<=" in r or ">" in r for r in flat)
    assert any(r.startswith("effect_feature") for r in flat)  # planted split found


def test_segment_profile_fairness_rows(segment_profile):
    profile = segment_profile

    assert {f.group for f in profile.fairness} == {"urban", "rural"}
    assert sum(f.n_eval for f in profile.fairness) == 1000
    for f in profile.fairness:
        assert 0.0 <= f.targeting_rate <= 1.0
        assert f.uplift_visit_ci_low <= f.uplift_visit <= f.uplift_visit_ci_high
