"""Tests for the model-semantics validator rule (V6).

V6 is the guardrail that keeps a response-probability score from being
narrated as a causal effect and vice versa — the one mix-up that silently
turns marketing arithmetic into pseudo-causality. The other rules (V1–V4)
are exercised in test_copilot.py; here only the V6 verdict matters, so the
prose fixtures deliberately violate the shape rules freely.
"""

from __future__ import annotations

import pytest

from packages.causal import catalog
from packages.copilot.schemas import (
    CampaignScenario,
    Intent,
    load_config,
)
from packages.copilot.validator import validate

_VALIDATOR = load_config().validator
_SCENARIO = CampaignScenario(intent=Intent.PLAN, question="what if contact cost doubles?")


def _v6(prose: str, score_kind: str | None):
    report = validate(
        prose,
        tool_runs=[],
        scenario=_SCENARIO,
        config=_VALIDATOR,
        score_kind=score_kind,
    )
    return next(c for c in report.checks if c.rule == "V6_model_semantics")


# --- a response/relative score must not be narrated as an effect -------------


@pytest.mark.parametrize(
    "prose",
    [
        "Top-decile customers have a causal effect of 0.09 on visits.",
        "True uplift for this segment is what our score reports.",
        "The incremental effect of the email is 9 percent.",
        "Contacting these persuadables produces a causal effect on visits.",
    ],
)
def test_v6_blocks_effect_language_for_response_score(prose):
    check = _v6(prose, catalog.SCORE_RESPONSE)
    assert not check.passed
    assert "must not be narrated" in check.detail


@pytest.mark.parametrize(
    "prose",
    [
        "The score is the probability that a contacted customer visits.",
        "Contact likelihood per customer, ranked.",
    ],
    ids=["plain", "ranking-wording"],
)
def test_v6_allows_response_language_for_response_score(prose):
    assert _v6(prose, catalog.SCORE_RESPONSE).passed


# --- an effect score must not be narrated as a probability -------------------


@pytest.mark.parametrize(
    "prose",
    [
        "Customers here have a 12% probability of visiting; rank them first.",
        "The likelihood of conversion under this plan is highest at k=30%.",
        "Likely to visit: this segment scores above everyone.",
    ],
)
def test_v6_blocks_probability_language_for_effect_score(prose):
    check = _v6(prose, catalog.SCORE_UPLIFT_EFFECT)
    assert not check.passed


@pytest.mark.parametrize(
    "prose",
    [
        "The estimated incremental effect per contacted customer.",
        "Estimated uplift in visit rate between treated and control.",
    ],
    ids=["incremental", "uplift-wording"],
)
def test_v6_allows_effect_language_for_effect_score(prose):
    assert _v6(prose, catalog.SCORE_UPLIFT_EFFECT).passed


# --- explicit contrast is always honest --------------------------------------


@pytest.mark.parametrize("score_kind", list(catalog.SCORE_MEANING))
def test_v6_allows_explicit_contrast(score_kind):
    prose = (
        "The score is an effect estimate, not a probability; equivalently it is "
        "a probability-like quantity, not an effect. Rather than a probability, "
        "read it as a ranking."
    )
    check = _v6(prose, score_kind)
    assert check.passed


# --- the rule is off when the draft carries no score kind ---------------------


def test_v6_absent_without_score_kind():
    report = validate("anything", tool_runs=[], scenario=_SCENARIO, config=_VALIDATOR)
    rules = [c.rule for c in report.checks]
    assert "V6_model_semantics" not in rules
