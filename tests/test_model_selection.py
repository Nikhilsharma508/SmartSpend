"""Tests for the deterministic model resolver (packages/copilot/selection.py).

Nothing here touches the network or a generative model: the resolver turns a
pinned model + the gate's ``model`` answer into exactly one registered artifact
under rules that must be provable, not narrated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.causal import catalog
from packages.copilot.router import parse_route
from packages.copilot.schemas import CopilotConfig, Intent, load_config
from packages.copilot.selection import (
    ModelSelection,
    assert_valid_model_name,
    resolve_model,
    trace_fields,
)


def _answers(intent="plan", conf=0.9, risk=0.02, miss=0.1):
    """Same shape the fake Decisions client returns (mirrors test_copilot)."""
    return {
        "intent": {
            "type": "choice",
            "choice": intent,
            "confidence": conf,
            "probabilities": {intent: conf},
        },
        "policy_risk": {"type": "noul", "noul": risk},
        "missing_inputs": {"type": "noul", "noul": miss},
    }


@pytest.fixture
def config() -> CopilotConfig:
    return load_config()


def _plan_route(answer: str = "dr-learner", conf: float = 0.9):
    answers = _answers(intent="plan", conf=0.9)
    answers["model"] = {"type": "choice", "choice": answer, "confidence": conf}
    return parse_route(answers)


def _registered(monkeypatch, names: list[str], tmp_path: Path | None = None) -> dict:
    """Point selection.is_registered at a canned name list."""
    seen: dict[str, Path | None] = {}

    def fake(name: str, models_dir=None):
        seen[name] = models_dir
        return name in names

    from packages.copilot import selection

    monkeypatch.setattr(selection, "is_registered", fake)
    return seen


# --- manual pins ------------------------------------------------------------


def test_manual_pin_wins_over_gate_and_mode_error(config):
    route = _plan_route("dr-learner", conf=0.9)
    sel = resolve_model(route, "champion", config=config)
    assert isinstance(sel, ModelSelection)
    assert sel.name == "champion" and sel.selector == "manual"
    assert sel.confidence is None  # manual is a decision, not a probability


def test_pin_unknown_model_lists_catalog(config):
    with pytest.raises(ValueError, match="unknown model 'x-forest'"):
        resolve_model(_plan_route(), "x-forest", config=config)


def test_pin_research_only_without_mode_raises(config, monkeypatch):
    _registered(monkeypatch, ["champion", "t-learner"])
    with pytest.raises(ValueError, match="Research Mode"):
        resolve_model(_plan_route(), "t-learner", research_mode=False, config=config)


def test_pin_research_only_with_mode_ok(config, monkeypatch):
    _registered(monkeypatch, ["t-learner"])
    sel = resolve_model(_plan_route(), "t-learner", research_mode=True, config=config)
    assert sel.name == "t-learner" and sel.research_only and sel.research_mode


def test_pin_unregistered_artifact_raises(config, monkeypatch):
    _registered(monkeypatch, ["champion"])  # dr-learner NOT registered
    with pytest.raises(ValueError, match="not registered"):
        resolve_model(_plan_route(), "dr-learner", config=config)


def test_assert_valid_model_name_fail_fast(config, monkeypatch):
    _registered(monkeypatch, ["champion", "dr-learner"])
    assert_valid_model_name(None, False)  # no pin is always fine
    assert_valid_model_name("dr-learner", False)  # challenger is fine
    with pytest.raises(ValueError):
        assert_valid_model_name("s-learner", False)
    with pytest.raises(ValueError):
        assert_valid_model_name("dr-learner-model-typo", False)


# --- gate choices -----------------------------------------------------------


def test_confident_gate_choice_runs_that_model(config, monkeypatch):
    seen = _registered(monkeypatch, ["champion", "dr-learner"])
    sel = resolve_model(_plan_route("dr-learner", 0.83), None, config=config)
    assert sel.name == "dr-learner"
    assert sel.selector == "gate"
    assert sel.confidence == 0.83
    assert "dr-learner" in seen


def test_gate_choice_below_threshold_falls_back(config, monkeypatch):
    _registered(monkeypatch, ["champion", "dr-learner"])
    sel = resolve_model(_plan_route("dr-learner", 0.42), None, config=config)
    assert sel.name == catalog.DEFAULT_MODEL
    assert sel.selector == "fallback"
    assert sel.confidence == 0.42


def test_gate_choice_missing_confidence_falls_back(config, monkeypatch):
    answers = _answers(intent="plan")
    answers["model"] = {"type": "choice", "choice": "dr-learner"}
    sel = resolve_model(parse_route(answers), None, config=config)
    assert sel.name == catalog.DEFAULT_MODEL and sel.selector == "fallback"


def test_gate_cannot_pick_research_model(config, monkeypatch):
    """The allow-list — not the prompt — is what keeps weak models out."""
    _registered(monkeypatch, ["t-learner"])
    sel = resolve_model(_plan_route("t-learner", 0.99), None, config=config)
    assert sel.name == catalog.DEFAULT_MODEL and sel.selector == "fallback"


def test_gate_choice_not_registered_falls_back(config, monkeypatch):
    _registered(monkeypatch, ["champion"])  # dr-learner artifact absent
    sel = resolve_model(_plan_route("dr-learner", 0.9), None, config=config)
    assert sel.name == catalog.DEFAULT_MODEL and sel.selector == "fallback"


# --- non-model-sensitive intents -------------------------------------------


def test_explain_and_test_design_default_to_champion(config, monkeypatch):
    _registered(monkeypatch, ["dr-learner"])
    for intent in (Intent.EXPLAIN, Intent.TEST_DESIGN):
        answers = _answers(intent=intent.value, conf=0.95)
        answers["model"] = {"type": "choice", "choice": "dr-learner", "confidence": 0.9}
        route = parse_route(answers)
        sel = resolve_model(route, None, config=config)
        assert sel.name == catalog.DEFAULT_MODEL
        assert sel.selector == "default"
        assert sel.confidence is None


def test_no_answer_falls_back_to_champion(config, monkeypatch):
    _registered(monkeypatch, [])
    sel = resolve_model(parse_route(_answers("plan")), None, config=config)
    assert sel.name == catalog.DEFAULT_MODEL and sel.selector == "fallback"


# --- trace fields ------------------------------------------------------------


def test_trace_fields_shape(config):
    sel = resolve_model(_plan_route("dr-learner", 0.9), None, config=config)
    fields = trace_fields(sel)
    assert fields["ml.model"] == "dr-learner"
    assert fields["ml.score_kind"] == "uplift_effect"
    assert fields["model.selector"] == "gate"
    assert fields["research.mode"] is False
