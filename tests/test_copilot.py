"""Tests for the Day-5 copilot pipeline: router, planner, validator, writer.

Everything runs offline: the gate is a fake decisions client, the writer is
the template or a scripted fake, and the holdout context is synthetic. The
identical code path ``run_request`` runs live is exercised end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.causal.benchmark import BusinessAssumptions
from packages.copilot import run_request
from packages.copilot.planner import (
    HoldoutContext,
    extract_scenario,
    run_tools,
)
from packages.copilot.providers import ProviderError, WriterBase
from packages.copilot.router import (
    build_questions,
    decide_fast_path,
    keyword_fallback_route,
    parse_route,
    route_question,
)
from packages.copilot.schemas import (
    BRIEF_SECTIONS,
    CopilotConfig,
    DraftRequest,
    DraftResult,
    Intent,
    load_config,
)
from packages.copilot.validator import extract_cited, validate
from packages.copilot.writer import UNVERIFIED_MARKER, get_writer, write_brief

# --- fixtures and fakes ----------------------------------------------------


def _answers(intent="plan", conf=0.9, risk=0.02, miss=0.1):
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


class FakeDecisions:
    """Duck-typed DecisionsClient returning canned answers."""

    def __init__(self, answers):
        self.answers = answers

    def decide(self, state, questions):
        return {"answers": self.answers}


class ScriptedWriter(WriterBase):
    """Returns each scripted text in order (last one repeats)."""

    name = "scripted"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def write(self, request: DraftRequest) -> DraftResult:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return DraftResult(text=text, provider=self.name, latency_s=0.0)


@pytest.fixture
def config() -> CopilotConfig:
    cfg = load_config()
    cfg.trace.enabled = False
    return cfg


@pytest.fixture(scope="module")
def context() -> HoldoutContext:
    """Synthetic holdout with planted uplift on x0 (same recipe as decision tests)."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 4))
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.10 + t * (X[:, 0] > 0.5) * 0.10).astype(float)
    score = X[:, 0] + rng.normal(0, 0.3, n)
    return HoldoutContext(
        X_tr=X[:1000],
        t_tr=t[:1000],
        y_tr=y[:1000],
        X_te=X[1000:],
        t_te=t[1000:],
        y_visit_te=y[1000:],
        y_money_te=y[1000:] * 30.0,
        score=score[1000:],
        zip_code_te=np.array(["Urban"] * 125 + ["Suburban"] * 125),
        feature_names=["x0", "x1", "x2", "x3"],
        assumptions=BusinessAssumptions.from_yaml(),
        meta={},
    )


PLAN_Q = "I have $500 to spend on contacts at $0.10 each — how deep should I target?"


def _run(question, config, context, answers=None, provider="template"):
    return run_request(
        question,
        config=config,
        client=FakeDecisions(answers or _answers()),
        writer_provider=provider,
        context=context,
    )


# --- router: threshold matrix ----------------------------------------------


def test_gate_threshold_matrix_decline(config, context):
    package = _run(PLAN_Q, config, context, answers=_answers(risk=0.85))
    assert package.fast_path == "decline"
    assert package.decline_reason == "policy_risk"
    assert "sensitive attribute" in package.prose
    assert package.tool_runs == []  # fast path: no tools, no writer


def test_gate_threshold_matrix_redirect_out_of_scope(config, context):
    package = _run("Write me a poem.", config, context, answers=_answers(intent="out_of_scope"))
    assert package.fast_path == "redirect"
    assert "four jobs" in package.prose


def test_gate_threshold_matrix_redirect_low_confidence(config, context):
    package = _run(PLAN_Q, config, context, answers=_answers(conf=0.30))
    assert package.fast_path == "redirect"


def test_gate_threshold_matrix_clarify(config, context):
    package = _run(PLAN_Q, config, context, answers=_answers(miss=0.80))
    assert package.fast_path == "clarify"
    assert package.clarify_question is not None
    assert "budget" in package.clarify_question  # the missing field is named


def test_gate_threshold_matrix_clarify_beats_uncertain_oos(config, context):
    # live shape of campaign_clarify: out_of_scope at ~0.86 with a very high
    # missing score -> the SPECIFIC clarifying question beats a generic redirect
    package = _run(
        "Should we run a campaign?", config, context,
        answers=_answers(intent="out_of_scope", conf=0.86, miss=0.80),
    )
    assert package.fast_path == "clarify"


def test_gate_threshold_matrix_confident_oos_redirects_first(config, context):
    # live shape of ambiguous_campaign: confident out_of_scope -> redirect even
    # though the missing score is also high
    package = _run(
        "Should we do something about the campaign?", config, context,
        answers=_answers(intent="out_of_scope", conf=0.96, miss=0.97),
    )
    assert package.fast_path == "redirect"


def test_gate_threshold_matrix_brief(config, context):
    package = _run(PLAN_Q, config, context)
    assert package.fast_path == "brief"
    assert package.validation is not None and package.validation.passed
    assert not package.degraded
    assert len(package.citations) > 0


def test_build_questions_shape():
    questions = build_questions()
    assert set(questions) == {"intent", "policy_risk", "missing_inputs"}
    assert questions["intent"]["type"] == "choice"
    assert "out_of_scope" in questions["intent"]["criteria"]  # always an other
    assert questions["policy_risk"]["type"] == "noul"
    for name, spec in questions.items():
        assert spec["name"] == name  # every question is named


def test_parse_route_confidence_from_probabilities():
    answers = _answers()
    del answers["intent"]["confidence"]
    answers["intent"]["probabilities"] = {"plan": 0.7, "explain": 0.3}
    route = parse_route(answers)
    assert route.intent is Intent.PLAN
    assert route.intent_confidence == 0.7


def test_parse_route_refusal_raises():
    answers = _answers()
    answers["intent"] = {"type": "refusal"}
    with pytest.raises(ProviderError, match="refused"):
        parse_route(answers)


def test_route_question_falls_back_on_provider_error():
    class Boom:
        def decide(self, state, questions):
            raise ProviderError("decisions", "connection refused")

    route = route_question("What if contact cost rises to $0.60?", Boom())
    assert route.router == "keyword_fallback"
    assert route.intent is Intent.WHAT_IF


# --- keyword fallback (T6) -------------------------------------------------


@pytest.mark.parametrize(
    "question,intent",
    [
        ("How deep should we target with a $500 budget?", Intent.PLAN),
        ("What if contact cost rises to $0.60?", Intent.WHAT_IF),
        ("Who are the persuadable segments?", Intent.EXPLAIN),
        ("How long should the experiment run to detect a lift?", Intent.TEST_DESIGN),
    ],
)
def test_keyword_fallback_routes_use_cases(question, intent):
    route = keyword_fallback_route(question)
    assert route.router == "keyword_fallback"
    assert route.intent is intent


def test_keyword_fallback_flags_policy_keywords():
    route = keyword_fallback_route("Only target women aged 25-34.")
    assert route.policy_risk >= 0.7


def test_keyword_fallback_no_keywords_is_out_of_scope():
    route = keyword_fallback_route("hello there")
    assert route.intent is Intent.OUT_OF_SCOPE
    assert decide_fast_path(route, load_config()) == "redirect"


# --- scenario parsing ------------------------------------------------------


def test_extract_scenario_budget_and_cost(config):
    scenario = extract_scenario(PLAN_Q, Intent.PLAN, config)
    assert scenario.budget_usd == 500.0
    assert scenario.cost_per_contact_usd == 0.10
    assert scenario.missing_fields == []


def test_extract_scenario_k_suffix_and_defaults(config):
    scenario = extract_scenario("Spend 500k on the push", Intent.PLAN, config)
    assert scenario.budget_usd == 500_000.0
    assert scenario.cost_per_contact_usd == pytest.approx(0.10)
    assert any("defaulted" in a for a in scenario.assumptions_used)
    assert scenario.missing_fields == []


def test_extract_scenario_missing_budget_flagged(config):
    scenario = extract_scenario("Who should we target?", Intent.PLAN, config)
    assert scenario.missing_fields == ["budget"]
    assert scenario.depth == pytest.approx(0.30)
    assert any("depth defaulted" in a for a in scenario.assumptions_used)


def test_extract_scenario_whatif_cost_not_budget(config):
    scenario = extract_scenario("What if contact cost rises to $0.60?", Intent.WHAT_IF, config)
    assert scenario.cost_per_contact_usd == 0.60
    assert scenario.budget_usd is None


def test_extract_scenario_test_design(config):
    scenario = extract_scenario(
        "How many per arm to detect a 1 percentage point lift? We see 2,000 customers a day.",
        Intent.TEST_DESIGN,
        config,
    )
    assert scenario.mde == pytest.approx(0.01)
    assert scenario.daily_eligible == 2000


def test_extract_scenario_ignores_coupon_percent(config):
    scenario = extract_scenario("How long to test a new 15% coupon?", Intent.TEST_DESIGN, config)
    assert scenario.mde == pytest.approx(config.planner.default_mde)


# --- planner: tool dispatch ------------------------------------------------


def test_run_tools_plan_dispatch(config, context):
    scenario = extract_scenario(PLAN_Q, Intent.PLAN, config)
    runs = {r.name: r for r in run_tools(scenario, context, config)}
    assert set(runs) == {
        "profit_point",
        "optimize_budget",
        "compare_scenarios",
        "design_ab_test",
        "get_model_card",
    }
    assert all(r.ok for r in runs.values())
    # budget $500 at $0.10 -> 5,000 contacts -> capped at max_depth 0.5
    assert runs["profit_point"].output["depth"] == pytest.approx(0.5)


def test_run_tools_budget_derives_depth(config, context):
    scenario = extract_scenario("$200 budget, how deep?", Intent.PLAN, config)
    runs = {r.name: r for r in run_tools(scenario, context, config)}
    # $200 / $0.10 = 2,000 contacts of 500 test rows -> capped
    assert runs["profit_point"].output["n_targeted"] <= context.n_test


def test_run_tools_explain_fits_and_caches(config, context):
    scenario = extract_scenario("Who are the persuadables?", Intent.EXPLAIN, config)
    first = {r.name: r for r in run_tools(scenario, context, config)}
    assert set(first) == {"get_segment_profile"}
    assert first["get_segment_profile"].ok
    assert context._profile is not None
    second = run_tools(scenario, context, config)
    assert second[0].output == first["get_segment_profile"].output  # cached, not refit


def test_run_tools_what_if_delta(config, context):
    scenario = extract_scenario("What if contact cost rises to $0.60?", Intent.WHAT_IF, config)
    runs = {r.name: r for r in run_tools(scenario, context, config)}
    assert {"compare_scenarios_base", "compare_scenarios_variant", "what_if_delta"} <= set(runs)
    delta = runs["what_if_delta"].output
    assert "cost per contact" in delta["changed"]
    assert delta["profit_delta_usd"] <= 0  # a cost rise can only lose money here


def test_run_tools_surfaces_tool_failure(config, context, monkeypatch):
    import packages.copilot.planner as planner_module

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(planner_module, "optimize_budget", boom)
    scenario = extract_scenario(PLAN_Q, Intent.PLAN, config)
    runs = {r.name: r for r in run_tools(scenario, context, config)}
    assert not runs["optimize_budget"].ok
    assert "synthetic failure" in runs["optimize_budget"].error
    assert runs["profit_point"].ok  # one failure never kills the request


# --- validator (V1–V4) -----------------------------------------------------


def _tool_runs(config, context):
    scenario = extract_scenario(PLAN_Q, Intent.PLAN, config)
    return scenario, run_tools(scenario, context, config)


def test_validator_catches_invented_number(config, context):
    scenario, runs = _tool_runs(config, context)
    report = validate(
        "## Recommendation\nTarget 40%.\n## What it earns\n$9,999 profit.\n"
        "## Uncertainties\n[x]\n## Assumptions\n- x\n## Next test\n- x",
        runs,
        scenario,
        config.validator,
    )
    v1 = next(c for c in report.checks if c.rule == "V1_number_trace")
    assert not v1.passed
    assert "9999" in v1.detail or "9,999" in v1.detail


def test_validator_catches_foreign_currency(config, context):
    scenario, runs = _tool_runs(config, context)
    prose = (
        "## Recommendation\nSpend ₹500.\n## What it earns\n[x]\n"
        "## Uncertainties\n[x]\n## Assumptions\n- x\n## Next test\n- x"
    )
    report = validate(prose, runs, scenario, config.validator)
    v2 = next(c for c in report.checks if c.rule == "V2_units")
    assert not v2.passed


def test_validator_catches_causal_overclaim(config, context):
    scenario, runs = _tool_runs(config, context)
    prose = (
        "## Recommendation\nThese customers will buy.\n## What it earns\n[x]\n"
        "## Uncertainties\n[x]\n## Assumptions\n- x\n## Next test\n- x"
    )
    report = validate(prose, runs, scenario, config.validator)
    v3 = next(c for c in report.checks if c.rule == "V3_causal_language")
    assert not v3.passed


def test_validator_catches_missing_section(config, context):
    scenario, runs = _tool_runs(config, context)
    prose = "## Recommendation\n[x]\n## What it earns\n[x]\n## Uncertainties\n[x]\n"
    report = validate(prose, runs, scenario, config.validator)
    v4 = next(c for c in report.checks if c.rule == "V4_sections")
    assert not v4.passed
    assert "assumptions" in v4.detail


def test_validator_percent_transform_matches_fraction():
    assert extract_cited("depth of 30%")[0][0] == pytest.approx(30.0)
    assert extract_cited("uplift +0.0114")[0][0] == pytest.approx(0.0114)
    assert extract_cited("profit -$155.64")[0][0] == pytest.approx(-155.64)
    assert extract_cited("4,262 contacts")[0][0] == pytest.approx(4262)


def test_validator_typographic_forms_match():
    # live gpt-oss drafts use U+2011 hyphen as a minus and U+202F digit
    # grouping ("2 443") — same tool numbers, different typography
    assert extract_cited("\u2011$155.64")[0][0] == pytest.approx(-155.64)
    assert extract_cited("2\u202f443 contacts")[0][0] == pytest.approx(2443.0)
    assert extract_cited("\u201340.8% relative")[0][0] == pytest.approx(-40.8)
    assert extract_cited("\u2212$5.00")[0][0] == pytest.approx(-5.0)


def test_validator_citations_carry_paths(config, context):
    scenario, runs = _tool_runs(config, context)
    from packages.copilot.providers import TemplateWriter

    draft = TemplateWriter().write(
        DraftRequest(
            intent=Intent.PLAN, question=PLAN_Q, scenario=scenario, tool_runs=runs
        )
    )
    report = validate(draft.text, runs, scenario, config.validator)
    assert report.passed
    sources = {c.source for c in report.citations}
    assert any(s.startswith("profit_point.") for s in sources)


# --- writer: template, retry loop, degradation ------------------------------


def test_template_writer_passes_battery_for_all_intents(config, context):
    from packages.copilot.providers import TemplateWriter

    for intent, question in [
        (Intent.PLAN, PLAN_Q),
        (Intent.WHAT_IF, "What if contact cost rises to $0.60?"),
        (Intent.EXPLAIN, "Who are the persuadables?"),
        (Intent.TEST_DESIGN, "How long to test? We see 2,000 customers a day."),
    ]:
        scenario = extract_scenario(question, intent, config)
        runs = run_tools(scenario, context, config)
        draft = TemplateWriter().write(
            DraftRequest(intent=intent, question=question, scenario=scenario, tool_runs=runs)
        )
        report = validate(draft.text, runs, scenario, config.validator)
        assert report.passed, f"{intent}: {[c.detail for c in report.failures]}"


def test_write_brief_retry_succeeds_second_draft(config, context):
    scenario, runs = _tool_runs(config, context)
    request = DraftRequest(
        intent=Intent.PLAN, question=PLAN_Q, scenario=scenario, tool_runs=runs
    )
    from packages.copilot.providers import TemplateWriter

    good = TemplateWriter().write(request).text
    writer = ScriptedWriter(["## Recommendation\n$9,999,999 profit!\n", good])
    final, report, attempts, degraded = write_brief(request, writer, config)
    assert writer.calls == 2  # first draft failed, rewrite fixed it
    assert report.passed and not degraded
    assert len(attempts) == 2


def test_write_brief_second_failure_degrades_with_unverified(config, context):
    scenario, runs = _tool_runs(config, context)
    request = DraftRequest(
        intent=Intent.PLAN, question=PLAN_Q, scenario=scenario, tool_runs=runs
    )
    writer = ScriptedWriter(["## Recommendation\n$9,999,999 profit!\n"])
    final, report, attempts, degraded = write_brief(request, writer, config)
    assert writer.calls == 1 + config.validator.max_rewrites  # retry budget respected
    assert degraded and not report.passed
    assert UNVERIFIED_MARKER in final.text
    assert final.text.count(UNVERIFIED_MARKER) == len(report.failures)


def test_write_brief_provider_failure_degrades_to_template(config, context):
    scenario, runs = _tool_runs(config, context)
    request = DraftRequest(
        intent=Intent.PLAN, question=PLAN_Q, scenario=scenario, tool_runs=runs
    )

    class ExplodingWriter(WriterBase):
        name = "exploding"

        def write(self, request):
            raise ProviderError("exploding", "gateway timeout")

    final, report, attempts, degraded = write_brief(request, ExplodingWriter(), config)
    assert "->template" in final.provider
    assert report.passed and not degraded  # template output is validator-clean


def test_get_writer_selection(config, monkeypatch):
    assert get_writer(config, "template").name == "template"
    assert get_writer(config, "ollama").name == "ollama"
    monkeypatch.setenv("NEUROMETRIC_BASE_URL", "https://gw/v1")
    monkeypatch.setenv("NEUROMETRIC_API", "tok")
    assert get_writer(config, "neurometric").name == "neurometric"
    with pytest.raises(ValueError, match="unknown writer provider"):
        get_writer(config, "gpt4-owo")
    monkeypatch.delenv("NEUROMETRIC_BASE_URL", raising=False)
    monkeypatch.delenv("NEUROMETRIC_API", raising=False)
    with pytest.raises(ProviderError, match="missing environment"):
        get_writer(config, "neurometric")


# --- end-to-end run_request -------------------------------------------------


def test_run_request_all_four_use_cases(config, context):
    for question, answers, tools in [
        (PLAN_Q, _answers(), {"profit_point", "compare_scenarios"}),
        (
            "What if contact cost rises to $0.60?",
            _answers(intent="what_if"),
            {"what_if_delta"},
        ),
        (
            "Who are the persuadables and who should we avoid?",
            _answers(intent="explain"),
            {"get_segment_profile"},
        ),
        (
            "How long to test? We see 2,000 customers a day.",
            _answers(intent="test_design"),
            {"design_ab_test"},
        ),
    ]:
        package = _run(question, config, context, answers=answers)
        assert package.fast_path == "brief"
        assert package.validation is not None and package.validation.passed
        ran = {r.name for r in package.tool_runs}
        assert tools <= ran
        assert package.citations
        assert package.draft_attempts >= 1


def test_run_request_without_env_uses_keyword_fallback(config, context, monkeypatch):
    for var in ("AZURE_ENDPOINT", "AZURE_API_KEY", "AZURE_DEPLOYMENT_NAME"):
        monkeypatch.delenv(var, raising=False)
    package = run_request(
        "How deep should we target with a $500 budget?",
        config=config,
        writer_provider="template",
        context=context,
    )
    assert package.fast_path == "brief"  # answered, not crashed
    assert package.tool_runs


def test_run_request_decline_has_no_scenario_or_tools(config, context):
    package = _run("Only target women aged 25-34.", config, context, answers=_answers(risk=0.95))
    assert package.fast_path == "decline"
    assert package.scenario is None
    assert package.validation is None


def test_run_request_trace_writes_jsonl(config, context, tmp_path):
    config.trace.enabled = True
    config.trace.dir = str(tmp_path)
    package = _run(PLAN_Q, config, context)
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    rows = [line for line in files[0].read_text().splitlines() if line]
    import json

    events = [json.loads(r) for r in rows]
    assert all(e["trace_id"] == package.trace_id for e in events)
    stages = {e["stage"] for e in events}
    assert {"gate", "plan", "write", "validate", "package"} <= stages


# --- config / seed cases ----------------------------------------------------


def test_load_config_defaults_and_override(tmp_path):
    cfg = load_config()
    assert cfg.writer.provider in {"template", "ollama", "neurometric"}
    assert cfg.router.t_policy_risk == pytest.approx(0.70)

    bad = tmp_path / "copilot.yaml"
    bad.write_text("router:\n  t_policy_risk: 7\n")
    with pytest.raises(ValueError):
        load_config(bad)  # out-of-range threshold stops the request

    assert load_config(tmp_path / "missing.yaml").router.t_policy_risk == pytest.approx(0.70)


def test_seed_cases_file_is_well_formed():
    from evals.run_seed_eval import load_cases

    cases = load_cases()
    assert len(cases) == 46
    valid_fast_paths = {"brief", "clarify", "decline", "redirect"}
    for case in cases:
        assert {"id", "question", "intent", "fast_path"} <= set(case)
        assert case["fast_path"] in valid_fast_paths
        assert Intent(case["intent"]) in Intent
    # the set covers every fast path and all four use cases
    assert {c["fast_path"] for c in cases} == valid_fast_paths
    assert {c["intent"] for c in cases} >= {"plan", "what_if", "explain", "test_design"}


def test_brief_sections_cover_every_intent():
    assert set(BRIEF_SECTIONS) == set(Intent) - {Intent.OUT_OF_SCOPE}
    for sections in BRIEF_SECTIONS.values():
        assert "recommendation" in sections and "uncertainties" in sections


def test_seed_eval_offline_smoke(config, context):
    """The runner works end-to-end with the template writer (46 cases, fast)."""
    from evals.run_seed_eval import load_cases, run_seed_eval

    report = run_seed_eval(provider="template", cases=load_cases(), context=context)
    assert report["summary"]["n_cases"] == 46
    # the gate (live when AZURE_* is present, keyword fallback otherwise)
    # answers at least 80% of cases on the right path — documented quality
    assert report["summary"]["fast_path_accuracy"] >= 0.8
