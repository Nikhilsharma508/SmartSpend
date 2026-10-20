"""The [2] PLAN stage — deterministic intent → Scenario → Day-4 tools.

Pure Python on purpose: money is parsed out of the question with rules (never
an LLM), every default is documented and flagged in ``assumptions_used``, and
the requested artifact is loaded **once per process** and cached — the heavy
work (model load, split rebuild, policy-tree fit) happens at first use, not
per request. The split data and the policy tree are shared across models; only
the scores are per artifact, so serving a second model refits nothing.

Every tool's output — or its failure — becomes a :class:`ToolRun` the writer
can quote; a failed tool can never be papered over with an invented number.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from packages.causal import catalog
from packages.causal.benchmark import BusinessAssumptions
from packages.causal.holdout import build_holdout
from packages.causal.registry import load_model
from packages.copilot.providers import tool_run, trim_tool_output
from packages.copilot.schemas import CampaignScenario, CopilotConfig, Intent, ToolRun
from packages.decision import (
    compare_scenarios,
    design_ab_test,
    get_model_card,
    get_segment_profile,
    optimize_budget,
    profit_point,
)
from packages.decision.schemas import ModelCard, SegmentProfile
from packages.tracking import set_span_outputs, trace_span

# --- question parsing ------------------------------------------------------
# USD only (decided Day 1): $500, $2,000.50, 500k, "$200 USD". No INR parsing.

_MONEY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?(k)?\b", re.IGNORECASE)
_USD_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s?(k)?\s?(?:usd|dollars?)\b", re.IGNORECASE)
#: Bare "500k" with no currency marker — only used when no $/USD amount exists,
#: so "at $0.10 each" never competes with it.
_K_SUFFIX_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s?(k)\b", re.IGNORECASE)
_BUDGET_WORDS = re.compile(r"budget|spend|allocate|pot\b", re.IGNORECASE)
_COST_WORDS = re.compile(r"cost|price|cpc|per\s*contact|per\s*email|per\s*send", re.IGNORECASE)
_PER_UNIT_RE = re.compile(
    r"(?:per|each|a|/)\s*(?:contact|email|send|customer)s?\b|\beach\b", re.IGNORECASE
)
_DEPTH_WORDS = re.compile(r"top\b|depth|richest|highest[- ]scoring", re.IGNORECASE)
_MDE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:percentage[\s-]?points?|pp)\b", re.IGNORECASE)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_MDE_WORDS = re.compile(r"detect|mde|lift|improvement|uplift", re.IGNORECASE)
_DAILY_RE = re.compile(
    r"(\d[\d,]*)\s*(?:customers?|people|users?)\s*(?:a|per)\s*day"
    r"|(\d[\d,]*)\s*/\s*day",
    re.IGNORECASE,
)


def _amounts(question: str) -> list[tuple[float, tuple[int, int]]]:
    """Every USD amount in order of appearance, with its span."""
    found: list[tuple[float, tuple[int, int]]] = []
    for match in _MONEY_RE.finditer(question):
        value = float(match.group(1).replace(",", "")) * (1000.0 if match.group(2) else 1.0)
        found.append((value, match.span()))
    claimed = {span for _, span in found}
    for match in _USD_RE.finditer(question):
        span = match.span()
        if any(s[0] <= span[0] < s[1] for s in claimed):
            continue  # already captured as a $-amount
        value = float(match.group(1).replace(",", "")) * (1000.0 if match.group(2) else 1.0)
        found.append((value, span))
    if not found:
        for match in _K_SUFFIX_RE.finditer(question):
            value = float(match.group(1).replace(",", "")) * 1000.0
            found.append((value, match.span()))
    return sorted(found, key=lambda a: a[1][0])


def _near(
    question: str, span: tuple[int, int], patterns: list[re.Pattern[str]], window: int
) -> bool:
    """True when one of ``patterns`` sits within ``window`` chars of ``span``."""
    start, end = span
    for pattern in patterns:
        for m in pattern.finditer(question):
            if end <= m.start() and m.start() - end <= window:
                return True
            if m.end() <= start and start - m.end() <= window:
                return True
    return False


def extract_scenario(
    question: str, intent: Intent, config: CopilotConfig
) -> CampaignScenario:
    """Rule-based scenario fill: stated values in, documented defaults flagged."""
    p = config.planner
    used: list[str] = []
    missing: list[str] = []
    context_patterns = [_COST_WORDS, _PER_UNIT_RE]

    budget: float | None = None
    cost: float | None = None
    if intent in (Intent.PLAN, Intent.WHAT_IF):
        for value, span in _amounts(question):
            if _near(question, span, context_patterns, 20) and value < 100:
                if value > 0:
                    cost = cost if cost is not None else value
            elif budget is None and value > 0:
                budget = value

    depth: float | None = None
    for match in _PCT_RE.finditer(question):
        if _near(question, match.span(), [_DEPTH_WORDS], 12):
            depth = float(match.group(1)) / 100.0
            break

    mde: float | None = None
    mde_match = _MDE_RE.search(question)
    if mde_match:
        mde = float(mde_match.group(1)) / 100.0
    else:
        for match in _PCT_RE.finditer(question):
            if _near(question, match.span(), [_MDE_WORDS], 20):
                mde = float(match.group(1)) / 100.0
                break

    daily: int | None = None
    daily_match = _DAILY_RE.search(question)
    if daily_match:
        daily = int((daily_match.group(1) or daily_match.group(2)).replace(",", ""))

    # Documented defaults — every one flagged, never silent.
    if intent in (Intent.PLAN, Intent.WHAT_IF):
        if cost is None:
            cost = BusinessAssumptions.from_yaml().cost_per_contact_usd
            used.append(
                f"cost per contact defaulted to ${cost:.2f} (configs/business_assumptions.yaml)"
            )
        if intent is Intent.PLAN and budget is None:
            missing.append("budget")
        if depth is None and budget is None:
            depth = p.default_depth
            used.append(f"targeting depth defaulted to {depth:.0%} (no budget stated)")
    if intent is Intent.TEST_DESIGN:
        if mde is None or mde <= 0:
            mde = p.default_mde
            used.append(f"MDE defaulted to +{mde * 100:.0f} percentage points on the visit rate")
        if daily is None or daily <= 0:
            daily = p.default_daily_eligible
            used.append(f"daily eligible traffic defaulted to {daily:,} customers/day")

    return CampaignScenario(
        intent=intent,
        question=question,
        budget_usd=budget,
        cost_per_contact_usd=cost,
        depth=depth,
        mde=mde,
        daily_eligible=daily,
        assumptions_used=used,
        missing_fields=missing,
    )


# --- the cached holdout context -------------------------------------------


@dataclass
class HoldoutData:
    """The model-agnostic split: arrays, assumptions, and the policy cache.

    Every per-model :class:`HoldoutContext` wraps one of these, so serving a
    second ranking model refits nothing heavy: the split is rebuilt never, and
    the segment policy tree is fitted once on the shared ``(X_tr, t_tr, y_tr)``
    and graded once on the shared holdout rows.
    """

    X_tr: np.ndarray
    t_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    t_te: np.ndarray
    y_visit_te: np.ndarray
    y_money_te: np.ndarray
    zip_code_te: np.ndarray
    feature_names: list[str]
    assumptions: BusinessAssumptions
    _profile: SegmentProfile | None = field(default=None, repr=False)

    @property
    def n_test(self) -> int:
        return int(len(self.y_money_te))

    @property
    def baseline_visit_rate(self) -> float:
        """Control-arm visit rate on the holdout — the power analysis baseline."""
        control = self.t_te == 0
        return float(self.y_visit_te[control].mean())

    def segment_profile(self) -> SegmentProfile:
        """The policy tree, fitted once (train) and graded on the holdout."""
        if self._profile is None:
            self._profile = get_segment_profile(
                self.X_tr,
                self.t_tr,
                self.y_tr,
                self.X_te,
                self.t_te,
                self.y_visit_te,
                feature_names=self.feature_names,
                y_eval_money=self.y_money_te,
                fairness_col=self.zip_code_te,
                max_depth=3,
                min_samples_leaf=500,
                random_state=42,
            )
        return self._profile


#: Split data keyed by seed — every catalog artifact trains on 42, so all six
#: contexts share one ``HoldoutData`` in practice.
_DATA_CACHE: dict[int, HoldoutData] = {}


def get_holdout_data(seed: int = 42, refresh: bool = False) -> HoldoutData:
    """The shared split for ``seed`` — built once per process."""
    cached = _DATA_CACHE.get(seed)
    if cached is None or refresh:
        arrays = build_holdout(seed)
        cached = HoldoutData(
            X_tr=arrays.X_tr,
            t_tr=arrays.t_tr,
            y_tr=arrays.y_tr,
            X_te=arrays.X_te,
            t_te=arrays.t_te,
            y_visit_te=arrays.y_visit_te,
            y_money_te=arrays.y_money_te,
            zip_code_te=arrays.zip_code_te,
            feature_names=list(arrays.feature_names),
            assumptions=BusinessAssumptions.from_yaml(),
        )
        _DATA_CACHE[seed] = cached
    return cached


@dataclass
class HoldoutContext:
    """One registered model + the shared holdout, cached per process.

    Construct one per artifact via :func:`build_context`; the arrays and the
    (expensive) segment profile come from the shared :class:`HoldoutData`, so
    only ``score`` — and the metadata describing what that score means — is
    per model.
    """

    X_tr: np.ndarray
    t_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    t_te: np.ndarray
    y_visit_te: np.ndarray
    y_money_te: np.ndarray
    score: np.ndarray
    zip_code_te: np.ndarray
    feature_names: list[str]
    assumptions: BusinessAssumptions
    meta: dict[str, Any]
    _profile: SegmentProfile | None = field(default=None, repr=False)
    # Model semantics (from packages/causal/catalog.py via the sidecar meta).
    # Defaults keep synthetic test contexts valid; real contexts fill them in.
    model_name: str = "champion"
    learner: str | None = None
    label: str | None = None
    score_kind: str | None = None
    tier: str = "champion"
    supports_intervals: bool = False
    research_only: bool = False
    data: HoldoutData | None = field(default=None, repr=False)

    @property
    def n_test(self) -> int:
        return int(len(self.y_money_te))

    @property
    def baseline_visit_rate(self) -> float:
        """Control-arm visit rate on the holdout — the power analysis baseline."""
        control = self.t_te == 0
        return float(self.y_visit_te[control].mean())

    def segment_profile(self) -> SegmentProfile:
        """The policy tree — the shared fit when the context wraps shared data."""
        if self._profile is None:
            if self.data is not None:
                self._profile = self.data.segment_profile()
            else:
                self._profile = get_segment_profile(
                    self.X_tr,
                    self.t_tr,
                    self.y_tr,
                    self.X_te,
                    self.t_te,
                    self.y_visit_te,
                    feature_names=self.feature_names,
                    y_eval_money=self.y_money_te,
                    fairness_col=self.zip_code_te,
                    max_depth=3,
                    min_samples_leaf=500,
                    random_state=42,
                )
        return self._profile

    def model_card(self) -> ModelCard:
        """This context's card — per artifact, never champion's text."""
        return get_model_card(self.model_name)


def _label_for(model_name: str, meta: dict[str, Any]) -> str:
    """Display label: the sidecar first, then the catalog as the fallback."""
    label = meta.get("label")
    if label:
        return str(label)
    spec = catalog.spec_for_learner(str(meta.get("learner", "response")))
    if spec is not None:
        return spec.label
    return model_name


def build_context(model_name: str = "champion") -> HoldoutContext:
    """Load a registered model and rebuild *its* split (same seed)."""
    model, meta = load_model(model_name)
    seed = int(meta.get("random_state", 42))
    data = get_holdout_data(seed)
    learner = str(meta.get("learner", "response"))
    spec = catalog.spec_for_learner(learner)
    return HoldoutContext(
        X_tr=data.X_tr,
        t_tr=data.t_tr,
        y_tr=data.y_tr,
        X_te=data.X_te,
        t_te=data.t_te,
        y_visit_te=data.y_visit_te,
        y_money_te=data.y_money_te,
        score=np.asarray(model.predict(data.X_te)).ravel(),
        zip_code_te=data.zip_code_te,
        feature_names=list(data.feature_names),
        assumptions=data.assumptions,
        meta=meta,
        model_name=model_name,
        learner=learner,
        label=_label_for(model_name, meta),
        score_kind=catalog.clean_score_kind(meta.get("score_kind"), learner),
        tier=catalog.clean_tier(meta.get("tier"), learner),
        supports_intervals=catalog.clean_bool(
            meta.get("supports_intervals"),
            bool(spec.supports_intervals if spec else False),
        ),
        research_only=catalog.clean_bool(
            meta.get("research_only"),
            bool(spec.research_only if spec else False),
        ),
        data=data,
    )


#: One cached context per artifact name — the split cache below it is shared.
_CONTEXTS: dict[str, HoldoutContext] = {}


def get_context(model_name: str = "champion", refresh: bool = False) -> HoldoutContext:
    """The process-wide context for ``model_name``: build on first use."""
    cached = _CONTEXTS.get(model_name)
    if cached is None or refresh:
        cached = build_context(model_name)
        _CONTEXTS[model_name] = cached
    return cached


# --- tool dispatch ---------------------------------------------------------


def _effective_depth(
    scenario: CampaignScenario, context: HoldoutContext, config: CopilotConfig
) -> float:
    """Stated depth, or the budget-derived one (capped at planner.max_depth)."""
    if scenario.depth is not None:
        return float(scenario.depth)
    if scenario.budget_usd is not None:
        cpc = scenario.cost_per_contact_usd or context.assumptions.cost_per_contact_usd
        derived = (scenario.budget_usd / cpc) / context.n_test
        return float(min(max(derived, 0.01), config.planner.max_depth))
    return config.planner.default_depth


def _champion_row(comparison: dict[str, Any]) -> dict[str, Any]:
    """The ranked-model row: every compare output names it '<model> ranking'."""
    for s in comparison.get("scenarios", []):
        if str(s.get("strategy", "")).endswith(" ranking"):
            return s
    return {}


def _what_if_delta(base: dict[str, Any], variant: dict[str, Any], changed: str) -> dict[str, Any]:
    """The side-by-side arithmetic — computed here (code), quoted by the writer."""
    champ_b, champ_v = _champion_row(base), _champion_row(variant)
    return {
        "changed": changed,
        "champion_profit_base_usd": champ_b.get("profit_usd", 0.0),
        "champion_profit_variant_usd": champ_v.get("profit_usd", 0.0),
        "profit_delta_usd": round(
            float(champ_v.get("profit_usd", 0.0)) - float(champ_b.get("profit_usd", 0.0)), 2
        ),
        "n_targeted_base": champ_b.get("n_targeted", 0),
        "n_targeted_variant": champ_v.get("n_targeted", 0),
    }


def run_tools(
    scenario: CampaignScenario,
    context: HoldoutContext,
    config: CopilotConfig,
) -> list[ToolRun]:
    """Dispatch to the Day-4 tools for this intent; failures become ToolRuns."""
    runs: list[ToolRun] = []
    # Scenario names follow the model in play ("Response baseline ranking");
    # the arithmetic is identical for any ranking.
    rank_label = context.label or context.model_name
    base_assumptions = context.assumptions
    cpc = scenario.cost_per_contact_usd or base_assumptions.cost_per_contact_usd
    effective = (
        base_assumptions
        if cpc == base_assumptions.cost_per_contact_usd
        else replace(base_assumptions, cost_per_contact_usd=float(cpc))
    )

    def _run(name: str, fn: Callable[[], Any]) -> ToolRun:
        with trace_span(
            name, "TOOL", inputs={"scenario": scenario.model_dump()}
        ) as span:
            try:
                run = tool_run(name, fn())
            except Exception as err:  # noqa: BLE001 - surfaced to the writer, never swallowed
                run = ToolRun(name=name, ok=False, error=f"{type(err).__name__}: {err}")
            runs.append(run)
            set_span_outputs(
                span,
                {
                    "ok": run.ok,
                    "error": run.error,
                    "output": trim_tool_output(name, run.output),
                },
            )
            return run

    if scenario.intent is Intent.PLAN:
        depth = _effective_depth(scenario, context, config)
        _run(
            "profit_point",
            lambda: profit_point(
                context.y_money_te, context.score, context.t_te, depth, effective
            ),
        )
        _run(
            "optimize_budget",
            lambda: optimize_budget(
                context.y_money_te,
                context.score,
                context.t_te,
                assumptions=base_assumptions,
                n_depths=10,
            ),
        )
        _run(
            "compare_scenarios",
            lambda: compare_scenarios(
                context.y_money_te,
                context.score,
                context.t_te,
                assumptions=effective,
                depth=depth,
                budget_usd=scenario.budget_usd,
                rank_label=rank_label,
            ),
        )
        _run("design_ab_test", _design_call(scenario, context, config))
        _run("get_model_card", context.model_card)
    elif scenario.intent is Intent.WHAT_IF:
        base_depth = _effective_depth(
            scenario.model_copy(update={"cost_per_contact_usd": None}), context, config
        )
        cpc_base = base_assumptions.cost_per_contact_usd
        variant_depth = base_depth
        if scenario.budget_usd is not None and scenario.cost_per_contact_usd is not None:
            variant_depth = min(
                max((scenario.budget_usd / float(cpc)) / context.n_test, 0.01),
                config.planner.max_depth,
            )
        base_run = _run(
            "compare_scenarios_base",
            lambda: compare_scenarios(
                context.y_money_te,
                context.score,
                context.t_te,
                assumptions=base_assumptions,
                depth=base_depth,
                budget_usd=scenario.budget_usd,
                rank_label=rank_label,
            ),
        )
        variant_run = _run(
            "compare_scenarios_variant",
            lambda: compare_scenarios(
                context.y_money_te,
                context.score,
                context.t_te,
                assumptions=effective,
                depth=variant_depth,
                budget_usd=scenario.budget_usd,
                rank_label=rank_label,
            ),
        )
        if base_run.ok and variant_run.ok and base_run.output and variant_run.output:
            if scenario.cost_per_contact_usd is not None:
                changed = (
                    f"cost per contact ${cpc_base:.2f} -> "
                    f"${float(scenario.cost_per_contact_usd):.2f}"
                )
            else:
                changed = (
                    f"budget ${scenario.budget_usd:,.0f} at "
                    f"${float(cpc):.2f} per contact -> depth "
                    f"{base_depth * 100:g}% -> {variant_depth * 100:g}%"
                )
            runs.append(
                tool_run(
                    "what_if_delta",
                    _what_if_delta(base_run.output, variant_run.output, changed),
                )
            )
        _run(
            "optimize_budget",
            lambda: optimize_budget(
                context.y_money_te,
                context.score,
                context.t_te,
                assumptions=base_assumptions,
                n_depths=10,
            ),
        )
    elif scenario.intent is Intent.EXPLAIN:
        _run("get_segment_profile", context.segment_profile)
    elif scenario.intent is Intent.TEST_DESIGN:
        _run("design_ab_test", _design_call(scenario, context, config))
        _run("get_model_card", context.model_card)
    return runs


def _design_call(
    scenario: CampaignScenario, context: HoldoutContext, config: CopilotConfig
) -> Callable[[], Any]:
    """The power-analysis call with documented defaults (shared by 3 intents)."""
    mde = scenario.mde if scenario.mde is not None else config.planner.default_mde
    daily = (
        scenario.daily_eligible
        if scenario.daily_eligible is not None
        else config.planner.default_daily_eligible
    )
    return lambda: design_ab_test(context.baseline_visit_rate, mde, daily_eligible=daily)
