"""The [1] GATE — one Decisions API call, thresholded into a route.

The gate asks four *named* questions in a single request (shape verified live
in ``research/notebooks/decision_api.ipynb``):

- ``intent``          — a ``choice`` over the four use cases plus ``out_of_scope``
- ``policy_risk``     — a ``noul``: does the request target or exclude people by
                        a sensitive attribute or a proxy for one?
- ``missing_inputs``  — a ``noul``: is a required parameter absent?
- ``model``           — a ``choice`` over the registered ranking models (plus
                        ``none``): which artifact should answer plan/what-if?

The model question produces a *suggestion*, never a decision: the resolver in
:mod:`packages.copilot.selection` applies the allow-list, the confidence
threshold and the champion fallback, so the gate can never pick a model the
user could not pick themselves.

Design rules from ``docs/decision_api.md`` that apply here: the gate emits
typed probabilities and never prose; every question is named; the choice always
carries an ``other`` escape (``out_of_scope`` / ``none``); independent
questions ride one call so the fast paths stay fast.

If the endpoint is unreachable, errors, or *refuses* a core question,
:func:`keyword_fallback_route` keeps the copilot answering deterministically —
recorded as ``router=keyword_fallback`` in the trace, never silently. A refusal
on the auxiliary ``model`` question is absorbed (the choice becomes ``None``);
a refusal on ``intent``/``policy_risk``/``missing_inputs`` poisons the whole
route.
"""

from __future__ import annotations

from typing import Any, Literal

from packages.causal import catalog
from packages.copilot.providers import DecisionsClient, ProviderError
from packages.copilot.schemas import CopilotConfig, Intent, Route

#: Question names the router reads out of the answers dict.
QUESTION_INTENT = "intent"
QUESTION_POLICY_RISK = "policy_risk"
QUESTION_MISSING = "missing_inputs"
QUESTION_MODEL = "model"

INTENT_CRITERIA: dict[str, str] = {
    "plan": (
        "Asks how to allocate a campaign budget or who to target — a plan for the "
        "setup exactly as stated ('I have $500 — how deep, what will I earn?'; "
        "nothing changes, they want the plan for what they already have). A bare "
        "'should we run a campaign?' — the campaign itself as the subject — is "
        "also plan: answer it with defaults or one clarifying question."
    ),
    "what_if": (
        "Asks how the outcome changes if an assumption CHANGES from a baseline — "
        "'what if the cost rose to $0.60?', 'if we doubled the budget' — a "
        "counterfactual with an explicit proposed change, not a plan as stated."
    ),
    "explain": "Asks who the segments are or why some customers should not be targeted.",
    "test_design": "Asks about experiment size, duration, MDE or success metrics.",
    "out_of_scope": (
        "Not about campaign planning, uplift, budget, targeting or testing. A vague "
        "question that names no concrete action or subject ('should we do something "
        "about X?') is out_of_scope — there is nothing to plan. Use this whenever "
        "the request fits none of the others."
    ),
}


def _model_instructions(research_mode: bool) -> str:
    base = (
        "Which registered ranking model should produce the plan's numbers? "
        "Pick the one whose score meaning matches what the marketer asked "
        "about. Pick 'none' when the request does not favour any model."
    )
    if research_mode:
        return base + " Research Mode is ON: research-only models are selectable."
    return base + " Research Mode is OFF: only the champion and the DR challenger are selectable."


def _model_criteria(research_mode: bool) -> dict[str, str]:
    """Choice criteria over the catalog — the names the resolver validates."""
    entries = {"none": "No model preference; leave the default champion in place."}
    for spec in catalog.MODEL_SPECS:
        if spec.research_only and not research_mode:
            continue
        text = (
            f"{spec.label} (tier: {spec.tier}). Its score is "
            f"{catalog.score_meaning(spec.score_kind)} Use case: {spec.use_case}"
        )
        if spec.tier == catalog.TIER_CHAMPION:
            text += " The safe default for budget planning."
        if spec.warning:
            text += f" Caveat: {spec.warning}"
        entries[spec.name] = text
    return entries


def build_questions(research_mode: bool = False) -> dict[str, dict[str, object]]:
    """The four named questions, exactly as the live endpoint expects them."""
    return {
        QUESTION_INTENT: {
            "type": "choice",
            "name": QUESTION_INTENT,
            "instructions": "Which use case does this marketing request belong to?",
            "criteria": dict(INTENT_CRITERIA),
        },
        QUESTION_POLICY_RISK: {
            "type": "noul",
            "name": QUESTION_POLICY_RISK,
            "instructions": (
                "Does the request target or exclude people by a sensitive attribute "
                "or a proxy for one (age, gender, race/ethnicity, religion, income, "
                "marital status, or location used as a stand-in for those)?"
            ),
            "criteria": {
                "true": "Requests targeting or exclusion by a sensitive attribute or a proxy.",
                "false": "No sensitive attribute or proxy is involved.",
            },
        },
        QUESTION_MISSING: {
            "type": "noul",
            "name": QUESTION_MISSING,
            "instructions": (
                "Is the request so empty of specifics that the only helpful reply is "
                "ONE clarifying question back? The copilot has standard defaults for "
                "contact cost, margin, targeting depth, test MDE and daily volume, "
                "and it can present a profit curve when no budget is given — so "
                "budget, audience and test questions are normally answerable as "
                "stated. Only a vague request with no goal, amount, audience or "
                "test needs a question back."
            ),
            "criteria": {
                "true": (
                    "No usable specifics at all — no goal, no amount, no audience, "
                    "no test (e.g. a bare 'should we run a campaign?') — so any "
                    "recommendation would be guesswork."
                ),
                "false": (
                    "What is stated, the standard defaults, or a presented profit "
                    "curve can carry an answer."
                ),
            },
        },
        QUESTION_MODEL: {
            "type": "choice",
            "name": QUESTION_MODEL,
            "instructions": _model_instructions(research_mode),
            "criteria": _model_criteria(research_mode),
        },
    }


def parse_route(
    answers: dict[str, dict], router: Literal["decisions", "keyword_fallback"] = "decisions"
) -> Route:
    """Turn the Decisions answers dict into a :class:`Route`.

    The ``model`` answer is auxiliary: a refusal there leaves the route valid
    with ``model_choice=None`` (the resolver keeps the champion), while a
    refusal on any core question raises, because half of a poisoned route is
    still poison.

    Raises
    ------
    ProviderError
        Any core question was refused — the whole route is untrustworthy, so
        the caller falls back to the keyword router rather than half-routing.
    """
    refused = [
        name
        for name, ans in answers.items()
        if isinstance(ans, dict) and ans.get("type") == "refusal" and name != QUESTION_MODEL
    ]
    if refused:
        raise ProviderError("decisions", f"question(s) refused: {refused}")

    intent_ans = answers.get(QUESTION_INTENT, {})
    choice = str(intent_ans.get("choice", "out_of_scope"))
    try:
        intent = Intent(choice)
    except ValueError:
        intent = Intent.OUT_OF_SCOPE

    confidence = intent_ans.get("confidence")
    if confidence is None:
        probabilities = intent_ans.get("probabilities") or {}
        confidence = max(probabilities.values(), default=0.5) if probabilities else 0.5

    def _noul(name: str) -> float:
        ans = answers.get(name, {})
        raw = ans.get("noul")
        try:
            return float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    model_ans = answers.get(QUESTION_MODEL, {})
    model_ans = model_ans if isinstance(model_ans, dict) else {}
    raw_model_choice = str(model_ans.get("choice") or "")
    model_choice: str | None = None
    if raw_model_choice and raw_model_choice not in ("none", "None"):
        model_choice = raw_model_choice
    model_confidence = model_ans.get("confidence")
    if model_confidence is None:
        probabilities = model_ans.get("probabilities") or {}
        model_confidence = (
            max(probabilities.values(), default=None) if probabilities else None
        )
    try:
        model_confidence = float(model_confidence) if model_confidence is not None else None
    except (TypeError, ValueError):
        model_confidence = None

    return Route(
        intent=intent,
        intent_confidence=float(confidence),
        policy_risk=_noul(QUESTION_POLICY_RISK),
        missing_inputs=_noul(QUESTION_MISSING),
        model_choice=model_choice,
        model_confidence=model_confidence,
        router=router,
        answers=dict(answers),
    )


def build_state(question: str, *, research_mode: bool = False) -> dict[str, Any]:
    """The Decisions API ``state`` payload for one question.

    The context line lists the registered models and the mode, so the gate's
    ``model`` answer is made against the same allow-list the resolver enforces.
    """
    selectable = [
        f"{spec.name} ({spec.label}, {spec.tier})"
        for spec in catalog.MODEL_SPECS
        if (not spec.research_only or research_mode)
    ]
    return {
        "user_question": question,
        "available_context": (
            "Campaign copilot for uplift-based email targeting on the Hillstrom "
            "experiment; tools: optimize_budget, compare_scenarios, "
            "get_segment_profile, design_ab_test, get_model_card. Registered "
            f"targeting models: {'; '.join(selectable)}. Research Mode is "
            f"{'ON' if research_mode else 'OFF'} — research-only models are "
            f"{'selectable' if research_mode else 'locked'}."
        ),
    }


def route_question(
    question: str,
    client: DecisionsClient,
    *,
    state: dict[str, Any] | None = None,
    questions: dict[str, dict[str, object]] | None = None,
    research_mode: bool = False,
) -> Route:
    """One gate call; on any provider failure, the deterministic keyword route."""
    state = state if state is not None else build_state(question, research_mode=research_mode)
    questions = questions if questions is not None else build_questions(research_mode)
    try:
        body = client.decide(state, questions)
        return parse_route(body.get("answers", {}))
    except ProviderError:
        return keyword_fallback_route(question)


# --- T6: the no-Decisions fallback (deterministic, no LLM) -----------------

_INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.WHAT_IF: (
        "what if",
        "what would",
        "suppose",
        "instead",
        "rise",
        "raise",
        "increase to",
        "drop to",
        "double",
        "halve",
        "change if",
    ),
    Intent.TEST_DESIGN: (
        "how long",
        "test",
        "experiment",
        "sample size",
        "power",
        "mde",
        "detect",
        "duration",
        "a/b",
    ),
    Intent.EXPLAIN: (
        "who are",
        "who should",
        "segment",
        "persuadable",
        "sleeping dog",
        "explain",
        "avoid",
        "profile",
    ),
    Intent.PLAN: (
        "budget",
        "spend",
        "target",
        "allocate",
        "campaign",
        "how deep",
        "who to",
        "worth",
    ),
}

#: Sensitive attributes and proxies — the keyword twin of the policy noul.
_POLICY_KEYWORDS: tuple[str, ...] = (
    "zip code",
    "zipcode",
    "postal code",
    "race",
    "ethnicity",
    "hispanic",
    "black",
    "gender",
    "female",
    "male",
    "women only",
    "men only",
    "religion",
    "marital",
    "single mothers",
    "seniors",
    "minorities",
    "years old",
    "aged",
    "income",
)


#: Uplift-vocabulary keywords — when the marketer asks about *incremental*
#: effects rather than response likelihood, the deterministic fallback points
#: at the DR challenger. Never a research-only model: the resolver keeps weak
#: models out of automatic choices by construction.
_MODEL_KEYWORDS: tuple[str, ...] = (
    "uplift",
    "persuadable",
    "persuade",
    "incremental",
    "increment",
    "causal effect",
    "cause to buy",
    "would they buy anyway",
)


def keyword_fallback_route(question: str) -> Route:
    """Score the question against keyword sets; no model, no network.

    Precision is intentionally modest — this path exists for availability, and
    the trace always says ``router=keyword_fallback`` so evals can price it.
    A DR-keyword hit sets ``model_choice`` as a *suggestion*; plan/what-if
    intents resolve it, everything else ignores it.
    """
    low = question.lower()
    policy_hit = any(k in low for k in _POLICY_KEYWORDS)
    scores = {intent: sum(1 for k in keys if k in low) for intent, keys in _INTENT_KEYWORDS.items()}
    if scores[max(scores, key=lambda i: scores[i])] == 0:
        return Route(
            intent=Intent.OUT_OF_SCOPE,
            intent_confidence=0.4,
            policy_risk=0.9 if policy_hit else 0.02,
            missing_inputs=0.0,
            model_choice=None,
            model_confidence=None,
            router="keyword_fallback",
            answers={"keyword_scores": {i.value: s for i, s in scores.items()}},
        )
    total = sum(scores.values())
    model_choice = "dr-learner" if any(k in low for k in _MODEL_KEYWORDS) else None
    return Route(
        intent=max(scores, key=lambda i: scores[i]),
        intent_confidence=round(scores[max(scores, key=lambda i: scores[i])] / total, 4),
        policy_risk=0.9 if policy_hit else 0.02,
        missing_inputs=0.0,
        model_choice=model_choice,
        model_confidence=0.65 if model_choice else None,
        router="keyword_fallback",
        answers={
            "keyword_scores": {i.value: s for i, s in scores.items()},
            "model_keywords": [k for k in _MODEL_KEYWORDS if k in low],
        },
    )


def decide_fast_path(route: Route, config: CopilotConfig) -> str:
    """Apply the thresholds in priority order.

    1. risk >= ``t_policy_risk``                -> decline (the fixed refusal)
    2. out_of_scope at >= ``t_oos_confidence``  -> redirect (confident it is off-scope)
    3. missing >= ``t_missing``                 -> clarify (a *specific* question beats
       a generic redirect even when the intent choice is uncertain)
    4. out_of_scope at low confidence, or any
       choice below ``t_confidence``            -> redirect (never guess)
    5. otherwise                                -> brief
    """
    if route.policy_risk >= config.router.t_policy_risk:
        return "decline"
    if (
        route.intent is Intent.OUT_OF_SCOPE
        and route.intent_confidence >= config.router.t_oos_confidence
    ):
        return "redirect"
    if route.missing_inputs >= config.router.t_missing:
        return "clarify"
    if route.intent is Intent.OUT_OF_SCOPE or route.intent_confidence < config.router.t_confidence:
        return "redirect"
    return "brief"
