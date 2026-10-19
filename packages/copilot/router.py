"""The [1] GATE — one Decisions API call, thresholded into a route.

The gate asks three *named* questions in a single request (shape verified live
in ``research/notebooks/decision_api.ipynb``):

- ``intent``          — a ``choice`` over the four use cases plus ``out_of_scope``
- ``policy_risk``     — a ``noul``: does the request target or exclude people by
                        a sensitive attribute or a proxy for one?
- ``missing_inputs``  — a ``noul``: is a required parameter absent?

Design rules from ``docs/decision_api.md`` that apply here: the gate emits
typed probabilities and never prose; every question is named; the choice always
carries an ``other`` escape (``out_of_scope``); independent questions ride one
call so the fast paths stay fast.

If the endpoint is unreachable, errors, or *refuses* a question,
:func:`keyword_fallback_route` keeps the copilot answering deterministically —
recorded as ``router=keyword_fallback`` in the trace, never silently.
"""

from __future__ import annotations

from typing import Literal

from packages.copilot.providers import DecisionsClient, ProviderError
from packages.copilot.schemas import CopilotConfig, Intent, Route

#: Question names the router reads out of the answers dict.
QUESTION_INTENT = "intent"
QUESTION_POLICY_RISK = "policy_risk"
QUESTION_MISSING = "missing_inputs"

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


def build_questions() -> dict[str, dict[str, object]]:
    """The three named questions, exactly as the live endpoint expects them."""
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
    }


def parse_route(
    answers: dict[str, dict], router: Literal["decisions", "keyword_fallback"] = "decisions"
) -> Route:
    """Turn the Decisions answers dict into a :class:`Route`.

    Raises
    ------
    ProviderError
        Any question was refused — the whole route is untrustworthy, so the
        caller falls back to the keyword router rather than half-routing.
    """
    refused = [
        name
        for name, ans in answers.items()
        if isinstance(ans, dict) and ans.get("type") == "refusal"
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

    return Route(
        intent=intent,
        intent_confidence=float(confidence),
        policy_risk=_noul(QUESTION_POLICY_RISK),
        missing_inputs=_noul(QUESTION_MISSING),
        router=router,
        answers=dict(answers),
    )


def route_question(question: str, client: DecisionsClient) -> Route:
    """One gate call; on any provider failure, the deterministic keyword route."""
    state = {
        "user_question": question,
        "available_context": (
            "Campaign copilot for uplift-based email targeting on the Hillstrom "
            "experiment; tools: optimize_budget, compare_scenarios, "
            "get_segment_profile, design_ab_test, get_model_card."
        ),
    }
    try:
        body = client.decide(state, build_questions())
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


def keyword_fallback_route(question: str) -> Route:
    """Score the question against keyword sets; no model, no network.

    Precision is intentionally modest — this path exists for availability, and
    the trace always says ``router=keyword_fallback`` so evals can price it.
    """
    low = question.lower()
    policy_hit = any(k in low for k in _POLICY_KEYWORDS)
    scores = {intent: sum(1 for k in keys if k in low) for intent, keys in _INTENT_KEYWORDS.items()}
    best = max(scores, key=lambda i: scores[i])
    if scores[best] == 0:
        return Route(
            intent=Intent.OUT_OF_SCOPE,
            intent_confidence=0.4,
            policy_risk=0.9 if policy_hit else 0.02,
            missing_inputs=0.0,
            router="keyword_fallback",
            answers={"keyword_scores": {i.value: s for i, s in scores.items()}},
        )
    total = sum(scores.values())
    return Route(
        intent=best,
        intent_confidence=round(scores[best] / total, 4),
        policy_risk=0.9 if policy_hit else 0.02,
        missing_inputs=0.0,
        router="keyword_fallback",
        answers={"keyword_scores": {i.value: s for i, s in scores.items()}},
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
