"""The model card: what the registered model is, and where it must not be trusted.

A registered artifact without boundaries is a liability. :func:`get_model_card`
pairs the registry's provenance with the *decided* honest boundaries — the
things Day 2/3/4 measured, recorded in ``docs/concepts.md`` — so any consumer
(API, copilot brief, human) sees the caveats next to the numbers instead of
three documents later.

Boundaries are per learner, not one champion text replayed everywhere: a DR
caller must hear "effect, not probability", and an S/T/X/forest caller must
hear "research only — the 10-seed benchmark ranked it at random".
"""

from __future__ import annotations

from pathlib import Path

from packages.causal.catalog import (
    clean_bool,
    clean_score_kind,
    clean_tier,
    score_meaning,
    spec_for_learner,
)
from packages.causal.registry import load_model
from packages.decision.schemas import AssumptionsSchema, ModelCard

#: What the card says about group-level treatment: the Day-4 fairness audit
#: measured contact rate and holdout uplift by ``zip_code``; the contact
#: disparity is real (flagged, not smoothed), the effect claim stays overall.
#: The audit lives on the segment *policy* (DRPolicyTree), which is not one of
#: the ranking artifacts — every ranking card points there instead of
#: pretending the fairness numbers belong to the ranking model.
_FAIRNESS_POLICY_TREE = (
    "Fairness audited on the holdout by zip_code (notebook 04, section 7): the "
    "policy tree would email Urban customers at 59% against their 41% "
    "population share — a zip-proxy contact disparity, flagged here rather "
    "than smoothed over. Group uplift intervals all exclude zero and overlap "
    "each other: no group-level effect disparity is supported, but "
    "group-level causal claims are not the model's to make either. This audit "
    "belongs to the segment policy, not to this ranking score."
)

_RESPONSE_FAIRNESS = _FAIRNESS_POLICY_TREE
_DEFAULT_FAIRNESS = (
    "Fairness is audited on the segment policy (the DRPolicyTree and its "
    "zip_code audit), not on this ranking score, and no group-level causal "
    "claims are made for any model. See the Segments page for the measured "
    "audit."
)

#: Common, learner-independent boundaries — measured, not generic disclaimers.
_COMMON_WHEN_NOT_TO_USE: list[str] = [
    "Not validated outside the Hillstrom email experiment — no claim transfers "
    "to another audience, channel or period.",
    "Not for individual-level causal claims (\"this customer will convert "
    "because of the email\") — intervals are wide per customer.",
]

#: Where the response baseline must not be used. Each line is a measured
#: result, not a generic disclaimer.
_RESPONSE_WHEN_NOT_TO_USE: list[str] = [
    *_COMMON_WHEN_NOT_TO_USE,
    "Do not target on predicted *spend*: the spend ordering sat inside its own "
    "random noise floor (Day 3, notebook 03 section 6).",
    "Do not present the champion's edge over the DR-learner as a win: its "
    "10-seed Qini edge (0.0048) is inside its own spread — a statistical tie, "
    "broken on profit and parsimony only.",
    "Do not use visit-scores as conversion-probabilities without the per-arm "
    "recalibration from notebook 03 (raw Brier skill vs base rate is ~zero).",
]

#: The DR challenger: an effect model with its own traps.
_DR_WHEN_NOT_TO_USE: list[str] = [
    *_COMMON_WHEN_NOT_TO_USE,
    "Its score is an estimated incremental effect, not a probability — do not "
    "read it as \"chance of converting\" or calibrate plans on it as one.",
    "A linear DR stage may miss uplift heterogeneity a nonlinear learner "
    "would find; a flat ranking is a finding, not necessarily the truth.",
    "Do not present the DR-learner as beating the champion: the 10-seed "
    "benchmark is a statistical tie, so the champion keeps the default seat "
    "on profit and parsimony.",
]

#: The research meta-learners (S/T/X): relative ranking scores, weak signal.
_META_LEARNER_WHEN_NOT_TO_USE: list[str] = [
    *_COMMON_WHEN_NOT_TO_USE,
    "Research mode only: across 10 seeds the benchmark ranked this learner at "
    "or inside random noise — it must not drive a live campaign.",
    "Its score is a relative ranking number, not a calibrated probability, "
    "not an effect size and not a dollar value.",
    "Use it to compare how meta-learners behave, never to rank customers for "
    "an actual send.",
]

#: The causal forest: effect + interval, research only.
_FOREST_WHEN_NOT_TO_USE: list[str] = [
    *_COMMON_WHEN_NOT_TO_USE,
    "Research mode only: across 10 seeds the benchmark ranked this learner "
    "statistically at random — it must not drive a live campaign.",
    "Effect intervals describe uncertainty; a positive interval on one "
    "customer's effect is not an actionable guarantee.",
    "Use it to probe heterogeneous effects, never to rank customers for an "
    "actual send.",
]

_WHEN_NOT_BY_LEARNER: dict[str, list[str]] = {
    "response": _RESPONSE_WHEN_NOT_TO_USE,
    "dr": _DR_WHEN_NOT_TO_USE,
    "s": _META_LEARNER_WHEN_NOT_TO_USE,
    "t": _META_LEARNER_WHEN_NOT_TO_USE,
    "x": _META_LEARNER_WHEN_NOT_TO_USE,
    "forest": _FOREST_WHEN_NOT_TO_USE,
}

#: Structural limitations that no retraining will fix.
LIMITATIONS: list[str] = [
    "Trained on the mens-email arm only; the womens arm is a separate fit.",
    "Randomized-experiment features (7 covariates) — richer behavioural data "
    "would likely move the ranking more than any learner swap has.",
    "Single-seed artifact; the honest performance envelope is the 10-seed "
    "benchmark in docs/concepts.md, not this file's one-shot metrics.",
]

#: The sentence a card carries under its "when not to use" list, restating
#: what the score is (per-learner differences already cover the traps).
def _score_text(score_kind: str) -> str:
    return score_meaning(score_kind)


def get_model_card(name: str = "champion", *, models_dir: Path | None = None) -> ModelCard:
    """Build the model card for a registered artifact.

    Raises whatever :func:`packages.causal.registry.load_model` raises — an
    unregistered name has no card.
    """
    _model, meta = load_model(name, models_dir=models_dir)
    a = meta["assumptions"]
    learner = str(meta.get("learner", "response"))
    spec = spec_for_learner(learner)
    tier = clean_tier(meta.get("tier"), learner)
    score_kind = clean_score_kind(meta.get("score_kind"), learner)
    default_intervals = spec.supports_intervals if spec is not None else bool(tier != "research")
    default_research = spec.research_only if spec is not None else True
    return ModelCard(
        name=name,
        learner=learner,
        label=str(meta.get("label") or (spec.label if spec else learner)),
        target=meta["target"],
        trained_at=meta["created_at"],
        train_rows=meta["train_rows"],
        data_sha256=meta["data_sha256"],
        metrics={k: v for k, v in meta["metrics"].items()},
        assumptions=AssumptionsSchema(
            margin_rate=a["margin_rate"],
            cost_per_contact_usd=a["cost_per_contact_usd"],
            value_per_conversion_usd=a["value_per_conversion_usd"],
        ),
        tier=tier,
        score_kind=score_kind,
        supports_intervals=clean_bool(meta.get("supports_intervals"), default_intervals),
        research_only=clean_bool(meta.get("research_only"), default_research),
        use_case=str(
            meta.get("use_case") or (spec.use_case if spec else "See the model card before use.")
        ),
        fairness_note=(
            _RESPONSE_FAIRNESS if learner == "response" else _DEFAULT_FAIRNESS
        ),
        when_not_to_use=list(
            _WHEN_NOT_BY_LEARNER.get(learner, _RESPONSE_WHEN_NOT_TO_USE)
        )
        + ([f"Score meaning: {_score_text(score_kind)}"] if learner != "response" else []),
        limitations=list(LIMITATIONS),
    )
