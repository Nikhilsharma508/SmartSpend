"""The model card: what the registered model is, and where it must not be trusted.

A registered artifact without boundaries is a liability. :func:`get_model_card`
pairs the registry's provenance with the *decided* honest boundaries — the
things Day 2/3/4 measured and the Day-7 model card will repeat — so any
consumer (API, copilot brief, human) sees the caveats next to the numbers
instead of three documents later.
"""

from __future__ import annotations

from pathlib import Path

from packages.causal.registry import load_model
from packages.decision.schemas import AssumptionsSchema, ModelCard

#: What the card says about group-level treatment: the Day-4 fairness audit
#: measured targeting rate and holdout uplift by ``zip_code``; every group's
#: uplift interval overlapped zero at this sample size.
FAIRNESS_NOTE = (
    "Fairness audited on the holdout by zip_code (notebook 05, section 7): "
    "targeting rates track each group's share of the list, and every group's "
    "uplift interval includes zero at this sample size — so the model supports "
    "no group-level causal claim, only an overall one."
)

#: Where this model must not be used. Each line is a measured result, not a
#: generic disclaimer.
WHEN_NOT_TO_USE: list[str] = [
    "Not validated outside the Hillstrom email experiment — no claim transfers "
    "to another audience, channel or period.",
    "Do not target on predicted *spend*: the spend ordering sat inside its own "
    "random noise floor (Day 3, notebook 04 section 6).",
    "Do not present the champion's edge over the DR-learner as a win: its "
    "10-seed Qini edge (0.0048) is inside its own spread — a statistical tie, "
    "broken on profit and parsimony only.",
    "Not for individual-level causal claims (\"this customer will convert "
    "because of the email\") — intervals are wide per customer.",
    "Do not use visit-scores as conversion-probabilities without the per-arm "
    "recalibration from notebook 04 (raw Brier skill vs base rate is ~zero).",
]

#: Structural limitations that no retraining will fix.
LIMITATIONS: list[str] = [
    "Trained on the mens-email arm only; the womens arm is a separate fit.",
    "Randomized-experiment features (7 covariates) — richer behavioural data "
    "would likely move the ranking more than any learner swap has.",
    "Single-seed artifact; the honest performance envelope is the 10-seed "
    "benchmark in docs/concepts.md, not this file's one-shot metrics.",
]


def get_model_card(name: str = "champion", *, models_dir: Path | None = None) -> ModelCard:
    """Build the model card for a registered artifact.

    Raises whatever :func:`packages.causal.registry.load_model` raises — an
    unregistered name has no card.
    """
    _model, meta = load_model(name, models_dir=models_dir)
    a = meta["assumptions"]
    return ModelCard(
        name=name,
        learner=meta["learner"],
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
        fairness_note=FAIRNESS_NOTE,
        when_not_to_use=list(WHEN_NOT_TO_USE),
        limitations=list(LIMITATIONS),
    )
