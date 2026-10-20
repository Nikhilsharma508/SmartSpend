"""The registered model catalog — one source of truth for the six artifacts.

Every consumer (``pipelines/train.py``, the API, the copilot gate, the model
lab) reads from here, so artifact names, tiers and score meanings exist in
exactly one place. The catalog answers three questions the code must never
answer with ad-hoc literals:

1. *Which artifacts exist and what are they called?*
2. *What does each model's score mean?* (a response probability is not an
   incremental effect; an uplift effect is not a probability)
3. *Which models may drive a live plan right now?* — normal mode restricts
   the allow-list to champion + DR; Research Mode unlocks all six, because
   the 10-seed benchmark showed S/T/X/forest ranking at or inside random
   noise (``docs/concepts.md`` § benchmark).

Note the DRPolicyTree is **not** here: it is the segment-policy tool, not one
of the ranking artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

TIER_CHAMPION = "champion"
TIER_CHALLENGER = "challenger"
TIER_RESEARCH = "research"

#: Every score kind the catalog understands, with the sentence a writer/UI
#: must use when it explains what the numbers are.
SCORE_RESPONSE = "response_probability"
SCORE_RELATIVE_UPLIFT = "relative_uplift"
SCORE_UPLIFT_EFFECT = "uplift_effect"
SCORE_UPLIFT_EFFECT_INTERVAL = "uplift_effect_interval"

SCORE_MEANING: dict[str, str] = {
    SCORE_RESPONSE: (
        "P(visit | emailed) — a likelihood learned on the treated arm; NOT an "
        "incremental or causal effect."
    ),
    SCORE_RELATIVE_UPLIFT: (
        "a relative uplift ranking score — ordering only, not a calibrated "
        "probability and not a dollar value."
    ),
    SCORE_UPLIFT_EFFECT: (
        "an estimated incremental effect of the email (treatment minus "
        "control); NOT a probability."
    ),
    SCORE_UPLIFT_EFFECT_INTERVAL: (
        "an estimated incremental effect with a statistical interval — "
        "research use; NOT a probability and not a guarantee."
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """One registered artifact: its name, role, and what its score means."""

    name: str
    learner: str
    label: str
    tier: str
    score_kind: str
    supports_intervals: bool
    research_only: bool
    use_case: str
    warning: str | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="champion",
        learner="response",
        label="Response baseline",
        tier=TIER_CHAMPION,
        score_kind=SCORE_RESPONSE,
        supports_intervals=False,
        research_only=False,
        use_case=(
            "Who is likely to visit after an email? Budget planning and "
            "general campaigns."
        ),
        warning=None,
    ),
    ModelSpec(
        name="dr-learner",
        learner="dr",
        label="DR uplift learner",
        tier=TIER_CHALLENGER,
        score_kind=SCORE_UPLIFT_EFFECT,
        supports_intervals=False,
        research_only=False,
        use_case=(
            "Who is persuadable, and what is the estimated incremental effect?"
        ),
        warning=(
            "Statistical tie with the response champion on the 10-seed "
            "benchmark; linear DR may miss heterogeneity the true uplift has."
        ),
    ),
    ModelSpec(
        name="s-learner",
        learner="s",
        label="S-learner (meta)",
        tier=TIER_RESEARCH,
        score_kind=SCORE_RELATIVE_UPLIFT,
        supports_intervals=False,
        research_only=True,
        use_case=(
            "Research comparison of meta-learner rankings; not for normal "
            "campaign decisions."
        ),
        warning=(
            "10-seed benchmark put its Qini inside random noise — do not "
            "drive a live campaign with it."
        ),
    ),
    ModelSpec(
        name="t-learner",
        learner="t",
        label="T-learner (meta)",
        tier=TIER_RESEARCH,
        score_kind=SCORE_RELATIVE_UPLIFT,
        supports_intervals=False,
        research_only=True,
        use_case=(
            "Research comparison of meta-learner rankings; not for normal "
            "campaign decisions."
        ),
        warning=(
            "10-seed benchmark ranked it statistically at random — do not "
            "drive a live campaign with it."
        ),
    ),
    ModelSpec(
        name="x-learner",
        learner="x",
        label="X-learner (meta)",
        tier=TIER_RESEARCH,
        score_kind=SCORE_RELATIVE_UPLIFT,
        supports_intervals=False,
        research_only=True,
        use_case=(
            "Research comparison of meta-learner rankings; not for normal "
            "campaign decisions."
        ),
        warning=(
            "10-seed benchmark ranked it statistically at random — do not "
            "drive a live campaign with it."
        ),
    ),
    ModelSpec(
        name="causal-forest",
        learner="forest",
        label="Causal forest",
        tier=TIER_RESEARCH,
        score_kind=SCORE_UPLIFT_EFFECT_INTERVAL,
        supports_intervals=True,
        research_only=True,
        use_case=(
            "Research estimate with effect intervals; not for normal campaign "
            "decisions."
        ),
        warning=(
            "10-seed benchmark ranked it statistically at random; intervals "
            "describe uncertainty, they do not promise an effect."
        ),
    ),
)

_BY_NAME: dict[str, ModelSpec] = {s.name: s for s in MODEL_SPECS}
_BY_LEARNER: dict[str, ModelSpec] = {s.learner: s for s in MODEL_SPECS}

#: The artifact the app opens with and falls back to everywhere.
DEFAULT_MODEL: str = _BY_NAME["champion"].name

_VALID_TIERS = {TIER_CHAMPION, TIER_CHALLENGER, TIER_RESEARCH}
_VALID_SCORE_KINDS = set(SCORE_MEANING)


def get_spec(name: str) -> ModelSpec:
    """The spec for an artifact name; ``ValueError`` names the catalog."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(
            f"unknown model {name!r}: registered models are "
            f"{[s.name for s in MODEL_SPECS]}"
        )
    return spec


def spec_for_learner(learner: str) -> ModelSpec | None:
    """The spec whose learner matches (None for learners outside the catalog)."""
    return _BY_LEARNER.get(learner)


def score_meaning(score_kind: str | None) -> str:
    """The one-sentence meaning of a score kind, for prompts and UI."""
    if score_kind is None:
        return "the registered model's ranking score."
    return SCORE_MEANING.get(score_kind, score_kind)


def allowed_models(research_mode: bool) -> tuple[str, ...]:
    """Model names selectable for live plans in the current mode."""
    if research_mode:
        return tuple(s.name for s in MODEL_SPECS)
    return tuple(s.name for s in MODEL_SPECS if not s.research_only)


def is_allowed(name: str, research_mode: bool) -> bool:
    """True when ``name`` is in the catalog and selectable in this mode."""
    return name in allowed_models(research_mode)


def assert_model_choice(name: str, *, research_mode: bool) -> ModelSpec:
    """Validate a *pinned* model choice; raise ``ValueError`` when unusable."""
    spec = get_spec(name)
    if not is_allowed(spec.name, research_mode):
        raise ValueError(
            f"model {name!r} is research-only here: enable Research Mode to "
            f"use it (normal mode allows {list(allowed_models(research_mode))})"
        )
    return spec


def clean_tier(value: object, learner: str) -> str:
    """A metadata ``tier`` that is actually one of the known tiers."""
    if isinstance(value, str) and value in _VALID_TIERS:
        return value
    spec = spec_for_learner(learner)
    if spec is not None:
        return spec.tier
    return TIER_RESEARCH


def clean_score_kind(value: object, learner: str) -> str:
    """A metadata ``score_kind`` that is actually one of the known kinds."""
    if isinstance(value, str) and value in _VALID_SCORE_KINDS:
        return value
    spec = spec_for_learner(learner)
    return spec.score_kind if spec else SCORE_RELATIVE_UPLIFT


def clean_bool(value: object, default: bool) -> bool:
    """A metadata boolean that may have been serialized as other types."""
    return value if isinstance(value, bool) else default
