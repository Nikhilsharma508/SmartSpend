"""Deterministic model resolution: the gate suggests, code decides.

Nothing here is generative. :func:`resolve_model` turns a *pinned* model and
the gate's ``model`` answer into exactly one registered artifact, under rules
that cannot be talked around by a prompt:

1. A valid explicit pin wins (``selector="manual"``) — validated against the
   catalog and the current mode, so a research-only pin without Research Mode
   is an error, never a silent downgrade.
2. Anything that is not plan/what-if answers with the champion
   (``selector="default"``): explain and test-design describe the registered
   decision stack, they do not re-plan under a different ranking.
3. A confident, allowed gate choice wins (``selector="gate"``) — same
   allow-list, same mode, and only for plan/what-if.
4. Everything else — no answer, low confidence, a model outside the
   allow-list, an unregistered artifact — falls back to the champion
   (``selector="fallback"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from packages.causal import catalog
from packages.causal.registry import is_registered
from packages.copilot.schemas import CopilotConfig, Intent, Route

#: Intents that can actually be answered with a different ranking model.
MODEL_SENSITIVE_INTENTS: tuple[Intent, ...] = (Intent.PLAN, Intent.WHAT_IF)

SelectorKind = Literal["gate", "manual", "fallback", "default"]


@dataclass(frozen=True)
class ModelSelection:
    """The one artifact that answers this request, and how it was chosen."""

    name: str
    label: str
    learner: str
    score_kind: str
    tier: str
    supports_intervals: bool
    selector: SelectorKind
    confidence: float | None
    research_mode: bool

    @property
    def research_only(self) -> bool:
        return self.tier == catalog.TIER_RESEARCH


def _selection(
    spec: catalog.ModelSpec,
    *,
    selector: SelectorKind,
    confidence: float | None,
    research_mode: bool,
) -> ModelSelection:
    return ModelSelection(
        name=spec.name,
        label=spec.label,
        learner=spec.learner,
        score_kind=spec.score_kind,
        tier=spec.tier,
        supports_intervals=spec.supports_intervals,
        selector=selector,
        confidence=confidence,
        research_mode=research_mode,
    )


def _resolved_pin(
    requested_model: str, research_mode: bool, models_dir: Path | None
) -> ModelSelection:
    """Validate a pinned model; ``ValueError`` names what is wrong.

    A pin must be a catalog name, selectable in the current mode, *and* backed
    by a registered artifact — a missing joblib is a caller error (a 422 at
    the API), not a crash inside the heavy load path (a 500).
    """
    spec = catalog.assert_model_choice(requested_model, research_mode=research_mode)
    if not is_registered(spec.name, models_dir=models_dir):
        raise ValueError(
            f"model {spec.name!r} is in the catalog but not registered under "
            f"{models_dir or 'models/'} — train it with pipelines/train_all.py"
        )
    return _selection(
        spec, selector="manual", confidence=None, research_mode=research_mode
    )


def resolve_model(
    route: Route,
    requested_model: str | None,
    *,
    research_mode: bool = False,
    config: CopilotConfig,
    models_dir: Path | None = None,
) -> ModelSelection:
    """Pin → intent → confident gate choice → champion fallback, in order.

    Raises
    ------
    ValueError
        A pinned model is unknown, unregistered, or not selectable in this
        mode (the caller surfaces it as a 4xx — never a silent swap).
    """
    if requested_model is not None:
        return _resolved_pin(str(requested_model), research_mode, models_dir)

    if route.intent not in MODEL_SENSITIVE_INTENTS:
        spec = catalog.get_spec(catalog.DEFAULT_MODEL)
        return _selection(
            spec, selector="default", confidence=None, research_mode=research_mode
        )

    choice, confidence = route.model_choice, route.model_confidence
    if (
        choice
        and confidence is not None
        and confidence >= config.router.t_model_confidence
        and catalog.is_allowed(choice, research_mode)
    ):
        try:
            spec = catalog.get_spec(choice)
            if is_registered(spec.name, models_dir=models_dir):
                return _selection(
                    spec,
                    selector="gate",
                    confidence=float(confidence),
                    research_mode=research_mode,
                )
        except ValueError:
            pass  # not a catalog name — fall through to the champion

    spec = catalog.get_spec(catalog.DEFAULT_MODEL)
    return _selection(
        spec,
        selector="fallback",
        confidence=float(confidence) if confidence is not None else None,
        research_mode=research_mode,
    )


def assert_valid_model_name(
    requested_model: str | None, research_mode: bool, models_dir: Path | None = None
) -> None:
    """Fail fast on a bad pin *before* the gate runs (a 4xx beats a 500).

    Checks everything ``resolve_model`` would check for a pin: catalog name,
    mode, and a registered artifact. ``None`` (no pin) is always fine.
    """
    if requested_model is None:
        return
    spec = catalog.assert_model_choice(requested_model, research_mode=research_mode)
    if not is_registered(spec.name, models_dir=models_dir):
        raise ValueError(
            f"model {spec.name!r} is in the catalog but not registered under "
            f"{models_dir or 'models/'} — train it with pipelines/train_all.py"
        )


def trace_fields(selection: ModelSelection) -> dict[str, Any]:
    """The flat tag/span fields MLflow records for the selection."""
    return {
        "ml.model": selection.name,
        "ml.score_kind": selection.score_kind,
        "model.selector": selection.selector,
        "model.confidence": selection.confidence,
        "research.mode": selection.research_mode,
    }
