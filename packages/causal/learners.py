"""One factory for every uplift model, behind a single fit/predict interface.

Day 2 fitted S/T/X-learners ad hoc in the notebook. Day 4's decision tools and
``pipelines/train.py`` need to call *any* model the same way, so the wrapping
lives here once. Two library quirks are absorbed in this module and nowhere
else:

- **causalml** (S/T/X) uses ``fit(X, treatment, y)`` and returns relative scores
  from ``predict(X)``. Its X-learner needs a *regressor* for the effect stage
  because it fits continuous imputed effects, while the outcome stage is a
  classifier.
- **econml** (DR, causal forest) uses ``fit(Y, T, X=X)``. ``effect()`` returns a
  1-D array for a binary treatment, and ``effect_interval(X, alpha)`` returns
  ``(lower, upper)`` — the point estimate must be fetched separately from
  ``effect()``.

Every model here outputs a **relative uplift ranking score** for binary
outcomes; for a continuous target (``spend``) the score is an estimated effect
in outcome units. Only the causal forest exposes effect intervals.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

#: Supported model names. ``"response"`` is the treated-only baseline, not an
#: uplift model, but it is scored the same way so tables stay uniform.
LearnerName = Literal["response", "s", "t", "x", "dr", "forest"]

#: Every name the factory accepts, in the order they appear in benchmark tables.
LEARNER_NAMES: tuple[LearnerName, ...] = ("response", "s", "t", "x", "dr", "forest")

#: The outcome columns a learner can be trained on. ``spend`` is the only
#: continuous one, and selecting it switches base learners to regressors.
VALID_TARGETS = frozenset({"visit", "conversion", "spend"})

#: Default LightGBM hyperparameters. Deliberately moderate: Day 2 showed the
#: uplift signal on Hillstrom is weak, so heavy capacity overfits noise.
N_ESTIMATORS = 300
LEARNING_RATE = 0.03
NUM_LEAVES = 63
MIN_CHILD_SAMPLES = 20


class UpliftModel(Protocol):
    """What every learner exposes to the rest of the project."""

    name: str
    supports_intervals: bool

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Per-customer uplift score (ranking, or effect in outcome units)."""

    def effect_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(lower, upper)`` per customer, or ``None`` if unsupported."""


def _clf(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        random_state=random_state,
        verbose=-1,
    )


def _reg(random_state: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        random_state=random_state,
        verbose=-1,
    )


class _CausalmlModel:
    """Adapter for the causalml S/T/X meta-learners.

    On a binary outcome the meta-learner predicts a relative score; on a
    continuous one (``spend``) it predicts an effect estimate. Either way only
    the ranking is consumed downstream, and no intervals are available.
    """

    supports_intervals = False

    def __init__(self, name: str, random_state: int, continuous: bool):
        # Import lazily so importing this module does not pull in causalml's
        # full dependency graph (it is slow to import).
        from causalml.inference.meta import (
            BaseSClassifier,
            BaseSRegressor,
            BaseTClassifier,
            BaseTRegressor,
            BaseXClassifier,
            BaseXRegressor,
        )

        self.name = name
        self.continuous = continuous
        # A continuous target (spend) needs the *Regressor meta-learners, not
        # just regressor base learners: the Classifier variants call
        # predict_proba internally and crash on a continuous outcome. Only the
        # X-learner's effect stage is a regressor regardless, since it always
        # fits continuous imputed effects.
        if continuous:
            s_cls, t_cls, x_cls = BaseSRegressor, BaseTRegressor, BaseXRegressor
        else:
            s_cls, t_cls, x_cls = BaseSClassifier, BaseTClassifier, BaseXClassifier
        base = _reg if continuous else _clf
        if name == "s":
            self._model = s_cls(base(random_state))
        elif name == "t":
            self._model = t_cls(base(random_state), base(random_state))
        elif name == "x":
            # The two X-learner variants take different kwarg names: the
            # Classifier wants outcome_learner/effect_learner, while the
            # Regressor wants learner= for the outcome plus separate
            # control/treatment effect learners (there is no single
            # effect_learner kwarg). Both effect stages are regressors, since
            # they fit continuous imputed effects either way.
            if continuous:
                self._model = x_cls(
                    learner=base(random_state),
                    control_effect_learner=_reg(random_state),
                    treatment_effect_learner=_reg(random_state),
                )
            else:
                self._model = x_cls(
                    outcome_learner=base(random_state),
                    effect_learner=_reg(random_state),
                )
        else:  # pragma: no cover - guarded by make_learner
            raise ValueError(f"{name!r} is not a causalml meta-learner")
        self._is_classifier = not continuous

    def fit(self, X, t, y) -> None:
        # Keyword args keep this correct across causalml versions: v1.0 swaps
        # the *positional* order of treatment and y, but the names stay fixed.
        self._model.fit(X, treatment=t, y=y)

    def predict(self, X) -> np.ndarray:
        out = self._model.predict(X)
        return np.asarray(out).ravel()

    def effect_interval(self, X, alpha=0.05):
        return None


class _EconmlModel:
    """Adapter for econml's DR-learner and causal forest.

    Both fit as ``fit(Y, T, X=X)`` and report ``effect(X)``. Both also support
    ``effect_interval``, which returns ``(lower, upper)`` — the point estimate
    is *not* included, so callers needing both must call ``effect()``. That is
    why both are useful on Day 3: they give effect *intervals*, which the
    Day-4 optimizer needs to report profit with uncertainty.
    """

    def __init__(self, name: str, random_state: int, continuous: bool, n_jobs: int = -1):
        from econml.dml import CausalForestDML
        from econml.dr import LinearDRLearner

        self.name = name
        # econml needs no classifier/regressor split for the *outcome* model in
        # these estimators; a regressor works for binary and continuous alike.
        # Both estimators are fit with inference, so both yield intervals.
        self.supports_intervals = True
        if name == "dr":
            # LinearDRLearner's model_propensity predicts P(T=1|X) — always
            # binary, so it is a classifier (econml warns if given a regressor).
            from sklearn.ensemble import RandomForestClassifier

            self._model = LinearDRLearner(
                model_regression=_reg(random_state),
                model_propensity=RandomForestClassifier(
                    n_estimators=100, min_samples_leaf=10, random_state=random_state
                ),
                random_state=random_state,
            )
        elif name == "forest":
            # CausalForestDML defaults to discrete_treatment=False, where
            # model_t *regresses* T (0/1 as a continuous value). Passing a
            # classifier here raises "Cannot use a classifier as a first stage
            # model when the target is continuous" — so model_t stays a regressor.
            self._model = CausalForestDML(
                model_y=_reg(random_state),
                model_t=_reg(random_state),
                n_estimators=200,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        else:  # pragma: no cover - guarded by make_learner
            raise ValueError(f"{name!r} is not an econml estimator")
        self._continuous = continuous

    def fit(self, X, t, y) -> None:
        self._model.fit(y, t, X=X)

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.effect(X)).ravel()

    def effect_interval(self, X, alpha=0.05):
        lower, upper = self._model.effect_interval(X, alpha=alpha)
        return np.asarray(lower).ravel(), np.asarray(upper).ravel()


class _ResponseBaseline:
    """The Day-2 baseline: a classifier on **treated customers only**.

    It predicts ``P(buy | X, treated)``, so its score tracks *purchasability*,
    not persuadability. It is included in every table because it is the bar an
    uplift model must clear — and because it is often embarrassingly competitive
    on outcome-ranking metrics (notebook 03).
    """

    name = "response"
    supports_intervals = False

    def __init__(self, random_state: int, continuous: bool):
        self._model = _reg(random_state) if continuous else _clf(random_state)
        self._continuous = continuous

    def fit(self, X, t, y) -> None:
        mask = np.asarray(t) == 1
        self._model.fit(np.asarray(X)[mask], np.asarray(y)[mask])

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        if self._continuous:
            return np.asarray(self._model.predict(X)).ravel()
        # Rank by P(buy | treated). Higher already means "more likely to buy",
        # which is the same direction as an uplift score (higher = target), so
        # no sign flip is needed. np.asarray first: LightGBM's stub types
        # predict_proba as a list, which mypy will not index with [:, 1].
        return np.asarray(self._model.predict_proba(X))[:, 1]

    def effect_interval(self, X, alpha=0.05):
        return None


def make_learner(
    name: LearnerName,
    *,
    target: str = "visit",
    random_state: int = 42,
) -> UpliftModel:
    """Build an unfitted learner by name.

    Parameters
    ----------
    name:
        One of :data:`LEARNER_NAMES`.
    target:
        ``"visit"``/``"conversion"`` (binary) or ``"spend"`` (continuous). This
        switches base learners between classifiers and regressors where the
        library requires it.
    random_state:
        Seeded into every LightGBM base learner and both econml estimators, so
        a benchmark run is reproducible.
    """
    if name not in LEARNER_NAMES:
        raise ValueError(f"unknown learner {name!r}; choose from {LEARNER_NAMES}")
    if target not in VALID_TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose from {sorted(VALID_TARGETS)} "
            "('spend' switches the base learners to regressors)"
        )
    continuous = target == "spend"
    if name == "response":
        return _ResponseBaseline(random_state, continuous)
    if name in {"s", "t", "x"}:
        return _CausalmlModel(name, random_state, continuous)
    return _EconmlModel(name, random_state, continuous)


def fit_learner(
    name: LearnerName,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    target: str = "visit",
    random_state: int = 42,
) -> UpliftModel:
    """Convenience: build and fit in one call, returning the fitted model."""
    model = make_learner(name, target=target, random_state=random_state)
    model.fit(X, t, y)
    return model
