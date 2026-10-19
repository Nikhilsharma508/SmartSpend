"""Typed schemas for the Day-4 decision layer.

Every tool in :mod:`packages.decision` speaks in these models: arrays and
assumptions go in, a Pydantic object comes out — the same object FastAPI will
serialise on Day 6 and the copilot will quote numbers from on Day 5. Making the
return types this explicit is what keeps "the app said \$569" and "the ledger
says \$569" the same claim instead of two coincidental copies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AssumptionsSchema(BaseModel):
    """The decided profit constants (mirrors ``BusinessAssumptions``)."""

    margin_rate: float
    cost_per_contact_usd: float
    value_per_conversion_usd: float


class ProfitPoint(BaseModel):
    """Profit from targeting the top-``depth`` fraction of the ranked list."""

    depth: float
    n_targeted: int
    incremental_conversions: float
    incremental_conversions_se: float
    profit_usd: float
    profit_ci_low_usd: float
    profit_ci_high_usd: float


class ProfitCurve(BaseModel):
    """Profit across targeting depths, with the break-even and optimum."""

    points: list[ProfitPoint]
    break_even_depth: float | None = Field(
        description="First depth whose point-estimate profit clears zero; None if never."
    )
    recommended_depth: float | None = Field(
        description="Depth with the highest point-estimate profit."
    )
    recommended_profit_usd: float | None
    assumptions: AssumptionsSchema
    note: str


class Scenario(BaseModel):
    """One targeting strategy evaluated at a fixed budget."""

    strategy: str
    n_targeted: int
    incremental_conversions: float
    profit_usd: float
    profit_ci_low_usd: float
    profit_ci_high_usd: float
    feasible: bool = Field(description="False when the budget cannot cover this strategy.")


class ScenarioComparison(BaseModel):
    """Champion vs baselines at the same money on the table."""

    depth: float
    budget_usd: float
    scenarios: list[Scenario]
    champion_wins: bool
    note: str


class PolicyValue(BaseModel):
    """Honest holdout value of a (multi-arm) targeting policy via IPW."""

    strategy: str
    n_customers: int
    value_rate: float = Field(description="Expected outcome per customer under the policy.")
    value_rate_se: float
    incremental_rate_vs_no_email: float
    incremental_conversions: float
    incremental_ci_low: float
    incremental_ci_high: float
    profit_usd: float
    note: str


class Segment(BaseModel):
    """One policy-tree leaf: the readable rule and what holdout data says."""

    segment_id: int
    rules: list[str]
    n_eval: int
    share: float
    policy_arm: str
    uplift_visit: float
    uplift_visit_ci_low: float
    uplift_visit_ci_high: float
    uplift_conversion: float | None = None


class FairnessRow(BaseModel):
    """Targeting rate and holdout uplift for one protected-group bucket."""

    group: str
    n_eval: int
    targeting_rate: float
    uplift_visit: float
    uplift_visit_ci_low: float
    uplift_visit_ci_high: float
    note: str


class SegmentProfile(BaseModel):
    """Policy-tree segments plus the fairness audit of the same policy."""

    segments: list[Segment]
    fairness: list[FairnessRow]
    note: str


class AbTestDesign(BaseModel):
    """Sample size and duration for the next experiment, given a realistic MDE."""

    baseline_rate: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float
    n_per_arm: int
    n_total: int
    daily_eligible: int
    duration_days: int
    note: str


class ModelCard(BaseModel):
    """The registered model's provenance, metrics, and honest boundaries."""

    name: str
    learner: str
    target: str
    trained_at: str
    train_rows: int
    data_sha256: str
    metrics: dict[str, float | int]
    assumptions: AssumptionsSchema
    fairness_note: str
    when_not_to_use: list[str]
    limitations: list[str]
