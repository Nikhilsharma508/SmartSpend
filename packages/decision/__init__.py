"""Decision layer: budget optimizer, scenario comparison, segment profiling,
power analysis, and the model card — the 5 typed tools Day 6's API exposes.

Every tool takes holdout arrays plus the decided business assumptions and
returns a Pydantic schema from :mod:`packages.decision.schemas`; nothing here
fits a ranking model (that happens once, in ``pipelines/train.py``, and the
registered artifact is loaded, never retrained).
"""

from packages.decision.card import get_model_card
from packages.decision.power import design_ab_test
from packages.decision.profit import optimize_budget, profit_point
from packages.decision.scenarios import compare_scenarios, policy_value_ipw
from packages.decision.segments import get_segment_profile

__all__ = [
    "compare_scenarios",
    "design_ab_test",
    "get_model_card",
    "get_segment_profile",
    "optimize_budget",
    "policy_value_ipw",
    "profit_point",
]
