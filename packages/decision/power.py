"""Power analysis for the next experiment: what MDE can this business afford?

Uplift numbers on Hillstrom have huge intervals (Day 3: every model's profit
CI overlaps its neighbours), which is a sample-size story before it is a model
story. :func:`design_ab_test` answers the planning question a marketer
actually asks — "we want to detect a +X% lift; how many customers and how long
is that?" — with the standard two-proportion sample size, sized on the
*ranking* outcome's baseline rate (visit), and converts the total into days
against the daily eligible pool.
"""

from __future__ import annotations

from math import ceil

from scipy.stats import norm

from packages.decision.schemas import AbTestDesign


def design_ab_test(
    baseline_rate: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    daily_eligible: int = 1000,
) -> AbTestDesign:
    """Sample size and duration to detect an absolute ``mde`` lift.

    Parameters
    ----------
    baseline_rate:
        Expected outcome rate in the control arm (e.g. Hillstrom's ~15%
        visit rate).
    mde:
        Minimum detectable effect, **absolute** (``0.01`` = +1 percentage
        point). The relative figure is reported alongside.
    alpha, power:
        Two-sided significance level and desired power; the usual 0.05/0.8.
    daily_eligible:
        Total customers eligible for the experiment per day, split evenly
        between arms; duration is ``n_total / daily_eligible`` rounded up.

    Notes
    -----
    Equal allocation is assumed (the standard A/B split); unequal allocation
    would inflate the total. The formula is the classic two-proportion normal
    approximation — it assumes the baseline rate is known, so treat the output
    as an order of magnitude with headroom, not a contract.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if not mde > 0:
        raise ValueError(f"mde must be positive, got {mde}")
    if baseline_rate + mde >= 1:
        raise ValueError("baseline_rate + mde must stay below 1")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    if daily_eligible < 1:
        raise ValueError(f"daily_eligible must be >= 1, got {daily_eligible}")

    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_bar = (p1 + p2) / 2
    z_alpha = float(norm.ppf(1 - alpha / 2))
    z_power = float(norm.ppf(power))
    n_per_arm = ceil(
        (
            z_alpha * (2 * p_bar * (1 - p_bar)) ** 0.5
            + z_power * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5
        )
        ** 2
        / mde**2
    )
    n_total = 2 * n_per_arm
    return AbTestDesign(
        baseline_rate=baseline_rate,
        mde_absolute=mde,
        mde_relative=round(mde / baseline_rate, 4),
        alpha=alpha,
        power=power,
        n_per_arm=n_per_arm,
        n_total=n_total,
        daily_eligible=daily_eligible,
        duration_days=ceil(n_total / daily_eligible),
        note=(
            "Two-proportion normal approximation, two-sided alpha, equal "
            "allocation; baseline treated as known. Duration assumes the daily "
            "eligible pool is split evenly between arms."
        ),
    )
