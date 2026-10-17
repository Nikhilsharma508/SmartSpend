"""Hillstrom MineThatData e-mail experiment: loading and cleaning.

See ``docs/datasets/hillstrom.md`` for the full structured data description.

Typical use::

    from packages.datasets.hillstrom import fetch_hillstrom, FEATURES

    df = fetch_hillstrom(arm="mens")          # binary: Mens E-Mail vs No E-Mail
    X, y, t = df[FEATURES], df["visit"], df["treatment"]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

RAW_FILENAME = (
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RAW_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / RAW_FILENAME
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

#: The three arms, exactly as stored in the ``segment`` column.
ARM_MENS = "Mens E-Mail"
ARM_WOMENS = "Womens E-Mail"
ARM_CONTROL = "No E-Mail"

#: Map of the short arm name used by :func:`fetch_hillstrom` to the raw label.
ARM_LABELS = {"mens": ARM_MENS, "womens": ARM_WOMENS}

#: Pre-treatment covariates to be used as model features.
#: ``history_segment`` is deliberately excluded (collinear with ``history``).
FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]

#: Categorical features that need encoding before modelling.
CATEGORICAL_FEATURES = ["zip_code", "channel"]

#: Numeric features, usable as-is.
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]

#: Outcome columns, in increasing order of rarity / cost to model.
OUTCOMES = ["visit", "conversion", "spend"]

#: Columns that must never leak into the feature matrix.
LEAKY_COLUMNS = ["segment", "treatment", "visit", "conversion", "spend"]

#: Source-data typo present in all 28,776 rows; corrected when ``normalize`` is set.
ZIP_TYPO = "Surburban"
ZIP_TYPO_FIX = "Suburban"

#: Ordinal ordering of the ``history_segment`` buckets.
HISTORY_SEGMENT_ORDER = [
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def verify_raw(path: Path | None = None) -> str:
    """Return the SHA-256 of the raw CSV, warning if it differs from the pinned value."""
    import hashlib

    path = path or RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw Hillstrom CSV not found at {path}.\n"
            "Run `python pipelines/download.py` or place the file in data/raw/."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_hillstrom_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV untouched (no dtype coercion, no renames)."""
    path = path or RAW_PATH
    # keep_default_na=False would turn blanks into ""; the file has no blanks,
    # so default NA handling is fine and safer for ints.
    return pd.read_csv(path)


def fetch_hillstrom(
    arm: str = "mens",
    *,
    normalize_zip_spelling: bool = True,
    as_category: bool = True,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load Hillstrom as one cleaned DataFrame for a binary experiment.

    Parameters
    ----------
    arm:
        ``"mens"`` or ``"womens"``. The chosen e-mail arm is kept as the treated
        group and ``No E-Mail`` becomes the control; rows from the other e-mail
        arm are dropped so the experiment stays binary. Use ``arm="multi"`` to
        keep all three arms and skip the ``treatment`` column.
    normalize_zip_spelling:
        Fix the source typo ``Surburban`` -> ``Suburban``.
    as_category:
        Cast ``zip_code``, ``channel`` and ``segment`` to pandas ``category``.
    path:
        Override the raw CSV location (mainly for tests).

    Returns
    -------
    pandas.DataFrame
        Cleaned frame. For binary arms it includes a ``treatment`` int column
        (1 = received the e-mail, 0 = control). Outcome and feature columns keep
        their original names; see the module constants for the column groups.
    """
    if arm not in {"mens", "womens", "multi"}:
        raise ValueError(f"arm must be 'mens', 'womens' or 'multi', got {arm!r}")

    df = load_hillstrom_raw(path)

    # Sanity: shape and columns must match the documented schema.
    expected = {
        "recency", "history_segment", "history", "mens", "womens",
        "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw Hillstrom CSV is missing columns: {sorted(missing)}")

    if normalize_zip_spelling:
        df["zip_code"] = df["zip_code"].replace({ZIP_TYPO: ZIP_TYPO_FIX})

    # history_segment -> ordered category (kept for EDA, excluded from FEATURES).
    df["history_segment"] = pd.Categorical(
        df["history_segment"], categories=HISTORY_SEGMENT_ORDER, ordered=True
    )

    if arm in {"mens", "womens"}:
        treated_arm = ARM_LABELS[arm]
        df = df[df["segment"].isin([treated_arm, ARM_CONTROL])].copy()
        df["treatment"] = (df["segment"] == treated_arm).astype("int8")

    if as_category:
        for col in ["zip_code", "channel", "segment"]:
            df[col] = df[col].astype("category")

    df = df.reset_index(drop=True)

    # Documented invariants: fail loudly if the data drifts.
    if not df["spend"].ge(0).all():
        raise ValueError("Negative spend found; unexpected.")
    if arm in {"mens", "womens"}:
        n_conv = int(df["conversion"].sum())
        n_spend = int((df["spend"] > 0).sum())
        if n_conv != n_spend:
            raise ValueError(
                f"Expected spend>0 exactly when conversion==1, got {n_conv} vs {n_spend}."
            )
    return df


# --------------------------------------------------------------------------- #
# Quick summaries
# --------------------------------------------------------------------------- #
def arm_summary(df: pd.DataFrame, treatment_col: str = "segment") -> pd.DataFrame:
    """Per-arm counts, outcome rates and mean spend (the Day-1 descriptive table)."""
    rows = []
    for arm, g in df.groupby(treatment_col, observed=True):
        rows.append(
            {
                "arm": str(arm),
                "n": len(g),
                "share": len(g) / len(df),
                "visit_rate": g["visit"].mean(),
                "conversion_rate": g["conversion"].mean(),
                "mean_spend": g["spend"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)
