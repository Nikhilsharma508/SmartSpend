"""Hillstrom MineThatData e-mail experiment: loading and cleaning.

See ``docs/datasets/hillstrom.md`` for the structured data description and
``docs/concepts.md`` for the vocabulary used here (arm, treatment, control).

Typical use::

    from packages.datasets.hillstrom import fetch_hillstrom, FEATURES

    df = fetch_hillstrom(arm="mens")          # binary: Mens E-Mail vs No E-Mail
    X, y, t = df[FEATURES], df["visit"], df["treatment"]
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

RAW_FILENAME = (
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

#: Canonical source of the raw CSV. The file has been byte-stable since 2008.
RAW_SOURCE_URL = "http://www.minethatdata.com/" + RAW_FILENAME

#: SHA-256 of the canonical file. Pins *content*, not just shape: a re-encoded or
#: truncated download can match on row count and column names and still be wrong.
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
#: ``history_segment`` is deliberately excluded (collinear with ``history``);
#: notebook 01 shows the correlation behind that call.
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

#: Columns present in the frame returned by :func:`fetch_hillstrom` that must
#: never be used as features: the treatment assignment (``segment``) and the
#: derived treatment indicator (``treatment``), plus the three post-treatment
#: outcomes. Notebook 01 states the evidence for each verdict.
LEAKY_COLUMNS = ["segment", "treatment", "visit", "conversion", "spend"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def raw_sha256(path: Path | None = None) -> str:
    """SHA-256 of the raw CSV, read in chunks.

    Raises ``FileNotFoundError`` if the file is absent. Chunked so a very
    large file does not have to be held in memory.
    """
    path = path or RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw Hillstrom CSV not found at {path}.\n"
            "Run `uv run python pipelines/download.py` to fetch it."
        )
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw(path: Path | None = None) -> bool:
    """True if the raw CSV matches :data:`RAW_SHA256` byte-for-byte."""
    return raw_sha256(path) == RAW_SHA256


def load_hillstrom_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV untouched (no dtype coercion, no renames)."""
    path = path or RAW_PATH
    # The file has no blank fields, so default NA handling is safe for the
    # integer indicators and does not introduce surprises.
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
        keep all three arms (no ``treatment`` column is added).
    normalize_zip_spelling:
        Fix the source typo ``Surburban`` -> ``Suburban``. The typo is surfaced
        in notebook 01; this flag exists so the fix is explicit and reversible.
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

    # Schema gate, derived from the constants above so a rename cannot drift.
    expected = set(FEATURES) | set(OUTCOMES) | {"history_segment", "segment"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw Hillstrom CSV is missing columns: {sorted(missing)}")

    if normalize_zip_spelling:
        # Source typo present in 28,776 rows; shown in notebook 01.
        df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})

    if arm in {"mens", "womens"}:
        treated_arm = ARM_LABELS[arm]
        df = df[df["segment"].isin([treated_arm, ARM_CONTROL])].copy()
        df["treatment"] = (df["segment"] == treated_arm).astype("int8")

    if as_category:
        for col in [*CATEGORICAL_FEATURES, "segment"]:
            df[col] = df[col].astype("category")

    df = df.reset_index(drop=True)

    # Documented invariants: fail loudly if the data drifts.
    if not df["spend"].ge(0).all():
        raise ValueError("Negative spend found; unexpected.")
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
    """Per-arm counts and mean outcome (the Day-1 descriptive table).

    Columns are ``arm``, ``n``, ``share`` and ``mean_<outcome>`` for every entry
    in :data:`OUTCOMES`. For the binary outcomes ``mean_visit`` and
    ``mean_conversion`` are the rates.
    """
    rows = []
    for arm, group in df.groupby(treatment_col, observed=True):
        row: dict[str, object] = {
            "arm": str(arm),
            "n": len(group),
            "share": len(group) / len(df),
        }
        for outcome in OUTCOMES:
            row[f"mean_{outcome}"] = group[outcome].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)
