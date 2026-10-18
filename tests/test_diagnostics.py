"""Diagnostics on synthetic data -- runs without the raw CSV.

Guards the non-numeric handling in ``smd_table``: notebook 01 reads the raw CSV,
where ``zip_code``/``channel`` arrive as pandas ``string`` dtype rather than
``category``, so the level expansion must key off "is numeric", not "is category".
"""

from __future__ import annotations

import pandas as pd

from packages.causal.diagnostics import ate_table, smd_table


def _frame(dtype: str) -> pd.DataFrame:
    """A tiny treated/control frame with a string column cast to ``dtype``."""
    return pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1, 0, 1],
            "recency": [1, 5, 2, 6, 3, 4],
            "zip_code": pd.Series(["Urban", "Rural", "Urban", "Suburban", "Rural", "Urban"])
            .astype(dtype),
            # treated rows (2, 3, 5) -> mean 2/3; control rows (0, 1, 4) -> mean 1/3
            "visit": [0, 1, 1, 0, 0, 1],
        }
    )


def test_smd_works_on_uncast_string_columns():
    table = smd_table(_frame("string"), ["recency", "zip_code"])
    assert {"recency", "zip_code=Urban", "zip_code=Rural", "zip_code=Suburban"} <= set(
        table["feature"]
    )


def test_smd_works_on_category_columns():
    table = smd_table(_frame("category"), ["recency", "zip_code"])
    assert "zip_code=Urban" in set(table["feature"])


def test_ate_uses_proportion_interval_for_binary_outcome():
    table = ate_table(_frame("string"), ["visit"]).set_index("outcome")
    assert table.loc["visit", "test"] == "two-proportion z"
    assert table.loc["visit", "ate"] == 1 / 3
