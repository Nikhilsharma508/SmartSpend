"""Tests for the figure-saving convention in ``packages/plotting.py``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — no window is opened during tests

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from packages.plotting import FIGURES_DIR, figure_markdown, figures_dir, save_fig


def _make_fig() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_save_fig_writes_named_png(tmp_path, monkeypatch):
    monkeypatch.setattr("packages.plotting.FIGURES_DIR", tmp_path)
    fig = _make_fig()
    path = save_fig(fig, "unit_test_plot")
    plt.close(fig)

    assert path == tmp_path / "unit_test_plot.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_fig_rejects_bad_names(tmp_path, monkeypatch):
    monkeypatch.setattr("packages.plotting.FIGURES_DIR", tmp_path)
    fig = _make_fig()
    with pytest.raises(ValueError, match="snake_case"):
        save_fig(fig, "Day 2 Qini")
    plt.close(fig)


def test_save_fig_records_caption(tmp_path, monkeypatch):
    monkeypatch.setattr("packages.plotting.FIGURES_DIR", tmp_path)
    fig = _make_fig()
    save_fig(fig, "unit_test_captioned", "a straight line")
    plt.close(fig)

    md = figure_markdown("unit_test_captioned")
    assert md == "![a straight line](figures/unit_test_captioned.png)"


def test_figures_dir_creates_and_returns():
    d = figures_dir(create=True)
    assert d == FIGURES_DIR
    assert d.exists()


def test_figures_dir_is_inside_the_repo():
    """Regression: the path must resolve *inside* the repo, not its parent.

    ``packages/plotting.py`` sits one level below the root, so the root is
    ``parents[1]``. Using ``parents[2]`` (as ``packages/datasets/hillstrom.py``
    legitimately does, being one level deeper) silently wrote figures to a
    directory outside the repo — which no other test caught, because they
    monkeypatched ``FIGURES_DIR``. Assert against the real, unpatched value.
    """
    from packages import plotting

    root = Path(plotting.__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists(), "expected parents[1] to be the repo root"
    assert root / "docs" / "figures" == plotting.FIGURES_DIR
    assert root in plotting.FIGURES_DIR.parents
