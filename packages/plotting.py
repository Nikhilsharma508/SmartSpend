"""One place every figure is saved through, so results are easy to re-read.

Plots are the primary evidence for modeling decisions, so they are stored as
real files rather than left inside executed notebooks. Everything lands in
``docs/figures/<name>.png`` (committed) with a descriptive snake_case name, and
each notebook links back to the file it produced.

Typical use inside a notebook::

    from packages.plotting import save_fig
    fig, ax = plt.subplots()
    ...
    save_fig(fig, "day2_qini_all_models",
             "Qini curves — every uplift model vs the response baseline and random")

The name is the filename, so it should read as a filename: lowercase, words
separated by underscores, no extension.
"""

from __future__ import annotations

import re
from pathlib import Path

from matplotlib.figure import Figure

#: Committed home for every figure. Relative to the repo root so the paths are
#: stable no matter where a notebook is opened from. ``packages/plotting.py`` is
#: one level below the root, so the repo root is ``parents[1]`` (not ``[2]`` —
#: that would be the parent *of* the repo and silently write outside it).
FIGURES_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"

#: Guard against typo-ed names producing stray files (``"day2 qini"`` etc.).
_NAME_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def figures_dir(create: bool = True) -> Path:
    """Return the figures directory, creating it if asked."""
    if create:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def save_fig(
    fig: Figure,
    name: str,
    caption: str | None = None,
    *,
    dpi: int = 150,
    transparent: bool = False,
) -> Path:
    """Save ``fig`` to ``docs/figures/<name>.png`` and return the path.

    Parameters
    ----------
    fig:
        The figure to persist. Use ``plt.gcf()`` if you have only a current
        figure handle.
    name:
        Snake_case filename stem, e.g. ``"day3_cate_calibration_deciles"``.
        The ``.png`` extension is added if missing.
    caption:
        Optional one-line description written next to the path and used by
        ``figure_markdown``. Keep it to what the plot shows, not the takeaway.
    dpi:
        Raster resolution. 150 keeps files small while staying legible in a
        diff viewer and on GitHub.
    transparent:
        Whether to drop the background colour (useful for dark-mode READMEs).
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"figure name {name!r} must be snake_case (lowercase words joined by "
            "underscores) so the filename reads as a filename"
        )

    path = figures_dir() / f"{name}.png"
    # bbox_inches="tight" drops the dead margin matplotlib adds by default, so
    # the saved PNG matches what the notebook shows.
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)

    if caption:
        _CAPTIONS[name] = caption
    return path


def figure_markdown(name: str, caption: str | None = None) -> str:
    """Return a Markdown snippet embedding the figure, for pasting in a doc.

    Renders inline on GitHub and in most Markdown viewers, and points at the
    committed file so the link keeps working.
    """
    caption = caption or _CAPTIONS.get(name, name.replace("_", " "))
    return f"![{caption}](figures/{name}.png)"


#: In-memory caption registry so a notebook can save once and reference later.
_CAPTIONS: dict[str, str] = {}
