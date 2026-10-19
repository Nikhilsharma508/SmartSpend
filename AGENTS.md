# AGENTS.md

UpliftIQ (working title) — uplift/CATE modelling, a budget optimizer, and a grounded
GenAI campaign copilot. The installed distribution name is **`smartspend`**
(`pyproject.toml`), but prose and docs call the project UpliftIQ. It is also not an
importable package: `import smartspend` fails. Code lives in `packages/`.

`plan.md` and `description.md` at the root are **planning inputs**, and both are partly
stale (see "Decisions that override the docs"). Treat this file and the code as current.

## Commands

Run everything from the repo root with `uv`.

```bash
uv sync                                        # create .venv, install (CI uses --frozen)
uv run python pipelines/download.py            # fetch raw CSV if missing, then verify
                                               #   --verify-only: no network; --force: re-download
uv run ruff check .                            # lint (line-length 100)
uv run mypy packages pipelines                 # typecheck (scope is packages + pipelines only)
uv run pytest                                  # all tests (skip, not fail, if raw CSV absent;
                                               #   first run ~35s, warm runs ~2s)
uv run pytest tests/test_hillstrom.py::test_raw_shape_and_columns   # single test
uv run jupyter lab                             # open research/notebooks/
uv add <pkg>        /  uv add --dev <pkg>      # change deps — never hand-edit pyproject deps
```

Order matters in CI: `ruff check .` → `mypy packages pipelines` → `pytest`.

## Git

- **Notebooks use `nbstripout` as a repo-local git filter** (`.git/info/attributes` + local
  git config). Outputs and environment metadata are stripped on commit, so never commit
  outputs and never clear them by hand. `nbstripout --install` drops the `--extra-keys`
  flag, so if the filter is ever reinstalled re-set the clean command to
  `"<repo>/.venv/bin/python3" -m nbstripout --extra-keys "metadata.kernelspec metadata.language_info"`.
- **After executing a notebook, `git status` lists it as modified while `git diff` is
  empty.** That is expected, not a bug to chase.

## Where things live

- **Hillstrom dataset facts** (columns, quirks, checksum, leakage rules) live in
  [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md) — read it instead of
  rediscovering the data. The loader that encodes those facts (`FEATURES`,
  `LEAKY_COLUMNS`, `RAW_SHA256`) is [`packages/datasets/hillstrom.py`](packages/datasets/hillstrom.py);
  use it rather than `sklift.datasets.fetch_hillstrom`. Its Appendix A holds the original
  challenge prose verbatim — do not re-wrap that block.
- **Figures**: every plot a notebook produces goes through
  [`packages/plotting.py`](packages/plotting.py) `save_fig(fig, name)` into
  `docs/figures/<name>.png` (committed), named in snake_case so it reads as a filename.
  Read them straight from that folder to review results; notebooks link back to each file.
- **Concepts and every measured number** live in [`docs/concepts.md`](docs/concepts.md).
- **Profit assumptions** live in [`configs/business_assumptions.yaml`](configs/business_assumptions.yaml)
  (single source; the copilot reads it) and mirror into the README profit table.
- **Python 3.11 is pinned** (`.python-version`). Do not use the system `python3` (3.13):
  `causalml` and `econml` have no 3.13 support, which is why the pin exists.
- **`packages/` is a plain directory tree, not an installed distribution.** `import
  packages.*` only resolves when the CWD is the repo root (pytest sets `pythonpath`, the
  notebook bootstraps `sys.path`). Running a script from elsewhere fails with
  `ModuleNotFoundError: No module named 'packages'`.
- **Most of the skeleton is empty stubs.** Real code exists only in `packages/datasets/`,
  `packages/causal/diagnostics.py`, `packages/plotting.py`, `pipelines/download.py`, and
  `tests/`. `packages/causal/` (rest), `packages/decision/`, `packages/copilot/`, `apps/`,
  `infra/`, `evals/` are placeholders. Do not assume planned modules exist.
- `data/raw/` and `data/processed/` are git-ignored (`.gitkeep` only); the raw CSV is
  fetched and checksum-verified by `pipelines/download.py`.

## Decisions that override the docs

- **Profit is reported in USD only.** Ignore the ₹/INR framing throughout `description.md`;
  there is no conversion step.
- **Day 6 UI is FastAPI + React** (as in `description.md`). `plan.md`'s "Streamlit campaign
  planner" is superseded.
- **Profit assumptions are decided** (30% margin, $0.10/contact, value per
  conversion $34.91) — see `configs/business_assumptions.yaml`; change only with a
  recorded reason.
- Scikit-uplift's PyPI name is `scikit-uplift`; the import name is `sklift`.

## Conventions

- Notebooks are research only; anything reused moves into `packages/`.
- **Keep `docs/concepts.md` in sync.** When a notebook or script introduces a term,
  a metric or a result, add it to that file in the same change — definitions in the
  numbered sections, values in the findings ledger. It is the canonical concept +
  numbers doc; notebooks only explain briefly and link back.
- `research/notebooks/` is split by concern: `00_hillstrom_eda` is descriptive only,
  `01_hillstrom_randomization_ate` introduces the causal vocabulary (arm, SMD, ATE).
  Keep causal concepts out of `00`.
- ruff ignores `E402`/`E401`/`I001` in `*.ipynb` because cells bootstrap `sys.path` before
  importing project packages — do not "fix" that ordering.
- mypy runs with `ignore_missing_imports = true` and `files = ["packages", "pipelines"]`;
  tests and notebooks are intentionally unchecked.
