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
uv run python pipelines/download.py            # verify raw CSV SHA-256; --fetch to download
uv run ruff check .                            # lint (line-length 100)
uv run mypy packages pipelines                 # typecheck (scope is packages + pipelines only)
uv run pytest                                  # all tests
uv run pytest tests/test_hillstrom.py::test_raw_shape_and_columns   # single test
uv run jupyter lab                             # open research/notebooks/
uv add <pkg>        /  uv add --dev <pkg>      # change deps — never hand-edit pyproject deps
```

Order matters in CI: `ruff check .` → `mypy packages pipelines` → `pytest`.

## Non-obvious structure

- **Python 3.11 is pinned** (`.python-version`). Do not use the system `python3` (3.13):
  `causalml` and `econml` have no 3.13 support, which is why the pin exists.
- **`packages/` is a plain directory tree, not an installed distribution.** `import
  packages.*` only resolves when the CWD is the repo root (pytest sets `pythonpath`, the
  notebook bootstraps `sys.path`). Running a script from elsewhere fails with
  `ModuleNotFoundError: No module named 'packages'`.
- **Most of the skeleton is empty stubs.** Real code exists only in `packages/datasets/`,
  `packages/causal/diagnostics.py`, `pipelines/download.py`, and `tests/`.
  `packages/causal/` (rest), `packages/decision/`, `packages/copilot/`, `apps/`, `infra/`,
  `evals/` are placeholders. Do not assume planned modules exist.
- `configs/business_assumptions.yaml` is the single source of profit assumptions; the
  copilot is meant to read it. Its `margin_rate` / `cost_per_contact_usd` are still
  `# DECIDE`.
- `data/raw/` and `data/processed/` are git-ignored (`.gitkeep` only). The raw CSV is
  checksum-pinned in `packages/datasets/hillstrom.py` (`RAW_SHA256`).

## Hillstrom data gotchas

Verified against the committed raw file; each one has bitten or would bite an agent:

- There is **no `exposure` column** — that pitfall is Criteo-only. Treatment is `segment`.
- **`spend > 0` iff `conversion == 1`** (exactly 578 rows). The loader raises if this breaks.
- `history_segment` buckets 6 and 7 **contain commas** (`"6) $750 - $1,000"`). Parse with a
  real CSV reader; naive comma-splitting silently corrupts those rows.
- The source has a typo, **`Surburban`** (28,776 rows); the loader normalizes it to
  `Suburban` by default.
- Never let `segment`, `visit`, `conversion` or `spend` into the feature matrix
  (`LEAKY_COLUMNS`). `history_segment` is also excluded from `FEATURES` (collinear with
  `history`).
- **Use the project's loader, not `sklift.datasets.fetch_hillstrom`.** The sklift one
  downloads over the network by default; ours is offline, checksum-verified, and keeps all
  three raw arm labels. Avoid "de-duplicating" the two.
- Tests **skip** (not fail) when the raw CSV is absent. The first `pytest` run takes ~35s
  (lightgbm/scipy import cost); warm runs are ~2s.

## Decisions that override the docs

- **Profit is reported in USD only.** Ignore the ₹/INR framing throughout `description.md`;
  there is no conversion step.
- **Day 6 UI is FastAPI + React** (as in `description.md`). `plan.md`'s "Streamlit campaign
  planner" is superseded.
- Scikit-uplift's PyPI name is `scikit-uplift`; the import name is `sklift`.

## Conventions

- Notebooks are research only; anything reused moves into `packages/`.
- ruff ignores `E402`/`E401`/`I001` in `*.ipynb` because cells bootstrap `sys.path` before
  importing project packages — do not "fix" that ordering.
- mypy runs with `ignore_missing_imports = true` and `files = ["packages", "pipelines"]`;
  tests and notebooks are intentionally unchecked.
- `docs/datasets/hillstrom.md` is the verified dataset reference; Appendix A holds the
  original challenge prose verbatim. Do not re-wrap that block.
