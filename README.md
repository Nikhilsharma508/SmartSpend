# UpliftIQ — Causal Promo Targeting with a GenAI Campaign Copilot

Decide **who should get a discount** by estimating each customer's *incremental*
response to it (CATE / uplift), then let a marketer plan a campaign in plain
language through a grounded GenAI copilot.

> Working name: **UpliftIQ** (alternatives: CausalEdge, Incremental Profit Engine).
> Repo directory: `SmartSpend`.

## The problem in one line

Standard response models (click / conversion prediction) rank **"sure things"**
highest — customers who would have bought anyway — so budget is spent on discounts
that change nothing. An uplift model ranks **persuadables** highest instead.

| Group | Buys without promo? | Buys with promo? | Target? |
| --- | --- | --- | --- |
| Persuadables | No | Yes | **Yes** — the only incremental-value group |
| Sure things | Yes | Yes | No — wasted margin |
| Lost causes | No | No | No — wasted contact cost |
| Sleeping dogs | Yes | No | No — the promo causes harm |

## Project status

**Days 1–2 complete.**

Day 1 — data, scaffold, environment, experiment validation.

- [x] Repo skeleton, `uv` environment with pinned versions
- [x] Structured dataset spec — [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md)
- [x] Concept reference — [`docs/concepts.md`](docs/concepts.md)
- [x] Hillstrom loader (`packages/datasets/hillstrom.py`) + experiment diagnostics
- [x] EDA notebook — [`research/notebooks/01_hillstrom_eda.ipynb`](research/notebooks/01_hillstrom_eda.ipynb)
- [x] Randomization + ATE notebook — [`research/notebooks/02_hillstrom_randomization_ate.ipynb`](research/notebooks/02_hillstrom_randomization_ate.ipynb)
- [x] SMD table reviewed — 0 of 11 flagged, largest `|SMD|` 0.014 (notebook 02 §2)
- [x] ATE table (95% CI) interpreted — `visit` +7.66 pp, `conversion` +0.68 pp, `spend` +$0.77 (notebook 02 §3)
- [x] Binary arm chosen and justified — `mens`, beats Womens on every outcome (notebook 02 §1)
- [x] Profit assumptions decided — 30% margin, $0.10/contact, `v` = $34.91 (notebook 02 §4)

Day 2 — feature pipeline, uplift metrics, response baseline + S/T/X learners.

- [x] Centralized figure saving — [`packages/plotting.py`](packages/plotting.py) → `docs/figures/`
- [x] Feature pipeline + stratified splits — [`packages/causal/features.py`](packages/causal/features.py)
- [x] Uplift metrics (Qini AUC, AUUC, uplift@k) + random baseline — [`packages/causal/metrics.py`](packages/causal/metrics.py)
- [x] Qini plotting — [`packages/causal/plotting.py`](packages/causal/plotting.py)
- [x] Baselines notebook — [`research/notebooks/03_hillstrom_uplift_baselines.ipynb`](research/notebooks/03_hillstrom_uplift_baselines.ipynb)
- [x] Qini comparison + interpretation — X-learner leads narrowly (0.0236), S-learner shrinks to ~0, response baseline is competitive (notebook 03 §4–5)

Days 3–7: see [`plan.md`](plan.md).

## Datasets

| Dataset | Rows | Treatment | Outcomes | Role |
| --- | --- | --- | --- | --- |
| [Hillstrom MineThatData](docs/datasets/hillstrom.md) | 64,000 | Mens / Womens e-mail / none | visit, conversion, spend | **Primary** — profit, explainability, copilot |

Pinned raw Hillstrom SHA-256:
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). Python is pinned to **3.11**
(`causalml` / `econml` do not yet support 3.13).

```bash
uv sync                              # create .venv and install everything
uv run python pipelines/download.py  # fetch the raw Hillstrom CSV if missing, then verify
uv run jupyter lab                   # open research/notebooks/
```

Run the tests and linters:

```bash
uv run pytest
uv run ruff check .
uv run mypy packages pipelines
```

## Profit assumptions

Stated openly, since every profit figure depends on them. **Decided on Day 1**
(notebook 02 §4; rationale in `configs/business_assumptions.yaml` — change only
with a recorded reason).

| Assumption | Value | Notes |
| --- | --- | --- |
| Reporting currency | USD ($) | Hillstrom spend is already USD |
| Gross margin on incremental spend | 30 % | retail gross-margin placeholder — assumption, not observed |
| Cost per contact (e-mail) | 0.10 USD | assumed fixed send cost per customer |
| Value per conversion (`v`) | 34.91 USD | $116.36 mean spend per converter × 30% margin — the conversion-route profit check |

Formula used throughout:

```text
profit(k) = v · Δconv(k) − c · n(k)
```

`v` = value per conversion, `Δconv(k)` = incremental conversions from targeting the
top k, `c` = cost per contact, `n(k)` = number targeted. `v` and `c` must share a
currency. Edit `configs/business_assumptions.yaml` — the copilot reads it.

## Headline results

**Day-1 baseline (notebook 02):** uniformly contacting the `mens` arm earns
≈ **+$0.13 per contact** (+$131–$138 per 1,000 contacted, after send costs) at
the assumptions above — the number every Day-2 model must beat.

**Day 2 (notebook 03):** ranking models on `visit`, test split. The honest
headline: uplift models do **not** cleanly beat the response baseline here, and
the signal is weak overall. The X-learner leads narrowly; the S-learner collapses
toward random. Full table and reading in
[`docs/concepts.md`](docs/concepts.md).

| Metric | Response model (baseline) | Best uplift model (X-learner) |
| --- | --- | --- |
| Qini AUC | 0.0199 | **0.0236** |
| uplift@30% (incremental visits/customer) | 0.0899 | **0.1015** |
| Incremental conversions at 30% targeted | — (Day 3) | — (Day 3) |
| Incremental profit per $10k spend | — (Day 4) | — (Day 4) |
| Copilot numeric faithfulness (% figures traced to tools) | — | — (Day 5–6) |

## Repo structure

```text
.
├── packages/          # importable Python: datasets, causal, decision, copilot
├── pipelines/         # download / train / evaluate scripts
├── research/notebooks # hands-on analysis (notebooks are research only)
├── data/raw           # git-ignored; verified by pipelines/download.py
├── data/processed     # git-ignored; regenerable Parquet
├── configs/           # business assumptions and model config YAML
├── docs/              # dataset specs, model card, ADRs
├── evals/             # copilot eval cases + runner (Day 5–6)
├── apps/              # FastAPI service + React frontend (Day 6)
├── infra/             # Docker / compose
└── .github/workflows/ # CI: lint, types, tests, eval smoke run
```

Boundaries that keep it honest: notebooks are research only; reusable code lives in
`packages/`; the API only ever loads a registered model, never trains.

## Risks and limitations

| Risk | Mitigation |
| --- | --- |
| Profit depends on assumed value and cost | State assumptions here + in every brief; show a range |
| Hillstrom spend is USD | Report in USD throughout; no conversion step |
| Results come from past experiments, not live deployment | Frame as "estimated on held-out randomized data" |
| Zip code can proxy for protected attributes | Router policy check + fairness note on targeting rates |
| An LLM could state unsupported numbers or causal claims | Grounding validator; causal-language judge rubric |

## References

- scikit-uplift docs — <https://www.uplift-modeling.com/>
- MineThatData Hillstrom challenge (original prose: [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md), Appendix A)
