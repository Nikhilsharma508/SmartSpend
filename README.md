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

**Day 1 (in progress)** — data, project scaffold, environment, Day-1 notebook.

- [x] Repo skeleton, `uv` environment with pinned versions
- [x] Structured dataset spec — [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md)
- [x] Concept reference — [`docs/concepts.md`](docs/concepts.md)
- [x] Hillstrom loader (`packages/datasets/hillstrom.py`) + experiment diagnostics
- [x] EDA notebook — [`research/notebooks/01_hillstrom_eda.ipynb`](research/notebooks/01_hillstrom_eda.ipynb)
- [x] Randomization + ATE notebook — [`research/notebooks/02_hillstrom_randomization_ate.ipynb`](research/notebooks/02_hillstrom_randomization_ate.ipynb)
- [ ] **TODO [YOU]** SMD table reviewed; flags explained
- [ ] **TODO [YOU]** ATE table (95% CI) reviewed and interpreted
- [ ] **TODO [YOU]** Binary arm chosen and justified
- [ ] **TODO [YOU]** Profit assumptions confirmed in `configs/business_assumptions.yaml`

Days 2–7: see [`plan.md`](plan.md).

## Datasets

| Dataset | Rows | Treatment | Outcomes | Role |
| --- | --- | --- | --- | --- |
| [Hillstrom MineThatData](docs/datasets/hillstrom.md) | 64,000 | Mens / Womens e-mail / none | visit, conversion, spend | **Primary** — profit, explainability, copilot |
| [Criteo Uplift v2.1](https://arxiv.org/abs/2111.10106) | 13,979,592 | Ad exposure (85/15) | visit, conversion | Optional, Day 7 — scale benchmark |

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

Stated openly, since every profit figure depends on them. **Decide these on Day 1.**

| Assumption | Value | Notes |
| --- | --- | --- |
| Reporting currency | USD ($) | Hillstrom spend is already USD |
| Gross margin on incremental spend | _TBD_ % | spend → profit |
| Cost per contact (e-mail) | _TBD_ USD | fixed send cost per customer |
| Value per conversion (Criteo only) | _TBD_ USD | Criteo records no spend |

Formula used throughout:

```text
profit(k) = v · Δconv(k) − c · n(k)
```

`v` = value per conversion, `Δconv(k)` = incremental conversions from targeting the
top k, `c` = cost per contact, `n(k)` = number targeted. `v` and `c` must share a
currency. Edit `configs/business_assumptions.yaml` — the copilot reads it.

## Headline results

Filled in as the build progresses (Day 2 onwards).

| Metric | Response model (baseline) | Best uplift model |
| --- | --- | --- |
| Qini AUC | | |
| Incremental conversions at 30% targeted | | |
| Incremental profit per $10k spend | | |
| Copilot numeric faithfulness (% figures traced to tools) | — | |

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
| Criteo features are anonymized → no explainable segments | Explain on Hillstrom; use Criteo for scale only |
| Profit depends on assumed value and cost | State assumptions here + in every brief; show a range |
| Hillstrom spend is USD | Report in USD throughout; no conversion step |
| Results come from past experiments, not live deployment | Frame as "estimated on held-out randomized data" |
| Zip code can proxy for protected attributes | Router policy check + fairness note on targeting rates |
| An LLM could state unsupported numbers or causal claims | Grounding validator; causal-language judge rubric |

## References

- Diemert et al., *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling* — <https://arxiv.org/abs/2111.10106>
- scikit-uplift docs — <https://www.uplift-modeling.com/>
- MineThatData Hillstrom challenge (original prose: [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md), Appendix A)
