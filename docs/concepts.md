# Concepts

One place for every idea this project uses and every number we measure with it.
Notebooks explain a concept briefly where it is first used and link back here;
this file is the canonical, growing version.

**Keeping it in sync.** When a notebook introduces a term, a metric or a finding,
add it here in the same change. Definitions go in the numbered sections; results
and metric values go in the [findings ledger](#findings-ledger). Anything that
only lives in a notebook cell is a bug.

Scope so far: Day 1 (experiment fundamentals, randomization, ATE) and Day 2
(response baseline, meta-learners, ranking metrics). Later days append to the
same structure — calibration, the profit curve, the copilot's grounding rules.

---

## 0. Notation

| Symbol | Meaning |
| --- | --- |
| `Y` | Outcome (here `visit`, `conversion` or `spend`) |
| `T` | Treatment indicator: `1` = treated, `0` = control |
| `Y(1)`, `Y(0)` | The two *potential outcomes* for one customer |
| `i` | Index of one customer |
| `n` | Number of customers |

---

## 1. The experiment

### Arm

An **arm** is one branch of an experiment: the group of units assigned to one
condition. The Hillstrom dataset has three arms, stored in the `segment` column:

| Arm | Raw label | Customers |
| --- | --- | --- |
| Mens e-mail | `Mens E-Mail` | 21,307 |
| Womens e-mail | `Womens E-Mail` | 21,387 |
| No e-mail | `No E-Mail` | 21,306 |

A **binary experiment** keeps one e-mail arm and drops the other, so the question
becomes "did *this* e-mail cause a change?" — the cleanest estimand, and the one
this project models. The three-arm version is a later multi-treatment extension
("which e-mail, if any?").

### Treatment and control

- **Treatment** = the arm that receives the intervention (`T = 1`).
- **Control** = the arm that does not (`T = 0`); in Hillstrom, `No E-Mail`.

`fetch_hillstrom(arm="mens")` maps `Mens E-Mail` → treated, `No E-Mail` → control,
and drops `Womens E-Mail`, so the frame is a clean two-arm comparison.

### Randomization

Randomization means the treatment was assigned **independently of how a customer
would have responded**. The customer's potential outcomes are decided before and
independently of the coin flip that assigns `T`.

This is the whole reason the project can estimate a *causal* effect from
observational-looking data: under randomization the treated and control groups are
statistically identical on every pre-treatment feature, so any difference in
outcomes afterwards is attributable to the e-mail. That assumption is **testable**
— see [SMD](#standardized-mean-difference-smd) — which is why Day 1 checks it
before any model is trained.

### Potential outcomes and the fundamental problem

For each customer there are two potential outcomes: `Y(1)` if treated, `Y(0)` if
not. We observe only one — the other is counterfactual. So the individual effect
is never observable; we estimate *averages* of it and, later, model how that
average varies with customer features.

### ITE, CATE, uplift

| Term | Definition | Observable? |
| --- | --- | --- |
| ITE — individual treatment effect | `Y_i(1) − Y_i(0)` | No, never |
| CATE — conditional average treatment effect | `E[Y(1) − Y(0) | X = x]` | Estimated |
| ATE — average treatment effect | `E[Y(1) − Y(0)]` | Estimated (the mean of CATE) |

**Uplift** is the same thing as CATE, viewed as a targeting score: the change in
outcome an intervention causes for a given customer. The project's end goal is a
per-customer uplift estimate; the ATE is the single number every model must be
worth more than.

---

## 2. Day-1 diagnostics

### Standardized mean difference (SMD)

SMD measures how far apart treated and control are on one feature, in units of
pooled spread rather than raw units, so different features are comparable:

```text
SMD = (mean_treated − mean_control) / sqrt((var_treated + var_control) / 2)
```

Why standardize: a gap of 0.5 years of recency and 0.5 dollars of history are not
equally meaningful; dividing by the pooled standard deviation puts both on the
same scale. Categorical features enter as one-vs-rest indicators (e.g.
`zip_code=Urban` = 1 for Urban customers, 0 otherwise).

The convention used here flags **|SMD| > 0.1**. Under successful randomization
every SMD is noise around zero, so a flag does not prove bias — at ~21,000 per arm
a small imbalance can even be statistically significant while remaining
practically irrelevant. The right reading is: *how many* are flagged, *how large*,
and does the pattern suggest a broken assignment. A clean randomized experiment
shows a scatter of small values, both signs, with no structure.

**Why a fixed threshold and not a p-value.** A test at this sample size is
extremely sensitive: a 50/50 binary feature with SMD = 0.03 — a 1.5-point gap
between arms, or about $8 of mean `history` — is statistically significant
(p ≈ 0.002), yet it is far too small to affect any targeting decision. With enough
data, *everything* is statistically significant. A p-value would eventually flag a
perfectly good randomization, which is why the rule uses an effect-size threshold
(0.1) that does not shrink as `n` grows.

### Average treatment effect (ATE) and confidence interval

```text
ATE = mean(Y | T = 1) − mean(Y | T = 0)
```

estimated on the same rows the model will later be scored on. The **standard
error** depends on the outcome type:

| Outcome type | Standard error | Interval |
| --- | --- | --- |
| Binary (`visit`, `conversion`) | `sqrt(p1(1−p1)/n1 + p0(1−p0)/n0)` | normal approximation (two-proportion z) |
| Continuous (`spend`) | `sqrt(s1²/n1 + s0²/n0)` | Welch's t interval |

The **95% confidence interval** is `ATE ± 1.96 · SE`. It is an interval for the
*estimate*, not a range of customer behaviour: it says where the true average
effect plausibly sits. If the interval excludes zero, the effect is
distinguishable from no effect at that confidence level.

The ATE is the **baseline every model must beat**. A model that cannot rank
customers better than a coin flip is worth exactly the ATE applied uniformly —
and random targeting has an uplift score of ~0 by construction.

---

## 3. The four customer groups

Every customer falls into one quadrant of (would they buy without the promo?) ×
(would they buy with it?). This is the lens the whole project is built around.

| Group | `Y(0)` | `Y(1)` | Effect | Target? |
| --- | --- | --- | --- | --- |
| **Persuadables** | No | Yes | + | **Yes** — only incremental value |
| **Sure things** | Yes | Yes | 0 | No — wasted margin, they'd buy anyway |
| **Lost causes** | No | No | 0 | No — wasted contact cost |
| **Sleeping dogs** | Yes | No | − | No — the promo actively harms |

A **response model** predicts `P(buy)`, which is highest for sure things — it
ranks the least useful group first. An **uplift model** predicts `Y(1) − Y(0)`,
which is highest for persuadables. That difference is the project's central
comparison.

---

## 4. Day-2 modelling and ranking metrics

### Response model (the baseline)

A **response model** is a plain supervised classifier trained on **treated
customers only** to predict `P(buy | X)`. It answers "who converts after the
e-mail?", not "who converts *because of* it?". It is the bar every uplift model
must clear, because it is simple, well-understood, and already good at finding
converters.

Its blind spot is the project's premise: it ranks **sure things** highest (high
`P(buy)`, low incremental effect). The check used in notebook 01 is to split the
scored population into terciles and compare the treated-vs-control outcome gap
in each. If the *level* rises in both arms together while the *gap* stays flat,
the score is tracking purchasability, not persuadability.

### Meta-learners (S, T, X)

Meta-learners reuse any supervised model as a building block to estimate a
per-customer uplift score. With LightGBM as the base:

| Learner | Estimation | Tendency |
| --- | --- | --- |
| **S** (single) | One model on `[X, T]`; read off the treatment interaction | Pools all data → **shrinks uplift toward zero** |
| **T** (two) | Fit treated and control separately; subtract predictions | Flexible, but each model sees only half the data |
| **X** (cross) | Impute per-customer effects from T, then fit a model on them | Strong with imbalance; the effect stage needs a *regressor* |

All three output a **relative ranking score**, not a probability — only the
order matters. The S-learner's shrinkage is not a bug: with regularisation, the
easiest way to fit `[X, T]` is to learn the dominant main effect and make the
treatment interaction tiny, which collapses the ranking toward random.

### Ranking metrics: Qini curve, AUUC, uplift@k

Uplift is never observed per customer, so ranking quality is measured by
*treated-vs-control outcomes within score buckets*:

- **Qini curve** — sort by score descending, target the top *k*, and plot the
  cumulative treated-minus-control outcome difference against targeting depth.
  Every curve **converges at depth 1.0**, because once everyone is targeted the
  total lift is fixed regardless of order. A **perfect** curve front-loads all
  incremental converters (steep rise, peak, return to the common endpoint); a
  **random** ranking sits near a shallow diagonal.
- **Qini AUC** — area under the Qini curve, normalised by the perfect curve.
  `0` = random, negative = worse than random.
- **AUUC** — area under the uplift curve (the same information in raw units).
- **uplift@k** — incremental outcome per customer when targeting only the top
  `k` fraction. This is the number that multiplies up to a campaign total, and
  the one the decision layer optimises.

These measure **ranking**, not **effect size**. A model can rank well with
poorly scaled scores (and vice versa); that is why Day 3 adds calibration, and
why the model is ultimately chosen on Qini **and** profit, not on Brier score.

---

## 5. Day-3 evaluation depth

### DR-learner (doubly robust)

A two-stage learner: first fit the propensity `P(T=1 | X)` and the two outcome
regressions `E[Y | X, T]`; combine them into an **AIPW pseudo-outcome** — an
unbiased per-customer estimate of the effect that stays consistent if *either*
the outcome model *or* the propensity model is correct (hence "doubly
robust"); then regress that pseudo-outcome on `X` for the final ranking. On a
randomised experiment the propensity is known 0.5, which removes one of the
two ways to be wrong and makes this a strong default.

### Causal forest

An ensemble that splits to **maximise treatment-effect heterogeneity**
directly, rather than outcome heterogeneity: each tree grows on orthogonalised
residuals (outcome minus propensity-weighted average, so the split criterion
targets the *difference* between arms). Leaves estimate local average
treatment effects, and the forest aggregates them with honesty (split-sample)
to keep the estimates from over-fitting their own data.

### Effect interval

A confidence interval for a per-customer effect, produced by the forest from
the spread of its trees (jackknife-over-trees). The width tells you where the
data supports heterogeneity: an interval that **excludes zero** means the
model is confident that customer is persuadable (or harmed). Caveat to keep
honest: the true ITE is never observable, so an interval can only be judged
for plausibility and aggregate coverage — not against per-customer truth.

### Brier score and reliability

- **Brier score** — `mean((p − y)²)`, the squared error of a probability
  forecast. Lower is better. Because a rare outcome makes "always predict the
  base rate" a strong competitor, the **Brier skill score**
  `1 − model / base-rate-model` says whether a model beats that null:
  `≥ 0` is better, negative means the fitted probabilities are *worse* than
  predicting one constant.
- **Reliability diagram** — bin rows by predicted probability and plot mean
  predicted vs observed rate per bin. Perfect calibration lies on the
  diagonal; `mean|gap|` summarises the average miss in probability points.
  Refinement (scores that spread out) is *not* calibration — a model can
  discriminate well (§4 ranking metrics) while its magnitudes are off.

### Recalibration

A monotone remapping from score to probability, fit on **validation** data and
applied to test: **Platt** (a logistic/sigmoid transform, 2 parameters, robust
on small samples) or **isotonic** (a non-parametric step function, more
flexible, prone to over-fitting on little data). Two invariants to remember:

- The remap must be fit on validation, never on the rows being scored —
  otherwise the score measures fit, not calibration.
- Being monotone, it **cannot change Qini or any ranking metric**; it only
  fixes magnitudes, which is what profit depends on.

### CATE calibration by decile

The ranking-metrics / calibration gap applied to *effects* rather than
probabilities: sort customers by predicted uplift, cut into deciles, compute
the treated-minus-control outcome gap **inside each decile**, and compare it
with the mean predicted effect for that decile. A model whose deciles track
the observed gaps (high rank correlation between predicted and observed) has
useful effect *magnitudes*; one that ranks well but whose decile means are
flat or inflated only supports ordering, not dollar figures.

---

## 6. Day-4 decision layer

### Champion registry

Where the *finalized* model stops being a variable and becomes an artifact:
`pipelines/train.py` fits the chosen learner under the exact Day-3 benchmark
protocol (seeded 64/16/20 stratified split, one-hot encoded on train, fit on
train only, evaluated once on the untouched test) and writes
`models/<name>.joblib` plus a `<name>.meta.json` **provenance sidecar** —
learner, seed, fit scope, data SHA-256, holdout metrics, business assumptions,
package versions, git commit. `packages/causal/registry.py` validates the
sidecar on both save and load: a model whose provenance cannot be checked is
not served. `models/` is git-ignored; artifacts are regenerable. The Day-6 API
loads a registered model and never trains.

### Profit curve and targeting depth

The champion's ranking turned into money, depth by depth: for each targeting
depth *k*, price the top-*k* cut with the profit formula (valued on the money
outcome) and plot the result — the **profit curve**. Two points get names:
the **break-even depth** (first depth whose point-estimate profit clears
zero) and the **recommended depth** (the curve's argmax). Each depth carries a
95% interval from the treated-control difference within the targeted set,
scaled to *n_targeted*; on Hillstrom these intervals are wide enough that the
curve supports "no depth is obviously underwater" but not "depth X is
optimal" — the second claim belongs to an experiment, not a split.

### Policy value and IPW (inverse-propensity weighting)

What a *policy* earns on untouched data, without pretending the experiment
ran the policy. Under uniform randomization over *A* arms, each customer's
arm has propensity 1/*A*, so the policy's value is the mean of
*A·y·1{actual arm = policy arm}* over holdout rows — customers whose
randomized arm happens to match the policy, upweighted to undo the
randomization. The incremental-vs-no-email estimate uses the paired per-row
contribution *A·y·(1{match policy} − 1{match control})*, whose sample SE is
reported. IPW buys honesty with variance (upweighting by ~*A*), which is why
"which email, if any?" comes back with wide intervals — that width is the
truth, not a bug.

### Policy tree (DRPolicyTree)

A shallow, honest decision tree trained to pick *treatments*, not to predict
outcomes: econml's `DRPolicyTree` splits for treatment heterogeneity via a
doubly-robust objective, and at depth ≤ 3 every leaf renders as a sentence a
marketer can argue with ("recency > 6 and history <= 500 → email"). The tree
only *proposes*: each segment is re-measured on the untouched holdout as
treated-minus-control within the leaf, with a 95% interval. A segment whose
interval straddles zero is unsupported by data regardless of what the tree's
leaf value claimed. (causalml's `SingleTreeCateInterpreter` from the original
plan was removed upstream in causalml 0.18; `DRPolicyTree` is its EconML
successor.)

### Split balance report (SMD)

A stratified split is a *claim* — arms and events balance across folds — and
the claim should be checkable. `split_balance_report` prints, per fold: size,
treated share, per-arm event counts and rates, the unadjusted ATE, and its
**standardized mean difference (SMD)**,
*(p_t − p_c)/√((p_t(1−p_t)+p_c(1−p_c))/2)*. Stable SMD across sibling folds
means the stratification is doing its job; drifting SMD means Qini and profit
will wobble for reasons unrelated to the model. `make_splits(stratify_extra=…)`
folds a second outcome (e.g. the rarer money outcome) into the label so the
quantity that gets *paid* is balanced too.

### Fairness audit

The same policy, sliced by a protected-ish attribute (`zip_code`): each
group's **contact rate** (share the policy would email) against its
**population share**, plus the group's own holdout uplift with an interval.
Two failure modes it surfaces: *disparate contact* (a group targeted far
above its population share without supporting evidence) and *unsupported
contact* (targeting a group whose uplift interval includes zero). Measured on
Hillstrom (notebook 03): the policy tree emails Urban at 59% against its 41%
share — a real *disparate-contact* finding driven by a `zip_code` split —
while every group's uplift interval excludes zero and overlaps its siblings,
so no *effect* disparity is supported. The model card carries both facts: the
contact disparity as a flag, the effect claim as overall-only.

### MDE, power, and experiment sizing

The **minimum detectable effect (MDE)** is the lift the business actually
cares about; **power** (usually 0.8) is the chance of detecting it at
significance α. `design_ab_test` converts the pair into a two-proportion
sample size (customers per arm) and, against the daily eligible pool, a
duration. It exists because every interval on Day 3 was wide for a
sample-size reason before it was a model reason: sizing the next test by the
business's interesting lift — not by what this dataset can barely see — is
the difference between an experiment and a hope.

### Model card

The registered model's ID plus its *decided boundaries*: provenance (data,
protocol, seed, versions), holdout metrics, the business assumptions those
metrics were priced under, a fairness note, and explicit **when-not-to-use**
lines — each one a measured result, not a generic disclaimer (spend targeting
inside its noise floor; champion-vs-DR a statistical tie; no group-level
causal claim). `get_model_card` reads the registry, so the card cannot drift
from the artifact.

---

## 7. Day-5 campaign copilot

The copilot (`packages/copilot/`) answers four marketer jobs — `plan`, `what_if`,
`explain`, `test_design` — as *briefs*: markdown prose whose every number traces
back to a Day-4 tool. Design rule: **tools compute, the writer narrates, the
validator decides** — the generative model never supplies a fact.

### The gate (router)

Three named questions ride **one** Decisions API call (fast paths stay fast):

| Question | Type | Answers |
| --- | --- | --- |
| `intent` | choice | plan / what_if / explain / test_design / out_of_scope + confidence |
| `policy_risk` | noul | targeting or excluding by a sensitive attribute or a proxy (0–1) |
| `missing_inputs` | noul | is the request so empty that only a clarifying question helps? (0–1) |

Thresholds (`configs/copilot.yaml`, calibrated on the 15 seed cases) decide the
path **in order** — decline ≥ confident redirect ≥ clarify ≥ cautious redirect ≥
brief:

| Threshold | Value | Fires |
| --- | --- | --- |
| `t_policy_risk` | 0.70 | decline — the fixed refusal text; no tools, no writer |
| `t_oos_confidence` | 0.90 | redirect — confidently `out_of_scope`, checked before clarify |
| `t_missing` | 0.60 | clarify — one named question; beats an *uncertain* out-of-scope redirect |
| `t_confidence` | 0.50 | redirect — never guess on a low-confidence choice |

Fast paths are deterministic templates: measured ~1.2 s (the gate call is the
only latency) and zero writer tokens. The missing-inputs criteria account for
the copilot's standard defaults (contact cost, margin, depth, MDE, daily
volume, a presented profit curve), so a concrete but budget-less request gets a
brief with flagged assumptions instead of a question back. When the API is
unreachable or refuses a question, a deterministic keyword router answers —
its reduced accuracy is a documented offline floor, recorded per case.

### Scenario extraction (the planner)

Rule-based parsing, never the LLM: USD amounts (`$`, commas, `k` suffix),
cost-vs-budget disambiguation near cost words, depth percentages, MDE in
percentage points, daily volume. Every filled default lands in
`assumptions_used`; every unstated money field in `missing_fields`. The planner
dispatches the intent to the Day-4 tools over a **cached holdout context**
(champion + split rebuilt once per process; the policy tree fitted lazily on
first use). What-if deltas are computed in code, not prose.

### Writers

Pluggable behind one `write(request) -> DraftResult` interface:

- **template** — deterministic render per intent; the always-on floor,
  no LLM.
- **ollama** — `gpt-oss:120b-cloud` over the local OpenAI-compatible endpoint
  (`/v1/chat/completions`, no key); the configured default.
- **neurometric** — `goal-setter` on the NEUROMETRIC gateway, Bearer auth.

The prompt contract (version-stamped in the config): every number VERBATIM
from the tool data and never *derived* (no count→share conversions, no
relative-CI arithmetic), causal-care wording, USD only, exact section headers.

### Validation (V1–V4)

Deterministic code — tolerance from `configs/copilot.yaml`:

| Rule | Checks |
| --- | --- |
| V1 number-trace | every `$`, `%`/`pp`, unit-tagged count and signed decimal matches a flattened tool/scenario/constant number within 1% relative (+ per-kind floors); fraction ↔ percent forms of the *same* number match; typography is normalized first (LLM drafts use U+2010+ dashes as minus and no-break digit grouping like `2 443`); each match becomes a `Citation` with its tool path |
| V2 units | USD only — ₹/INR/€/£/"lakh"/"crore" fail the brief |
| V3 causal language | deny-list ("will buy", "guaranteed", "because they are", …) + interval phrasing required when a tool output carries an interval |
| V4 sections | every header for the intent must appear as a markdown heading at line start |

A failing draft gets **one rewrite** with the error list attached; if the final
draft still fails, the prose ships with explicit `[UNVERIFIED: rule — detail]`
flags and `degraded=True` — the reader sees exactly what could not be verified.

### Seed eval and torture

`evals/seed_cases.yaml` (15 labelled cases) + `evals/run_seed_eval.py` score
fast-path accuracy (all cases), intent and tool-call correctness (brief cases),
validator pass@1 (first writer draft) / pass@2 (final draft after the rewrite,
briefs only), and p50/p95 latency. Notebook 05 adds a **validator torture
test**: five hand-corrupted drafts (invented `$`, a rupee amount, a causal
overclaim, a deleted section, a misquoted depth) — each must be caught by the
owning rule.

### Trace

Every request appends one JSONL row per stage (gate / plan / write / validate /
package) to gitignored `traces/` — the same schema the Day-6 eval runner reads,
so there is no second telemetry system.

---

## 8. Day-6 planner web app

The frontend (`apps/web`, launched by `npm run start` at the repo root) is a React
planner for the Day-4 decision layer and the Day-5 copilot: budget slider, profit
curve with a draggable depth marker, segment rule cards, test-design grid, copilot
chat, and CSV export of the recommended targeting plan.

- **Live API, no fallback data.** Pages read `GET /api/v1/overview` (profit curve,
  scenario comparison, segment profile, MDE grid, model card) and the copilot posts
  `POST /api/v1/copilot`; the header badge polls `GET /health` and shows `api offline`
  rather than stale numbers. The overview is computed once per API process from the
  registered champion's seed-42 holdout; boot warms the one expensive piece (a ~20 s
  `DRPolicyTree` fit) in a background thread, then the payload is cached. A retrained
  champion is picked up by restarting the API. The only snapshot in the repo is
  `apps/web/tests/overview.json`, a unit-test fixture the shipped app never imports.
- **The LLM is one box in a five-stage pipeline.** The Copilot page's collapsible
  "LLM architecture" panel mirrors the code: **gate** (Azure Decisions API classifies
  intent plus three probabilities; keyword router on failure), **plan** (Python rules
  extract budget/cost/MDE; defaults flagged), **tools** (deterministic
  `packages.decision` calls on the holdout), **write** (the LLM narrates the tool
  outputs; template floor on failure), **validate** (V1 number trace, V2 units, V3
  causal language, V4 sections; one rewrite). The writer selector sends `provider`
  with the request (`template`, `ollama`, `neurometric`); the model never computes a
  number. Azure credentials are loaded from `.env` into `os.environ` when the API is
  imported, because the gate client reads the environment directly (as the eval
  runner does).
- **Verdict shapes (question taxonomy)** — the copilot answers in one of four
  verdicts: `brief`, `clarify` (one question back, nothing granted), `decline`
  (policy refusal), `redirect` (off-scope). The Copilot page's "what can I ask"
  menu lists all seven question kinds behind them — plan / what-if /
  explain / test-size / policy decline / clarify / off-scope — each entry a
  verbatim seed question, sent to the live pipeline when clicked (the chip shows the
  verdict it targets; the answer shows the verdict the gate returned).

- **Model audit (Day 6)** — only the champion (response baseline, seed 42) is
  persisted as `joblib` + provenance sidecar via `pipelines/train.py`. The
  S/T/X/DR learners and causal forest were trained in notebooks 02/03 for the
  benchmark but never saved (re-trainable via `train.py --learner …`); the
  policy tree is refit on demand in `HoldoutContext.segment_profile` and is
  deterministic. MLflow backfill is therefore limited to the champion plus
  decision runs — the full benchmark history is not reconstructable from
  artifacts (its numbers live in the ledger below).

## Findings ledger

Accumulated results. Every number here should be reproducible from a notebook or
script, and every figure quoted in the README or a copilot brief should trace back
to this table.

### Hillstrom — Day 1

| Quantity | Value | Source |
| --- | --- | --- |
| Raw shape | 64,000 × 12 | notebook 00 |
| Arms | 21,307 Mens / 21,387 Womens / 21,306 No E-Mail | notebook 00 |
| Missing values | 0 | notebook 00 |
| Exact duplicate rows | 6,562 (10.2%) | notebook 00 |
| Spend zero-inflation | 99.1% of rows have `spend == 0` | notebook 00 |
| `spend > 0` ⟺ `conversion == 1` | 578 rows, exactly | notebook 00 |
| Spend among converters | min $29.99, mean $116.36, max $499.00 | notebook 00 |
| `zip_code` typo | `Surburban` in 28,776 rows | notebook 00 |
| Binary arm (`mens`) | 42,613 rows, 0 of 11 SMD flagged, largest \|SMD\| 0.014 | notebooks 00–02 |
| ATE `visit` (mens vs control) | +0.0766 [0.0700, 0.0832] | notebook 00 |
| ATE `conversion` (mens vs control) | +0.0068 [0.0050, 0.0086] | notebook 00 |
| ATE `spend` (mens vs control) | +$0.7698 [$0.4851, $1.0545] | notebook 00 |
| Day-1 profit baseline | ≈ +$0.13 per uniform contact (30% margin, $0.10/contact) | notebook 00 |

### Hillstrom — Day 2 (ranking models on `visit`)

Split: train 27,272 / val 6,818 / test 8,523, stratified on `treatment × visit`.
Metrics are on the **test** split. Baselines: response model (treated-only
classifier) and a seeded random ranking.

| Model | Qini AUC | AUUC | uplift@10% | uplift@30% |
| --- | --- | --- | --- | --- |
| X-learner | **0.0236** | **0.0127** | **0.1217** | **0.1015** |
| T-learner | 0.0201 | 0.0104 | 0.1026 | 0.0961 |
| Response baseline | 0.0199 | 0.0096 | 0.1187 | 0.0899 |
| Random | 0.0079 | 0.0021 | 0.0655 | 0.0713 |
| S-learner | 0.0069 | 0.0045 | 0.1055 | 0.0834 |

Findings (notebook 01):

| Observation | Reading |
| --- | --- |
| Uplift signal is weak | Qini AUCs cluster near random; heterogeneity around the +7.76 pp ATE is small vs noise. Raising capacity moved T-learner from *below* to *even with* response, so the ceiling is set by the data, not the learner. |
| S-learner shrinks to ~0 | Textbook failure: one model on `[X, T]` learns the main effect and makes the treatment interaction tiny, collapsing the ranking toward random. |
| Response baseline is competitive | Its score tracks *purchasability*, and since the e-mail lifts everyone, top-purchasability customers carry much of the total outcome. The edge disappears under **profit** optimisation (sure things cost full contact for little margin) — Day 4's job. |
| X-learner leads narrowly | Best Qini AUC (0.0236) and uplift@30% (0.1015), but the margin over response is thin — not a decisive win on one split. Motivates Day-3 DR/causal-forest challengers and the 10-seed benchmark. |

### Hillstrom — Day 3 (DR/forest, calibration, stability)

**Single split** (`RANDOM_STATE = 42`, test n = 8,523, ranking on `visit`;
notebook 02 §2). Day-3 challengers added to the Day-2 pack:

| Model | Qini AUC | uplift@10% | uplift@30% |
| --- | --- | --- | --- |
| X-learner | 0.0236 | 0.1217 | 0.1015 |
| T-learner | 0.0201 | 0.1026 | 0.0961 |
| Response baseline | 0.0199 | 0.1187 | 0.0899 |
| Causal forest | 0.0193 | 0.0901 | 0.0861 |
| DR-learner | 0.0081 | 0.1604 | 0.0725 |
| Random | 0.0079 | 0.0655 | 0.0713 |
| S-learner | 0.0069 | 0.1055 | 0.0834 |

Single split, so this table is reference only — the benchmark below is the
result of record (note DR's one-split Qini of 0.0081 ≈ random against its
10-seed mean of 0.0183: exactly the flakiness the benchmark exists to catch).

**Effect intervals — causal forest on `visit` (same split):**

| Quantity | Value |
| --- | --- |
| Point-estimate range across test | [−0.0962, 0.1999] |
| Mean 95% CI width | 0.1764 (vs ATE 0.0767) |
| Point estimates > 0 | 97.6% |
| CIs excluding 0 | 42.4% |

**CATE calibration by deciles** (decile 1 → 10 means, predicted vs observed
uplift; notebook 02 §4):

| Model | Predicted (d1 → d10) | Observed (d1 → d10) | Spearman |
| --- | --- | --- | --- |
| Causal forest | 0.009 → 0.144 | 0.067 → 0.092 | 0.49 |
| DR-learner | 0.048 → 0.134 | 0.084 → 0.160 | −0.09 |
| X-learner | −0.037 → 0.212 | 0.083 → 0.123 | 0.37 |

Predicted spreads dwarf the observed movement for forest and X — effect
*magnitudes* are over-dispersed even where ordering is decent. DR's spread
matches in size but its decile ordering is uncorrelated with what actually
happened (Spearman −0.09): single-split decile means on this data are noise.

**Outcome-model calibration (Brier, scored on own arm, test):**

| Model | Brier | Skill vs base rate | Reliability mean\|gap\| |
| --- | --- | --- | --- |
| Treated-arm outcome model | 0.15013 | −0.0051 | 0.0377 |
| Control-arm outcome model | 0.09501 | −0.0019 | 0.0343 |

Recalibration, fit on **validation**, scored on test (treated model): raw
0.15013 → isotonic 0.14800 → **Platt 0.14756**. Decision: **Platt per arm,
validation only**, applied before any dollar valuation; raw scores kept for
ranking (monotone → Qini unchanged). Negative skill means the fitted
probabilities barely beat one constant — the weak-heterogeneity story on the
probability scale.

**10-seed benchmark** — seeds 0–9; models train and rank on `visit`, **profit
is valued on `conversion`** (passed as `make_fold`'s 7th array); test
n = 8,523, k = 30% → 2,557 targeted; assumptions from
`configs/business_assumptions.yaml`.

| Model | Qini AUC | uplift@30% | incr. conv @30% | profit @30% | $/contact |
| --- | --- | --- | --- | --- | --- |
| **Response (champion)** | **0.0231 ± 0.0094** | **0.0912** | **23.6** | **$569 ± 239** | **$0.223** |
| DR-learner | 0.0183 ± 0.0091 | 0.0911 | 20.7 | $466 ± 221 | $0.182 |
| S-learner | 0.0128 ± 0.0144 | 0.0854 | 22.4 | $524 ± 268 | $0.205 |
| X-learner | 0.0046 ± 0.0161 | 0.0849 | 21.4 | $491 ± 345 | $0.192 |
| T-learner | 0.0041 ± 0.0132 | 0.0857 | 20.6 | $463 ± 317 | $0.181 |
| Causal forest | 0.0027 ± 0.0182 | 0.0767 | 21.7 | $500 ± 304 | $0.196 |
| *Uniform (Day-1 ATE)* | *≈ 0* | *0.0766* | *17.4* | *≈ $351* | *$0.137* |

Champion per the pre-registered rule: response's Qini edge over DR (0.0048)
is **smaller than its own std** (0.0094) → declared a **statistical tie**;
tie-break on profit + parsimony → **response model**. No uplift model clears
it. Qini vs a random ranking (SE of the mean over 10 seeds): response 7.8,
DR 6.4, S 2.8, X 0.9, T 1.0, forest 0.5 — **X, T and forest are
statistically at random** on `visit`. Against uniform targeting, response is
+62% per contact (~2.9 SE of the mean above the uniform profit line).

**`spend` challenger — single split** (test n = 8,523, 83 converters, test
ATE $0.96; full-arm ATE $0.77), uplift@10% / @30% in USD:

| Model | @10% | @30% |
| --- | --- | --- |
| Causal forest | 1.871 | 0.876 |
| DR-learner | 1.509 | 0.595 |
| T-learner | 0.566 | 0.636 |
| Random (this split) | 0.229 | 0.705 |
| X-learner | 0.071 | 0.095 |

**Noise floor — random ranking re-drawn over 30 fresh splits** (the bar every
single-split number above must clear):

| Target @ k | mean | std | min | max |
| --- | --- | --- | --- | --- |
| spend @ 10% | 1.0726 | **1.0095** | −0.8895 | 4.1777 |
| spend @ 30% | 0.8197 | 0.6750 | −0.5010 | 2.3850 |
| visit @ 10% | 0.0773 | 0.0216 | 0.0421 | 0.1339 |
| visit @ 30% | 0.0753 | 0.0146 | 0.0474 | 0.1104 |

Findings (notebook 02):

| Observation | Reading |
| --- | --- |
| No uplift model beats the response baseline over 10 seeds | Response tops Qini (0.0231) *and* profit ($569); the declared verdict is a statistical tie, broken toward response on profit + parsimony. Day-2's premise — response is the bar — survives its own test. |
| Day-2's X-learner lead was noise | 0.0236 on one split → 0.0046 ± 0.0161 over seeds (0.9 SE above random). Single-split champions are coin flips with good PR; that is what the benchmark exists to catch. |
| Qini and profit disagree | DR is 2nd on Qini but 5th on profit; S is 3rd on Qini but 2nd on profit. Ranking `visit` ≠ counting conversion dollars — the project rule demands both metrics for exactly this reason. |
| Calibration: usable magnitudes, no skill | Both outcome models sit at slightly negative Brier skill vs the base rate, reliability gaps ≈ 0.034–0.038; Platt improves the held-out Brier only modestly. Calibrate before valuing, but select on Qini + profit. |
| Forest intervals are wide but directional | Mean CI width 0.176 ≈ 2.3× the ATE; 42% of customers get intervals excluding zero and 97.6% of point estimates are positive — the forest says "who", not "how much", confidently. |
| Spend ordering does not survive its own floor | A random ranking on spend spans −0.89 → 4.18 at 10% across seeds (std 1.01); the forest's 1.87 sits inside that band. `visit`'s floor is ~45× tighter — which is why the benchmark runs on `visit`. Any Day-4 spend claim must carry this uncertainty. |
| Profit must be valued on the money outcome | The benchmark passes `conversion` as a 7th fold array; valuing the ranking outcome (`visit`) instead would have reported ≈ $7,900 rather than $569 on the same run — a ~14× overstatement. See the profit-formula note below. |

### Hillstrom — Day 4 (decision layer)

Numbers from notebook 03; definitions in section 6. Holdout = the champion's
seed-42 test split (n = 8,523, 81 converters) unless noted.

**Registered champion** (`models/champion.joblib`, via `pipelines/train.py`;
git-ignored, regenerable). Fit: response baseline, target `visit`, seed 42,
train-only. At k = 30% (2,557 targeted): Qini 0.0199, uplift@30% 0.0899,
incremental conversions 31.89 ± 10.49, profit **$857.60** ($0.3354/contact vs
uniform reference $435.35 = $0.170/contact). Qini/uplift reproduce notebook
04's single-split row exactly — the artifact traces to the ledger; the honest
performance envelope remains the 10-seed benchmark ($569 ± 239).

**Stratification visibility** (`split_balance_report`): SMD
0.2191/0.2191/0.2195 across train/val/test; converter shares per arm
0.64/0.16/0.20 exactly. `stratify_extra=conversion` is a **no-op on
Hillstrom** — conversion is nested inside visit (no converter-without-visitor
cells exist), so `treatment × visit` already balances converters in exact
proportion. The option matters only when the money outcome is *not* nested in
the ranking target (e.g. ranking on open-rate, paid on spend); here the
report's job is to *verify* the default, not fix it.

**Profit curve** (champion): point breaks even at depth 2%; peaks at the
ledger's 30% with $857.60 (CI [140, 1,576]) — reproducing the registry metric
to the cent. The 95% CI clears zero only at depths 4–8%, 14% and 24–32%, and
includes zero at 16 of 25 grid points; neighbouring depths wobble (8% dipping
at 10%) = single-split noise. Verdict: no depth is obviously underwater; no
optimal-depth claim is supportable from one split.

**Scenario comparison** at the 30% budget ($255.70):

| Strategy | incr. conversions | profit (95% CI) |
| --- | --- | --- |
| Champion ranking | 31.89 | $858 [140, 1,576] |
| Random targeting (null) | 19.80 | $435 [68, 803] |
| Target all | 65.98 | $1,451 [226, 2,676] — *infeasible at this budget* |
| No targeting | 0 | $0 |

Champion leads the null on the mean but the intervals overlap (the Day-3 tie,
in dollars). Target-all is **not** a losing strategy — it needs 3.3× the
budget and earns $0.17/contact vs the champion's $0.34; the tool's job is to
make the budget constraint visible, not to declare a knockout.

**Multi-treatment IPW** (three-arm dataset, test n = 12,800, conversion-valued,
policies fit on train, valued on holdout):

| Policy | incr. conversions (95% CI) | profit |
| --- | --- | --- |
| Mens for all | 48.0 [1.7, 94.3] | $396 |
| Womens for all | 42.0 [−3.6, 87.6] | $186 — CI includes zero |
| Argmax email (all) | 51.0 [4.4, 97.6] | $500 — edge over mens-all is IPW noise |
| Argmax email (top 30%) | 9.0 [−16.6, 34.7] | **−$70** — 9 conversions can't pay 3,840 contacts |

IPW buys honesty with ~3× variance; the capability ("which email, if any?",
uncertainty attached) is the deliverable, not a precision claim.

**Policy tree** (honest `DRPolicyTree`, depth 3, min leaf 500): 6 segments,
readable rules on `zip_code_Urban`, `history` (≤100.4 / ≤468.4 / >475) and
`channel_Phone`. Holdout-graded visit uplifts 0.053–0.102 — **every segment's
CI excludes zero, including the three the tree labeled "no email"**. The
capped-depth labels rank relative effect; at these economics the
profit-maximising action is email-everything (consistent with target-all
being profitable but budget-infeasible). Read the tree as an effect map.

**Fairness audit** (`zip_code`, same policy): contact disparity is **real** —
Urban 59.4% contacted vs 40.7% population share (the tree splits on
`zip_code_Urban`); Suburban 14.4% vs 44.2%; Rural 13.7% vs 15.1%. Effect
side clean: per-group uplift 0.071–0.083, all CIs exclude zero and overlap →
no group-level effect disparity supported. The contact disparity is a flagged
proxy-attribute risk handed to Day 6's adversarial eval; the model card
states both facts.

**Power analysis** (measured control visit rate 0.1062, 80% power, α = 0.05,
2,000 eligible/day):

| MDE | per arm | total | duration |
| --- | --- | --- | --- |
| +0.5pp (+5%) | 60,818 | 121,636 | 61 days |
| +1.0pp (+9%) | 15,510 | 31,020 | 16 days |
| +2.0pp (+19%) | 4,029 | 8,058 | 5 days |
| +3.0pp (+28%) | 1,857 | 3,714 | 2 days |

Day 3's wide intervals are a sample-size story before a model story; the next
test should be sized by the business's minimum interesting lift, not by what
this dataset can barely see.

**Model card**: 5 when-not-to-use rules (spend inside its noise floor;
champion-vs-DR a tie; no off-Hillstrom claims; no individual causal claims;
recalibrate before reading probabilities), 3 structural limitations, and a
fairness note that names the zip-proxy contact disparity.

| Day-4 finding | Evidence |
| --- | --- |
| The champion is now an artifact, not a variable | Registry + sidecar; notebook 03's load reproduces the ledger's single-split row (Qini 0.0199, uplift 0.0899) and the profit metric to the cent ($857.60). |
| The stratified protocol was already sound here — verification, not repair | Converter shares are exactly proportional under the default split because conversion nests inside visit; `stratify_extra` is a tested no-op on this dataset, kept for datasets where money ≠ ranking target. |
| One split cannot price a depth | 16 of 25 profit-curve CIs include zero; the wobble between neighbouring depths is noise. Break-even at 2% is the only robust depth statement. |
| Target-all is budget-feasibility's victim, not economics' | $1,451 absolute profit at 3.3× the budget; $0.17/contact vs champion $0.34. The scenario tool exists to make that trade visible. |
| "Which email" is answerable but expensive to answer honestly | IPW intervals ~3× wider; argmax-all's edge over mens-all is noise; the budget-limited multi-arm policy lost money on this split. |
| Every holdout-graded segment clears zero — the tree under-targets | Policy-tree labels rank relative effect at capped depth; at $34.91 vs $0.10, email-everything is the profit verdict. Trees are effect maps, not profit oracles. |
| The fairness audit found a real proxy-attribute risk | Urban contacted at 59% vs 41% share via a `zip_code_Urban` split; group effects overlap cleanly. Flagged in the card, escalated to Day 6's adversarial eval. |
| Day 3's uncertainty was always a sample-size story | +1pp on a 10.6% visit rate needs 15,510/arm ≈ Hillstrom's whole list; repetition (10 seeds) was the affordable substitute, experiments are the real answer. |

### Hillstrom — Day 5 (campaign copilot)

Live numbers (Decisions API + ollama `gpt-oss:120b-cloud` + NEUROMETRIC
gateway) from notebook 04 and the calibration runs; definitions in section 7.
Context = the Day-4 champion's split (n = 8,523). Fast-path and pass@ ranges
are spread across repeated live runs on the same day.

| Quantity | Value |
| --- | --- |
| Gate latency (one call, three named questions) | 1.22 s (1.2–1.8 s across runs); 578 in / 3 out tokens |
| Fast-path latency (decline / redirect / clarify) | 1.16–1.25 s, gate-only, 0 writer tokens — the "<1 s" claim measured honestly |
| Context boot (champion + split rebuild, once per process) | 0.40 s |
| First `explain` request | ~30 s total: one-time policy-tree fit (25.1 s), cached afterwards |
| Day-4 plan tools (5 runs) | 0.02–0.03 s |
| Writer latency, same brief | template <0.01 s / 166 words · ollama 3.56 s, 2,804 in / 1,015 out tokens / 268 words · neurometric 20.88 s / 850 words |
| Validator V1–V4 | <0.01 s per draft |
| Brief total (gate + plan + write + validate) | 6.4–10.0 s typical; median 8.26 s in the figure |
| Seed eval — fast-path accuracy (15 live cases) | **1.00** (0.93–1.00 across three runs) |
| Seed eval — intent accuracy / tool-call correctness (9 briefs) | **1.00 / 1.00** |
| Seed eval — validator pass@1 / pass@2 (9 briefs, live ollama) | 0.556 / 0.667 in notebook 04 (ranges 0.44–0.78 / 0.50–1.00) — the rewrite-retry is what closes first-draft gaps |
| Seed eval — latency | p50 4.93 s, p95 9.45 s |
| Torture test (5 hand-corrupted drafts) | 5/5 caught, each by the owning rule |
| Provider verdict | ollama is the default writer (passes V1–V4 at ~3.6 s); `neurometric/goal-setter` fails V1+V3 on the same brief (derived numbers, causal overclaims, 850 words) — the gateway needs a general-purpose model; the template is the always-on floor |
| Copilot tests | 64 (175 repo-wide) |

Day-5 finding table:

| Finding | Evidence |
| --- | --- |
| The gate routes the labelled taxonomy perfectly — once calibrated | Fast-path 1.00 on all three live runs *after* the questions were reworded: the first draft over-fired `missing_inputs` on 6 of 9 brief cases (0.533 accuracy) because the criteria ignored the copilot's documented defaults |
| One API call, three questions, ~1.2 s | The whole fast-path budget is the gate call; redirect/decline/clarify spend zero writer tokens |
| The validator is the safety lever, and it earns it | 5/5 torture mutations caught by the owning rule; live first drafts fail V1 often enough (0.44–0.78 pass@1) that the rewrite loop is load-bearing, not decoration |
| LLM drafts fail V1 for *typography*, not lies | "2 443 contacts" (U+202F) and "‑$155.64" (U+2011 hyphen) cite real tool numbers; normalizing before extraction fixed most live first-draft failures |
| Deriving numbers is the remaining failure mode | gpt-oss converts counts to shares and intervals to relative percentages; prompt v2 forbids it, and the rewrite with the error list usually lands it |
| Writer verdicts, measured not assumed | ollama ~3.6 s, passes V1–V4; `goal-setter` produced an 850-word brief that fails V1 and V3 on the same data — provider abstraction proved its worth by making the swap a config flip |
| The template floor is not a toy | It passes the full V1–V4 battery in <0.01 s — every degraded path in the system lands on a validator-passing brief |

### App — Day 6 (planner web app)

| Insight | Evidence |
| --- | --- |
| The planner UI computes, it does not replay | `GET /api/v1/overview` runs `optimize_budget` + `compare_scenarios` + `get_segment_profile` + `design_ab_test` on the champion's holdout once per API process; the Day-4 figures still match the ledger to the cent ($857.60 profit at 30% depth, break-even mark 2%). The payload is cached after boot's ~20 s policy-tree warm-up. |
| The app exercises the real gate, not a stand-in | Loading `.env` into `os.environ` flipped `/api/v1/copilot` from `router: keyword_fallback` to `router: decisions` (gate ≈1.3 s). The writer selector exposes template / ollama / neurometric; one live ollama explain run took 29.1 s and two drafts and still passed V1–V4, while a failing LLM degrades to the template. |
| One command boots the app | `npm run start` → `scripts/start_app.sh`: mise-pinned node 22/pnpm 10, installs web deps when missing, starts FastAPI :8000, health-checks it, then Vite :5173, with trap cleanup on exit. The script is re-runnable: it frees a stale SmartSpend listener, reuses a healthy stack, refuses a foreign port holder, and Vite runs `strictPort` so :5173 never silently becomes :5175. |
| MLflow tracks the champion and every decision run | Local SQLite backend + `mlruns/` artifacts; `pipelines/log_mlflow.py` backfills the champion (run id logged), and each `/api/v1/copilot` call logs a decision run. Setuptools pinned <71 and protobuf <5 for mlflow 1.27 compat. |
| Model audit | Only the champion (response baseline, seed 42) is persisted; S/T/X/DR/forest trained in notebooks but not saved (re-trainable via `train.py --learner`); policy tree refit on demand. MLflow backfill limited accordingly. |
| Day-6 eval (46 cases) | fast_path 0.913, intent 1.000, tool-call 1.000, validator pass@1 1.000, p50 1.15s (template writer, live gate). Ablation: keyword fallback 0.739 / 0.844 / 0.800 / 0.00s — router earns its place (+0.174 fast-path, +0.200 tools, +1.15s). |

### Profit formula

```text
profit(k) = v · Δconv(k) − c · n(k)
```

| Symbol | Meaning |
| --- | --- |
| `k` | Targeting depth (top-k customers, ranked by predicted uplift) |
| `Δconv(k)` | Incremental conversions caused by targeting the top k |
| `Δspend(k)` | Incremental spend caused by targeting the top k |
| `v` | Value per conversion |
| `c` | Cost per contact |
| `n(k)` | Number contacted |
| `margin` | Gross margin applied to incremental spend (30%) |

`v` and `c` must share a currency (USD here). On Hillstrom the same profit can
also be computed directly from `spend` — `profit(k) = margin · Δspend(k) − c · n(k)`
— because `v` was defined as `margin × mean spend per converter`; the two routes
agree (within ~5%, rounding of `Δconv`). Values live in
`configs/business_assumptions.yaml` (decided Day 1: `margin` 30%, `c` $0.10,
`v` $34.91) and mirror into the README profit-assumptions table.

**Day-3 money-outcome rule:** `Δconv(k)` must always be measured on the
*conversion* outcome, even when the model ranks on another column (`visit`).
`run_benchmark` therefore takes conversions as an extra fold array and values
profit on it while Qini/uplift stay on the ranking target — valuing `visit`
at $34.91/conversion instead overstated the same run ~14× ($7,900 → $569).
