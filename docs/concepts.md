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
`P(buy)`, low incremental effect). The check used in notebook 03 is to split the
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

## Findings ledger

Accumulated results. Every number here should be reproducible from a notebook or
script, and every figure quoted in the README or a copilot brief should trace back
to this table.

### Hillstrom — Day 1

| Quantity | Value | Source |
| --- | --- | --- |
| Raw shape | 64,000 × 12 | notebook 01 |
| Arms | 21,307 Mens / 21,387 Womens / 21,306 No E-Mail | notebook 01 |
| Missing values | 0 | notebook 01 |
| Exact duplicate rows | 6,562 (10.2%) | notebook 01 |
| Spend zero-inflation | 99.1% of rows have `spend == 0` | notebook 01 |
| `spend > 0` ⟺ `conversion == 1` | 578 rows, exactly | notebook 01 |
| Spend among converters | min $29.99, mean $116.36, max $499.00 | notebook 01 |
| `zip_code` typo | `Surburban` in 28,776 rows | notebook 01 |
| Binary arm (`mens`) | 42,613 rows, 0 of 11 SMD flagged, largest \|SMD\| 0.014 | notebooks 01–02 |
| ATE `visit` (mens vs control) | +0.0766 [0.0700, 0.0832] | notebook 02 |
| ATE `conversion` (mens vs control) | +0.0068 [0.0050, 0.0086] | notebook 02 |
| ATE `spend` (mens vs control) | +$0.7698 [$0.4851, $1.0545] | notebook 02 |
| Day-1 profit baseline | ≈ +$0.13 per uniform contact (30% margin, $0.10/contact) | notebook 02 |

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

Findings (notebook 03):

| Observation | Reading |
| --- | --- |
| Uplift signal is weak | Qini AUCs cluster near random; heterogeneity around the +7.76 pp ATE is small vs noise. Raising capacity moved T-learner from *below* to *even with* response, so the ceiling is set by the data, not the learner. |
| S-learner shrinks to ~0 | Textbook failure: one model on `[X, T]` learns the main effect and makes the treatment interaction tiny, collapsing the ranking toward random. |
| Response baseline is competitive | Its score tracks *purchasability*, and since the e-mail lifts everyone, top-purchasability customers carry much of the total outcome. The edge disappears under **profit** optimisation (sure things cost full contact for little margin) — Day 4's job. |
| X-learner leads narrowly | Best Qini AUC (0.0236) and uplift@30% (0.1015), but the margin over response is thin — not a decisive win on one split. Motivates Day-3 DR/causal-forest challengers and the 10-seed benchmark. |

### Hillstrom — Day 3 (DR/forest, calibration, stability)

**Single split** (`RANDOM_STATE = 42`, test n = 8,523, ranking on `visit`;
notebook 04 §2). Day-3 challengers added to the Day-2 pack:

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
uplift; notebook 04 §4):

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

Findings (notebook 04):

| Observation | Reading |
| --- | --- |
| No uplift model beats the response baseline over 10 seeds | Response tops Qini (0.0231) *and* profit ($569); the declared verdict is a statistical tie, broken toward response on profit + parsimony. Day-2's premise — response is the bar — survives its own test. |
| Day-2's X-learner lead was noise | 0.0236 on one split → 0.0046 ± 0.0161 over seeds (0.9 SE above random). Single-split champions are coin flips with good PR; that is what the benchmark exists to catch. |
| Qini and profit disagree | DR is 2nd on Qini but 5th on profit; S is 3rd on Qini but 2nd on profit. Ranking `visit` ≠ counting conversion dollars — the project rule demands both metrics for exactly this reason. |
| Calibration: usable magnitudes, no skill | Both outcome models sit at slightly negative Brier skill vs the base rate, reliability gaps ≈ 0.034–0.038; Platt improves the held-out Brier only modestly. Calibrate before valuing, but select on Qini + profit. |
| Forest intervals are wide but directional | Mean CI width 0.176 ≈ 2.3× the ATE; 42% of customers get intervals excluding zero and 97.6% of point estimates are positive — the forest says "who", not "how much", confidently. |
| Spend ordering does not survive its own floor | A random ranking on spend spans −0.89 → 4.18 at 10% across seeds (std 1.01); the forest's 1.87 sits inside that band. `visit`'s floor is ~45× tighter — which is why the benchmark runs on `visit`. Any Day-4 spend claim must carry this uncertainty. |
| Profit must be valued on the money outcome | The benchmark passes `conversion` as a 7th fold array; valuing the ranking outcome (`visit`) instead would have reported ≈ $7,900 rather than $569 on the same run — a ~14× overstatement. See the profit-formula note below. |

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
