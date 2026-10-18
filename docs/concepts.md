# Concepts

One place for every idea this project uses and every number we measure with it.
Notebooks explain a concept briefly where it is first used and link back here;
this file is the canonical, growing version.

**Keeping it in sync.** When a notebook introduces a term, a metric or a finding,
add it here in the same change. Definitions go in the numbered sections; results
and metric values go in the [findings ledger](#findings-ledger). Anything that
only lives in a notebook cell is a bug.

Scope so far: Day 1 (experiment fundamentals, randomization, ATE). Later days
append to the same structure — Qini/AUUC and calibration, the profit curve, the
copilot's grounding rules.

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
| Binary arm (`mens`) | 42,613 rows, 0 SMD flagged | notebooks 01–02 |
| ATE, `mens` vs control | see notebook 02 | notebook 02 |

### Profit formula

```text
profit(k) = v · Δconv(k) − c · n(k)
```

| Symbol | Meaning |
| --- | --- |
| `k` | Targeting depth (top-k customers, ranked by predicted uplift) |
| `Δconv(k)` | Incremental conversions caused by targeting the top k |
| `v` | Value per conversion |
| `c` | Cost per contact |
| `n(k)` | Number contacted |

`v` and `c` must share a currency (USD here). Values live in
`configs/business_assumptions.yaml` and are still to be decided — see the README
profit-assumptions table.
