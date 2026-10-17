# Hillstrom MineThatData E-Mail Analytics — Structured Data Description

Structured reference for the Hillstrom e-mail experiment, derived from the original
challenge prose (`hillstrom_data description.md`, kept at the repo root as the raw source).
Everything below is verified against the actual file in `data/raw/`.

## 1. Summary

| Item | Value |
| --- | --- |
| Full name | MineThatData E-Mail Analytics And Data Mining Challenge |
| Author / publisher | Kevin Hillstrom (MineThatData), 2008-03-20 |
| Rows | 64,000 customers |
| Columns | 12 |
| Grain | One row = one customer |
| Treatment | Randomized e-mail campaign, 3 arms, ~1/3 each |
| Observation window | 14 days after the e-mail was sent |
| Outcome variables | `visit`, `conversion`, `spend` |
| Local file | `data/raw/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` |
| Size | ~3.8 MB (CSV, header + 64,000 rows) |
| Encoding / format | UTF-8, RFC-4180 CSV, comma-delimited, quoted fields |
| SHA-256 | `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` |

### Role in UpliftIQ

Hillstrom is the **primary dataset**. It carries the profit story because it records real
`spend`, and its features are interpretable, which makes it the source for segment
explanations and copilot demos. Criteo is a scale benchmark only.

## 2. Experiment design

Customers were randomly assigned to exactly one of three arms:

| Arm | Count | Share |
| --- | --- | --- |
| `Mens E-Mail` | 21,307 | 33.29% |
| `Womens E-Mail` | 21,387 | 33.42% |
| `No E-Mail` (control) | 21,306 | 33.29% |
| **Total** | **64,000** | **100%** |

All customers "last purchased within twelve months". Randomization is at the customer
level; no customer appears in more than one arm.

### Observed outcome rates by arm

Rates are over the full 14-day window. `mean_spend` is the per-customer average of `spend`
(zero for the vast majority), so it equals conversion rate × average spend per converter.

| Arm | Visit rate | Conversion rate | Mean spend (USD) |
| --- | --- | --- | --- |
| Mens E-Mail | 0.1828 | 0.0125 | 1.4226 |
| Womens E-Mail | 0.1514 | 0.0088 | 1.0772 |
| No E-Mail | 0.1062 | 0.0057 | 0.6528 |

Naive arm-vs-control differences (to be replaced by ATE with CIs on Day 1):

| Contrast | Δ Visit (pp) | Δ Conversion (pp) | Δ Mean spend (USD) |
| --- | --- | --- | --- |
| Mens vs No E-Mail | +7.66 | +0.68 | +0.770 |
| Womens vs No E-Mail | +4.52 | +0.31 | +0.424 |

**Both e-mail arms beat control, and Mens beats Womens on every outcome.** This is the
evidence used on Day 1 to pick the binary experiment (see §6).

## 3. Data dictionary

### 3.1 Pre-treatment covariates (historical attributes)

| Column | Type | Description | Observed values |
| --- | --- | --- | --- |
| `recency` | int | Months since last purchase | 1–12 (mean 5.76) |
| `history_segment` | categorical (ordinal) | Bucket of dollars spent in the past year | 7 ordered buckets, see §3.2 |
| `history` | float | Actual dollars spent in the past year | 29.99–3345.93 (mean 242.09) |
| `mens` | int (0/1) | Bought mens merchandise in the past year | 0: 28,734 · 1: 35,266 |
| `womens` | int (0/1) | Bought womens merchandise in the past year | 0: 28,818 · 1: 35,182 |
| `zip_code` | categorical | Zip type | Urban: 25,661 · Surburban: 28,776 · Rural: 9,563 |
| `newbie` | int (0/1) | New customer in the past 12 months | 0: 31,856 · 1: 32,144 |
| `channel` | categorical | Channel(s) purchased from in the past year | Phone: 28,021 · Web: 28,217 · Multichannel: 7,762 |

### 3.2 `history_segment` buckets (ordinal, ascending)

| Bucket label (as stored) | Customers |
| --- | --- |
| `1) $0 - $100` | 22,970 |
| `2) $100 - $200` | 14,254 |
| `3) $200 - $350` | 12,289 |
| `4) $350 - $500` | 6,409 |
| `5) $500 - $750` | 4,911 |
| `6) $750 - $1,000` | 1,859 |
| `7) $1,000 +` | 1,308 |

Note: labels `6` and `7` contain a comma inside the quoted string
(`"6) $750 - $1,000"`). Always parse with a real CSV reader — naive comma splitting
corrupts these rows.

### 3.3 Treatment variable

| Column | Type | Description | Values |
| --- | --- | --- | --- |
| `segment` | categorical | The campaign the customer received | `Mens E-Mail`, `Womens E-Mail`, `No E-Mail` |

This is the **assignment** variable and the only valid treatment definition. There is no
`exposure` column in Hillstrom (unlike Criteo), so the post-assignment contamination trap
from the Criteo notes does not apply here.

### 3.4 Post-treatment outcomes (14-day window)

| Column | Type | Description | Observed values |
| --- | --- | --- | --- |
| `visit` | int (0/1) | Visited the website in the 2 weeks after the campaign | 0: 54,606 · 1: 9,394 (14.68%) |
| `conversion` | int (0/1) | Purchased in the 2 weeks after the campaign | 0: 63,422 · 1: 578 (0.90%) |
| `spend` | float | Dollars spent in the 2 weeks after the campaign | 0 for 63,422 rows; >0 for exactly the 578 converters; max 499.00 |

**Key structural facts to remember:**

- `spend > 0` iff `conversion == 1` (578 rows). Spend is therefore zero-inflated and
  lives on a hurdle/mixture structure: P(convert) × spend|convert.
- `conversion == 1` implies `visit == 1`, but not the reverse (9,394 visited, 578 bought).
- `visit` is ~16× more frequent than `conversion`, so it is the primary modelling target
  for stable metrics; `conversion` and `spend` are noisier secondary targets.

## 4. Causal structure (per-customer groups)

For a given promo, each customer falls into one of four response groups. Hillstrom cannot
label these directly (only one potential outcome is observed per customer), which is the
entire reason uplift models are needed.

| Group | Buys without promo? | Buys with promo? | Should get the promo? |
| --- | --- | --- | --- |
| Persuadables | No | Yes | Yes — the only incremental-value group |
| Sure things | Yes | Yes | No — discount wastes margin |
| Lost causes | No | No | No — contact cost wasted |
| Sleeping dogs | Yes | No | No — the promo causes harm |

## 5. Modelling conventions for this project

- **Binary experiment (default).** `treatment = (segment == "Mens E-Mail")`, dropping the
  `Womens E-Mail` rows, with `No E-Mail` as control. Both arms are retained only for the
  multi-treatment extension.
- **Primary rank target:** `visit`. **Secondary:** `conversion`, then `spend`.
- **Features:** `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`.
  Drop `history_segment` (collinear with `history`) at model time; one-hot `zip_code` and
  `channel`. **Never** let `segment`, `visit`, `conversion` or `spend` leak into features.
- **Currency:** `spend` is USD. For ₹ display, convert at a stated, dated rate and label it;
  otherwise keep USD and label it USD.

## 6. Data quality notes

- **`zip_code` contains a typo in the source data:** `Surburban` (not `Suburban`). It is
  consistent across all 28,776 rows; normalize on load if you want clean labels, but be
  aware the raw value differs from the intuitive spelling.
- Quoted fields with embedded commas in `history_segment` (§3.2) — parse properly.
- No missing values detected; all 64,000 rows have all 12 fields populated.
- Arm sizes differ slightly (21,306–21,387), well within randomization noise; the
  randomization check (SMD < 0.1) is the formal test.

## 7. Provenance / original source

Original prose (`hillstrom_data description.md`) states: the dataset is a CSV "about the
size of a typical mp3 file", 64,000 customers who last purchased within twelve months,
randomized 1/3 Mens / 1/3 Womens / 1/3 no e-mail, with a two-week tracking window. The
original challenge page is no longer reliably hosted; mirrors exist on Kaggle and the
Minethatdata archive. The local copy is treated as canonical and verified by the SHA-256
in §1.
