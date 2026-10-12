# Dataset card — Freddie Mac competing-risks table

*Built 11 Oct 2026 by `src/build_dataset.py`.*

## Source

Freddie Mac **Single-Family Loan-Level Dataset**, Standard Dataset, 50,000-loan sample per vintage year, downloaded from Clarity Data Intelligence on 11 Oct 2026. Free for academic and non-commercial research.

Field positions taken from the official **File Layout (effective July 2026)**:
`https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx`
— origination 31 fields, monthly performance 35. The loader asserts both counts and skips malformed rows.

**Vintages:** 2013–2018, six years, **300,000 loans**, performance observed to March 2026.
Raw archives in `data/raw/` (295 MB zipped). Processed table in `data/processed/loans.parquet` (5.2 MB).

## Event definitions

| event | code | definition |
|---|---|---|
| censored | 0 | neither event observed by March 2026 |
| default | 1 | first month with Current Loan Delinquency Status ≥ 3 (90+ days past due), or `RA` (REO acquisition) |
| prepaid | 2 | Zero Balance Code `01`, if it occurs before any default |

`time` is **loan age in months**, so the time origin is origination — which is what makes vintage-ordered splits meaningful.

The default definition follows **Blumenstock et al. (2022)**, who use first 3-month delinquency. It is observed earlier and more consistently than a disposition code. For 30-year loans originated 2013–2018, legal maturity falls in 2043 or later, so Zero Balance Code `01` is prepayment rather than maturity within this window.

`credit_event_zb` records the alternative, disposition-based definition (Zero Balance Code in 02, 03, 09, 15) for sensitivity analysis.

## What the data shows

### The censoring contrast replicates

Applying Dirick et al.'s two event definitions to these loans:

| vintage | competing risks | default only |
|---|---|---|
| 2013 | 19.9% | 96.35% |
| 2014 | 16.3% | 96.30% |
| 2015 | 21.4% | 96.07% |
| 2016 | 26.0% | 95.41% |
| 2017 | 22.3% | 94.33% |
| 2018 | 17.6% | 94.75% |
| **all** | **20.6%** | **95.53%** |

**Dirick et al. report 20–85% and 94.56–98.16%** on ten European consumer-credit portfolios. Our figures fall inside both ranges on entirely different data — US agency mortgages, a different decade, a different product. That is an independent replication of the number the literature review was built on, and it is the natural opening result for the project.

### Event rates

| event | n | share | median time (months) |
|---|---|---|---|
| censored | 61,732 | 20.6% | 119 |
| default | 13,396 | 4.5% | 42 |
| prepaid | 224,872 | 75.0% | 41 |

Prepayment dominates by a factor of ~17. Any model that censors it is discarding three quarters of the observed outcomes.

### Default definition matters by an order of magnitude

- Delinquency-based (90+ dpd): **4.47%**
- Zero-balance credit event: **0.31%**

A 14× difference from a definitional choice alone. Worth a sensitivity row in the report.

## Columns

`loan_id`, `vintage`, `time`, `event`, `credit_event_zb`, `last_age`, plus origination covariates:
`fico`, `first_payment_date`, `first_time_buyer`, `mi_pct`, `n_units`, `occupancy`, `cltv`, `dti`, `orig_upb`, `ltv`, `orig_rate`, `channel`, `state`, `property_type`, `loan_purpose`, `orig_term`, `n_borrowers`.

Freddie Mac's missing sentinels (9999 for FICO, 999 for ratios, 99 for counts) are converted to NaN.

## Known issues

- **`dti` is 10.2% missing.** Decide on imputation vs exclusion and state it in the Method.
- **Covariates are origination-time only.** No time-varying macro variables yet. Bellotti & Crook's contribution was exactly those, so this is a limitation to declare or a later extension.
- **Samples, not populations.** 50,000 loans per vintage, drawn by Freddie Mac. Fine for model comparison; not for absolute portfolio statistics.
- **Right truncation.** Later vintages have shorter follow-up (median 29 months for 2018 vs 119 for 2013). Handled correctly by the survival setup, but it means vintage and follow-up length are confounded — relevant when interpreting vintage-ordered splits.

## Reproducing

```bash
cd project
python3 -m venv .venv && .venv/bin/pip install pandas pyarrow
.venv/bin/python src/build_dataset.py            # all six vintages, ~20 s
.venv/bin/python src/build_dataset.py --years 2013
```
