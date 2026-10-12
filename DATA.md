# Dataset card — Freddie Mac competing-risks tables

*Built 11 Oct 2026 by `src/build_dataset.py`; crisis cohort added 12 Oct.*

## Source

Freddie Mac **Single-Family Loan-Level Dataset**, Standard Dataset, 50,000-loan sample per vintage year, downloaded from Clarity Data Intelligence. Free for academic and non-commercial research, **not redistributable** — hence `data/` is excluded from version control.

Register at <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>, then download the *Sample File* column for each vintage from the SFLLD Data Download page and put the archives in `data/raw/`.

Field positions come from the official **File Layout (effective July 2026)**:
`https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx`
— origination 31 fields, monthly performance 35. The loader asserts both counts and skips malformed rows.

## The two cohorts

| | vintages | loans | default | prepaid | censored | median follow-up |
|---|---|---|---|---|---|---|
| **Post-crisis** `loans` | 2013–2018 | 300,000 | 4.47% | 75.0% | 20.6% | 54 mo |
| **Crisis** `loans_crisis` | 2005–2009 | 250,000 | 10.53% | 87.1% | 2.4% | 48 mo |

Performance observed to March 2026. Raw archives ~295 MB zipped; processed tables ~5 MB each.

```bash
.venv/bin/python src/build_dataset.py --years 2013 2014 2015 2016 2017 2018 --out loans
.venv/bin/python src/build_dataset.py --years 2005 2006 2007 2008 2009 --out loans_crisis
```

The `--out` flag matters: without it both runs write to `loans.parquet` and the second silently overwrites the first.

## Event definitions

| event | code | definition |
|---|---|---|
| censored | 0 | neither event observed by March 2026 |
| default | 1 | first month with delinquency status ≥ 3 (90+ days past due), or `RA` (REO acquisition) |
| prepaid | 2 | zero-balance code `01`, if it occurs before any default |

`time` is **loan age in months**, so the time origin is origination — which is what makes vintage-ordered splits meaningful.

The default definition follows **Blumenstock et al. (2022)**, who use first 3-month delinquency; it is observed earlier and more consistently than a disposition code. For 30-year loans originated 2005–2018, legal maturity falls in 2035 or later, so zero-balance code `01` is prepayment rather than maturity throughout.

`credit_event_zb` records the alternative disposition-based definition (zero-balance code in 02, 03, 09, 15) for sensitivity analysis.

## What the data shows

### The censoring contrast replicates

Applying Dirick et al.'s two event definitions:

| cohort | competing risks | default only |
|---|---|---|
| post-crisis | **20.6%** | **95.5%** |
| crisis | 2.4% | 89.5% |

Dirick et al. report **20–85%** and **94.56–98.16%** on ten European consumer-credit portfolios. The post-crisis figures fall inside both ranges on entirely different data — US agency mortgages, a different decade, a different product. This is the independent replication the report leads with.

The crisis cohort sits outside those ranges in the opposite direction: by 2026 almost every 2005–09 loan has exited, so only 2.4% remain censored.

### Prepayment dominates

Post-crisis, prepayment outnumbers default **17 to 1**; in the crisis cohort, **8 to 1**. Censoring it discards most of the observed outcomes.

### The definition of default matters by an order of magnitude

Post-crisis: delinquency-based 4.47% against zero-balance credit event 0.31% — a **14×** difference from a definitional choice alone.

## Columns

`loan_id`, `vintage`, `time`, `event`, `credit_event_zb`, `last_age`, plus origination covariates:
`fico`, `first_payment_date`, `first_time_buyer`, `mi_pct`, `n_units`, `occupancy`, `cltv`, `dti`, `orig_upb`, `ltv`, `orig_rate`, `channel`, `state`, `property_type`, `loan_purpose`, `orig_term`, `n_borrowers`.

Freddie Mac's missing sentinels (9999 for FICO, 999 for ratios, 99 for counts) are converted to NaN.

The experiments use two feature sets: nine numeric covariates, and those plus one-hot geography and product type (~70 features). Adding the categorical block helped the Cox model more than it helped DeepHit — see `FINDINGS.md`.

## Known issues

- **`dti` is missing for 10.2% of post-crisis loans** and 4.1% of crisis loans. Median-imputed inside `_Standardiser`; stated in the report.
- **Covariates are origination-time only.** No time-varying macroeconomic variables, which is verified as Bellotti & Crook's (2009) contribution — their abstract describes including macro variables "as time-varying covariates". A real gap against that literature.
- **Samples, not populations.** 50,000 loans per vintage, drawn by Freddie Mac. Fine for model comparison, not for absolute portfolio statistics.
- **Vintage and follow-up length are confounded.** Later vintages have shorter observation windows. Handled correctly by the survival setup, but it matters when interpreting vintage-ordered splits.
- **Crisis cohort is nearly fully resolved** (2.4% censored), so censoring-weighted metrics there rest on very few censored observations.

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install pandas pyarrow
.venv/bin/python src/build_dataset.py --out loans            # ~20 s for six vintages
```

Then `src/validate_harness.py` before trusting anything downstream: it checks that the Aalen–Johansen null model scores exactly 0.5 AUC and is calibrated in-distribution.
