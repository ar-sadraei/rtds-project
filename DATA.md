# Dataset card — financial competing-risks table

*Built by `src/prepare_financial_data.py`.*

## Source

Freddie Mac **Single-Family Loan-Level Dataset**, Standard Dataset, 50,000-loan
sample per origination year.

**Free for academic and non-commercial research, but NOT redistributable.** This
is why `data/` is excluded from version control and why the processed CSV must
not be committed or posted publicly either — it is a derived work of licensed
data. Share it with the group directly, not through a public repository.

To get the raw files: register at
<https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>, then from
the SFLLD Data Download page take the **Sample File** column for each vintage
year and put the archives in `data/raw/` **without renaming them**
(`sample_2013.zip`, `sample_2014.zip`, ...).

Each zip contains two pipe-delimited files with no header row:

| file | rows | size |
|---|---|---|
| `sample_orig_YYYY.txt` | one per loan (50,000) | ~6 MB |
| `sample_perf_YYYY.txt` | one per loan per month | ~475 MB |

Field positions come from the official **File Layout (effective July 2026)**:
`https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx`
— origination 31 fields, monthly performance 35. The script asserts both counts
and fails loudly if the layout changes, rather than silently shifting columns.

## Why this dataset for competing risks

A mortgage can end in two ways that exclude each other. It can **default**, or
it can be **prepaid** (refinanced or the house is sold). Once one happens the
other never can. That is exactly the competing-risks structure, and it is the
reason this dataset is used for it in the literature.

It also has the property that makes the competing event matter rather than being
a nuisance: **prepayment is far more common than default**. Treating prepaid
loans as merely "censored" — which single-event survival analysis does — throws
away the majority of the outcomes and biases the default estimate upward.

## Build

```bash
python3 src/prepare_financial_data.py
```

Defaults: origination years 2013–2018, 50,000 loans, follow-up cut at 60 months,
seed 0. Output: `data/processed/financial_competing_risks.csv`.

```bash
# other sizes / windows
python3 src/prepare_financial_data.py --max-rows 250000
python3 src/prepare_financial_data.py --years 2016 2017 2018 --max-follow-up 36
python3 src/prepare_financial_data.py --max-rows 0 --max-follow-up 0   # everything
```

The script prints a cleaning report — what it dropped and why — before writing.
Read it. It is the audit trail for the file you hand to the group.

## Columns

| column | meaning |
|---|---|
| `loan_id` | Freddie Mac loan sequence number |
| `vintage` | origination year |
| `time` | **loan age in months** when the outcome below was observed |
| `event` | `0` censored, `1` default, `2` prepaid |
| `default_zb_alt` | `1` if a credit-loss disposition code ever appeared — the alternative, stricter default definition, kept so the choice can be tested |
| `fico` | credit score at origination |
| `ltv`, `cltv` | loan-to-value, combined loan-to-value |
| `dti` | debt-to-income |
| `orig_upb` | original loan amount |
| `orig_rate` | interest rate at origination |
| `orig_term` | loan term in months |
| `mi_pct` | mortgage insurance coverage % (`0` = none) |
| `n_units`, `n_borrowers` | property units, number of borrowers |
| `first_time_buyer` | `Y` / `N` / `9` unknown |
| `occupancy` | `P` owner-occupied, `S` second home, `I` investment |
| `channel` | `R` retail, `B` broker, `C` correspondent, `T` TPO |
| `loan_purpose` | `P` purchase, `C` cash-out refi, `N` no-cash-out refi |
| `property_type` | `SF` single family, `CO` condo, `PU` planned unit, `MH` manufactured, `CP` co-op |
| `state` | US state code |

`time` is measured from **origination**, not from a calendar date. That is what
makes it comparable across vintages, and what makes a train-on-early /
test-on-later split meaningful.

## Event definitions, and the judgement in them

| event | code | definition |
|---|---|---|
| censored | 0 | neither event observed within the follow-up window |
| default | 1 | first month with delinquency status ≥ 3 (90+ days past due), or `RA` (property repossessed) |
| prepaid | 2 | zero-balance code `01`, if it occurs before any default |

**Default is defined by delinquency, not by final disposition**, and the
threshold is **90 days**. That is the data owner's own convention for this
product: FHFA and Freddie Mac report "seriously delinquent" loans as those 90
or more days past due or in foreclosure, and it is the headline credit metric
in their monthly volume summaries. It also matches the general Basel/CRR
reference definition of default.

Two thresholds we deliberately did not use:

- **180 days.** US capital regulation ([12 CFR 3.101](https://www.law.cornell.edu/cfr/text/12/3.101))
  puts a *residential mortgage* exposure in default at 180 days, and Basel's
  IRB approach allows 180 for retail exposures secured by residential real
  estate. That threshold exists for sizing regulatory capital, which is not
  what we are modelling. Worth stating in any write-up, because a reader from
  a credit-risk background will expect 180.
- **Any delinquency.** Peng and Lessmann (2026, arXiv:2601.20533) use this on
  these same sample files and call it "an early-delinquency definition" as
  against "a conventional default definition based on the Basel criterion".
  Different question.

**This choice moves the default rate by a factor of 12** — 1.7–5.3% per vintage
at 90 days, against 0.1–0.6% if you wait for a credit-loss disposition code.
Only ~7% of 90-day delinquencies reach such a disposition within 60 months; the
rest cure or are still in process. `default_zb_alt` carries the
disposition-based version in a separate column so the choice stays testable.

If anyone asks why your default rate differs from a published figure, this is
the first place to look. In particular, do **not** cite Blumenstock et al.
(2022) for the 90-day rule: their reported rates on this dataset (0.1% for
2013, 0.0% for 2016–17) are far too low for it, their online companion does not
state what they used, and the main paper is paywalled and unchecked.

## Follow-up window

By default every loan is censored at **60 months**. A 2013 loan has been
observed far longer than a 2018 one; without a common cut, `time` means a
different thing for each vintage and any comparison across them is confounded by
observation length alone. Pass `--max-follow-up 0` to keep full follow-up.

## Known limitations

1. **Origination-time covariates only.** Nothing time-varying — no unemployment,
   no house-price index, no current interest rate. Prepayment in particular is
   driven by the gap between the loan's rate and the current market rate, which
   is not in here.
2. **The sample files are a 50,000-loan random sample per year**, not the full
   population. Freddie Mac draws them; we do not control the draw.
3. **Agency loans only.** These conform to Freddie Mac's underwriting standards,
   so the credit quality range is narrower than the whole mortgage market. No
   subprime.
4. **Censoring is light at 60 months**, because most loans prepay first. The
   competing risk, not censoring, is what shapes this data.
