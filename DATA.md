# Dataset card — financial competing-risks table

*Built by `src/0_prepare_financial_data.py`.*

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
year and put the archives in `data/banking/0_raw/` **without renaming them**
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

## The two cohorts

The financial data is built as **two disjoint cohorts**, not one pooled set.
They share no loans, and the same protocol is applied to each.

| | calm | crisis |
|---|---|---|
| origination years | 2013–2018 | 2005–2009 |
| loans | 55,000 (9,167 per vintage) | 55,000 (11,000 per vintage) |
| default | 3.97% | 8.71% |
| prepaid | 60.65% | 65.20% |
| censored at 72 months | 35.38% | 26.09% |
| DTI not reported | 10.2% | 4.1% |
| file | `financial_calm_2013_2018.csv` | `financial_crisis_2005_2009.csv` |

**Why two.** The hypothesis under test is that evaluation protocol matters
*because the population shifts between training and test*. That predicts a
larger effect when the shift is larger. The two cohorts are a dose
manipulation of exactly that, with everything else held fixed — same models,
same metrics, same window, same sample size, same equal-per-vintage draw.

Default rate by origination year shows the contrast:

```
calm     2013 1.7%   2014 1.6%   2015 2.9%   2016 4.1%   2017 5.3%   2018 5.2%
crisis   2005 6.4%   2006 10.2%  2007 12.9%  2008 7.6%   2009 1.8%
```

The calm cohort drifts monotonically over roughly a 3x range. The crisis
cohort swings over a 7x range **and reverses**: 2009 loans default less often
than 2005 loans, because they were underwritten after the collapse to much
tighter standards. A time-ordered split on the crisis cohort therefore tests on
an *easier* population, which is the opposite of the usual intuition and should
not be mistaken for a bug.

## Build

```bash
python3 src/0_prepare_financial_data.py --cohort calm
python3 src/0_prepare_financial_data.py --cohort crisis
```

`--cohort` is required. It sets the origination years, the row cap and the output
path from the `COHORTS` table in the script, because those three together are what
makes a cohort rather than three settings that happen to be passed at the same
time. Both cohorts use 55,000 loans, a 72-month window and seed 0, and draw an
equal number of loans from each origination year so no vintage dominates.

```bash
# other sizes / windows: the cohort still supplies whatever is not overridden
python3 src/0_prepare_financial_data.py --cohort calm --max-rows 250000
python3 src/0_prepare_financial_data.py --cohort calm --years 2016 2017 2018 \
  --max-follow-up 36 --out /tmp/check.csv
python3 src/0_prepare_financial_data.py --cohort calm --max-rows 0 \
  --max-follow-up 0 --out /tmp/everything.csv   # everything
```

Give an override run its own `--out`, or it writes over the cohort's file.

The script prints a cleaning report and a per-column value summary before it
writes. Read them. They are the audit trail for the file you hand to the group,
and the value summary is what catches a category that exists in one cohort and
not the other — `channel` code `T` (third-party origination) appears in the
early vintages and is absent from 2013 onward, which matters when the feature
is encoded.

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
Only ~7% of 90-day delinquencies reach such a disposition within the window; the
rest cure or are still in process. `default_zb_alt` carries the
disposition-based version in a separate column so the choice stays testable.

If anyone asks why your default rate differs from a published figure, this is
the first place to look.

**Blumenstock et al. (2022) use the same definition** on this dataset: *"Default
is defined as a loan turning into 3-month delinquency for the first time"*
(§3.1). So the 90-day rule is citable, and our event definition matches the
study whose conclusions we are testing.

One number in their paper looks like it contradicts this and does not. Their
Table 1 reports default shares of 0.1% for 2013 and 0.0% for 2016–17, far below
our 1.7–5.3%. That table is a **per-calendar-year** distribution of event
occurrences, not a per-vintage cumulative default rate — a loan contributes to
the year in which the event falls, not to its origination year. The two
quantities are not comparable, and an earlier version of this file wrongly
inferred from them that Blumenstock used a stricter definition.

## Follow-up window

**Why 72.** Three reasons, none of which is "it is a round number":

1. It captures the large majority of what ever happens. Within the available
   follow-up, 72 months covers 87% of defaults and 86% of prepayments, against
   76% and 76% at 60 months. Going further hits diminishing returns.
2. It fits inside the follow-up available in the financial data: even the
   newest mortgage vintage (2018) can be observed for 99 months. It does not
   fully fit the medical data, see "Medical data" below.
3. Blumenstock et al. (2022) evaluate at 24, 48 and 72 months on this dataset,
   so our horizons line up with the study whose conclusions we are testing.
   They do not justify the choice themselves -- reasons 1 and 2 are ours -- but
   matching them makes the results directly comparable.

By default every loan is censored at **72 months**. A 2013 loan has been
observed far longer than a 2018 one; without a common cut, `time` means a
different thing for each vintage and any comparison across them is confounded by
observation length alone. Pass `--max-follow-up 0` to keep full follow-up.

### Medical data

The medical data (NHANES, built by `src/0_prepare_medical_data.py`) is cut at the
same 72 months, in `HORIZON_MONTHS`. All ten survey cycles (1999 to 2018) are kept.
After cleaning there are 52,235 people.

The cut moves 4,402 deaths after month 72 to censored. Within 72 months there are
1,250 heart disease or stroke, 923 cancer and 1,720 other deaths. At 60 months there
would be 1,049, 769 and 1,407, so 72 gives about 20% more events per cause.

The cost is that the later cycles cannot be followed for 72 months. Share of people
in each cycle who died or were followed for at least the horizon:

| cycle | 60 months | 72 months |
|---|---|---|
| 1999 to 2012 | 100% | 100% |
| 2013 to 2014 | 99% | 57% |
| 2015 to 2016 | 6% | 5% |
| 2017 to 2018 | 2% | 2% |

So for 2013 to 2014, 43% of people are censored before month 72, and for 2015 to
2018 nearly everyone is. The models handle this as censoring, but the last test
folds rest more on the censoring estimate, and the 2017 to 2018 fold has only about
30 heart disease and 30 cancer events. Read their 72 month metrics with care, and
also look at shorter horizons (24 and 48 months).

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
4. **In the crisis cohort both competing events shift, not only default.**
   Interest rates collapsed from 2008, so prepayment behaviour changes as much
   as default risk does. This makes the crisis cohort a genuinely larger
   distribution shift, but it means no result from it can be attributed to
   default risk alone.
5. **Censoring is light at 72 months**, because most loans prepay first. The
   competing risk, not censoring, is what shapes this data.
