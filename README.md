# Competing-risks survival models on medical and financial data

Group project in *Research Topics in Data Science* (DSV, Stockholm University, 2026).
Ali Emre Nebiler, Amirreza Sadraei, Filip Becic.

## What this project does

The project compares deep and classical survival models on two real datasets where
more than one thing can happen to a subject:

- **Medical:** adults from the US health survey NHANES (1999 to 2018), followed to
  their death. The cause of death is the event: heart disease, cancer, or any other
  cause.
- **Financial:** US mortgages from the Freddie Mac loan-level data. A loan ends in
  default or in early repayment (prepayment).

These are **competing risks**: once one event happens, the others can no longer
happen. Every person or loan gets a `time` (when something happened or when we
stopped watching) and an `event` (`0` means censored, so nothing was observed yet,
and `1`, `2`, ... mean which event happened).

What is in the repository today is the data side: pipelines that turn the raw files
into clean tables, build the train and test folds, and normalize inside each fold.
The model code (Deep Survival Machines and Neural Fine-Gray) is vendored in
`vendor/`. The tune, benchmark and analyze pipelines are not written yet.

## Quick start

**1. Environment.** Keep the virtualenv outside iCloud. A venv inside an iCloud
synced folder is about 36,000 files, and syncing them makes Python startup take
minutes.

```bash
python3 -m venv ~/venvs/rtds
~/venvs/rtds/bin/pip install -r requirements.txt comprisk \
  "git+https://github.com/soda-inria/hazardous.git@26e64323fb462cc283bf145f3e37af4f78e1a5d4"
source ~/venvs/rtds/bin/activate
```

The hazardous commit is PR #100 of `soda-inria/hazardous`. It adds CR D-calibration,
a calibration measure for competing risks, and is not in any release. It reports its
version as `0.2.0`, the same as the released package, so pin the commit or nobody can
tell which one they have. `comprisk` caps `numba<0.66`, so installing it moves numba
and numpy back a version from what `requirements.txt` pins. Check that nothing broke:

```bash
python -c "import torch, numpy; from pycox.models import DeepHit; from comprisk import FineGrayRegression; from hazardous.metrics import d_cr_calibration; print('ok', torch.__version__, numpy.__version__)"
```

**2. Data.** The data is not in the repository (see "Data licensing" in the notes).
Download it and put it in the folders below. The links and file details are in the
notes under "Datasets".

| | folder | content |
|---|---|---|
| medical | `data/medical/0_raw/demography/` | one `.xpt` file per cycle |
| medical | `data/medical/0_raw/smoking/` | one `.xpt` file per cycle |
| medical | `data/medical/0_raw/mortality/` | one `.dat` file per cycle |
| financial | `data/financial/0_raw/` | `sample_YYYY.zip` for every year, not renamed |


The medical folders moved too. Folds went from `3_folds` to `2_folds` and the
normalized folds from `4_normalized` to `3_normalized`, because the separate
preprocessing step is now part of the merge. Rebuild those by running the pipeline
rather than moving them, since folds are saved row positions and the merged table
changed.

**3. Run.** From the project folder:

```bash
python src/pipelines/0_prepare_medical.py
python src/pipelines/0_prepare_financial.py
```

Each command runs all of its steps and prints what it did and how many rows each
cleaning step removed. Read that output rather than trusting the files. The financial
pipeline builds both cohorts (calm and crisis) in one run.

The pipelines take no command line arguments. Folders and settings are constants at
the top of each pipeline file: the origination years and row cap of each cohort, the
72-month cut, the seed and whether to drop loans with no DTI. Edit them there.

The financial zip files stay zipped. The pipeline reads them directly, so do not
unzip them.

## Project structure

```
src/
  utils/                      functions, grouped by content
    medical_utils.py          read NHANES, clean, label, recode, normalize
    financial_utils.py        read Freddie Mac, clean, subsample, normalize
    fold_utils.py             build and check the folds, shared by both datasets
  pipelines/                  runnable scripts, one per process and domain
    0_prepare_medical.py
    0_prepare_financial.py
data/                         not in git
  medical/    0_raw  1_merged  2_folds  3_normalized
  financial/  0_raw  1_merged  2_folds  3_normalized
vendor/                       other people's model code
AGENTS.md                     rules for people and agents working in the repo
```

Functions live in `src/utils/`. A pipeline only holds the folders and settings and
chains the steps. Pipelines are numbered in run order. The later processes (tune,
benchmark, analyze) will be separate pipelines in the same folder, split by domain
and numbered in order.

**The prepare pipelines.** Each one runs its stages in order, and every stage writes
to disk so the next one reads a file.

| stage | medical | financial |
|---|---|---|
| 0 raw files | `data/medical/0_raw/` | `data/financial/0_raw/` |
| 1 merge, clean and label (`time`, `event`) | `data/medical/1_merged/` | `data/financial/1_merged/` (one CSV per cohort) |
| 2 folds | `data/medical/2_folds/medical.json` | `data/financial/2_folds/financial-{cohort}.json` |
| 3 recode and normalize | `data/medical/3_normalized/{arm}/` | `data/financial/3_normalized/{cohort}/{arm}/` |

**Folds.** They are built once and saved as row positions, so every model sees
identical splits and comparisons between models are paired. There are two arms:

- **temporal:** an expanding window on the group (survey cycle or origination year).
  Fold `k` trains on every group before group `k` and tests on group `k`. With `G`
  groups there are `G - 1` folds.
- **random:** a stratified random split matched to its temporal partner on both
  training size and test size, so the difference between the arms comes from the
  split rule and not from the amount of training data.

The medical data has 9 folds per arm, the calm financial cohort 5 and the crisis
cohort 4. The folds are checked for leaks before they are saved: no group on both
sides, no row on both sides, and matched sizes.

**Normalization.** Each fold is written as `fold_N_train.csv` and `fold_N_test.csv`.
Every statistic (the scaling, the imputation, the category lists) comes from the
training rows of that fold alone and is then applied to its test rows. The numbers
used are saved in `scaling.json` next to the folds. A model runner just reads a file
and fits. Nothing downstream re-splits or re-scales, which is how a leak is ruled
out and not just hoped away.

- Medical: sex, race and smoking are recoded to 0/1 columns (fixed rules, nothing learned), and `age_years` becomes `age_std`.
- Financial: nine numeric columns become `*_std`, missing values are filled with the
  training median, and the categorical columns are one-hot encoded. See the notes.

The financial normalized folds take about 160 MB. Nothing under `data/` is tracked,
so they stay local.

## Notes

### Datasets

| | medical | financial |
|---|---|---|
| source | NHANES 1999 to 2018 with NCHS linked mortality (2019) | Freddie Mac Single-Family Loan-Level Dataset |
| unit | a person | a loan |
| size | 52,235 people | two cohorts of 55,000 loans |
| events | `1` heart disease, `2` cancer, `3` other cause | `1` default, `2` prepaid |
| `time` | months from the exam | loan age in months |
| group for the time-ordered split | survey cycle (10 cycles) | origination year |

Links:

- NHANES survey files: <https://wwwn.cdc.gov/nchs/nhanes/>
- NCHS linked mortality files (2019 public use): <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/>
- Linked mortality data dictionary: <https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-files-data-dictionary.pdf>
- Linkage methods and analytic considerations (explains the altered public-use values): <https://www.cdc.gov/nchs/data/datalinkage/2019NDI-Linkage-Methods-and-Analytic-Considerations-508.pdf>
- Freddie Mac dataset (register first, it is free): <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>
- Freddie Mac file layout (July 2026): <https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx>

#### Medical data (NHANES)

**Source.** Three kinds of files are joined on `SEQN`, which is unique across the
continuous cycles 1999 to 2018. NHANES III is not used, because its `SEQN` overlaps.

- Demography (`.xpt`, one per cycle): age, sex, race and ethnicity, and the cycle code
  `SDDSRVYR` (`1` is 1999 to 2000, up to `10` for 2017 to 2018).
- Smoking questionnaire (`.xpt`, one per cycle): `SMQ020` (smoked 100 cigarettes in
  life) and `SMQ040` (smokes now).
- Linked mortality (`.dat`, fixed width, one per cycle): eligibility, vital status,
  leading cause of death and follow-up months. The positions used are `SEQN` 1 to 6,
  `ELIGSTAT` 15, `MORTSTAT` 16, `UCOD_LEADING` 17 to 19 and `PERMTH_EXM` 46 to 48.
  Follow-up ends on 31 December 2019.

File names do not matter. The script reads every `.xpt` and `.dat` file in the folder
and stops if a participant id appears twice.

**Cleaning.** Each step prints how many people it removed:

| step | removed | left |
|---|---|---|
| merged participants | | 101,316 |
| under 20 | 46,235 | 55,081 |
| no mortality status (not linkable to death records) | 136 | 54,945 |
| no answer to "smoked 100 cigarettes" | 9 | 54,936 |
| smoked 100 cigarettes but no "smokes now" answer | 1 | 54,935 |
| died without a cause of death | 2 | 54,933 |
| no follow-up time | 2,653 | 52,280 |
| refused or "don't know" in a smoking question | 45 | 52,235 |

Why these:

- Linked mortality exists only for adults, so children have no event or follow-up.
  The cutoff is 20 because the smoking questions were only asked of people 20 and
  over until the 2013 to 2014 cycle.
- A blank mortality status is not "alive". Those people could not be matched to
  death records, so we do not know what happened to them.
- Never-smokers are not asked "smokes now", so an empty answer there is expected and
  those people are kept.
- A blank follow-up time (`PERMTH_EXM`) most likely means the person was interviewed
  but never attended the exam, and the time is measured from the exam.
- The `mortality_status` column decides who died. `UCOD_LEADING` value 10 is also
  used for people who are alive.

**Event.** `event` is `0` for censored, `1` for heart disease (NCHS cause 1), `2` for
cancer (cause 2) and `3` for all other causes (causes 3 to 10, including stroke).
Follow-up is cut at 72 months, so a death after that counts as censored. After the cut
there are 48,342 censored, 1,033 heart disease, 923 cancer and 1,937 other.

**Table for the models** (the files in `data/medical/3_normalized/`): `participant_id`,
`cycle`, `sex_female`, `race_mexican_american`, `race_other_hispanic`, `race_black`,
`race_other` (white is the reference), `smoking_former`, `smoking_current_some_days`,
`smoking_current_every_day` (never is the reference), `age_std`, `time`, `event`.
Smoking is one status built from the two questions. The recoding is done in the
normalize step, from the merged table, and `age_std` is `age_years` scaled with the
training rows of each fold.

**Limitations.**

1. There is **no diagnosis date**. The time axis is months from the exam, and "later
   data" means people examined in a later survey cycle.
2. The public-use mortality files are **partly altered by NCHS** for disclosure
   control: synthetic values were substituted for follow-up time or underlying cause
   of death for select records. Vital status is not altered (NCHS linkage methods,
   section 4.11).
3. The three events are **grouped**. "Other" mixes very different causes, including
   stroke. Stroke is kept in "other" in every cycle on purpose: for the 2015 to 2016
   and 2017 to 2018 cycles the NCHS data dictionary says that only `UCOD_LEADING`
   values 001 (heart disease), 002 (cancer) and 010 ("All other causes (residual)") are
   available, so stroke cannot be told apart there. Putting it in "other" everywhere
   keeps the meaning of each event the same in all cycles.
4. The **later cycles are mostly censored** at 72 months and have few events. See
   "Follow-up window" below.
5. Only **four features**: age, sex, race and ethnicity, and smoking.
6. NHANES is a complex survey. **Survey weights are not used**, so results are not
   population estimates.

#### Financial data (Freddie Mac)

**Source.** Freddie Mac Single-Family Loan-Level Dataset, Standard Dataset, the
50,000-loan sample per origination year. It is free for academic and non-commercial
research but **not redistributable**. Register, then on the SFLLD Data Download page
take the **Sample File** column for each year and put the zips in
`data/financial/0_raw/` without renaming them (`sample_2013.zip`, ...).

Each zip holds two pipe-delimited files with no header row:

| file | rows | size |
|---|---|---|
| `sample_orig_YYYY.txt` | one per loan (50,000), 31 fields | about 6 MB |
| `sample_perf_YYYY.txt` | one per loan per month, 35 fields | about 475 MB |

The field positions come from the official file layout. The script asserts both field
counts and fails loudly if the layout changes, instead of silently shifting columns.

**Why this dataset for competing risks.** A mortgage can end in two ways that exclude
each other: default, or prepayment (a refinance or a sale). Prepayment is far more
common than default, so treating prepaid loans as merely censored, as single-event
survival analysis does, throws away most of the outcomes and biases the default
estimate upward.

**The two cohorts.** The data is built as two disjoint cohorts that share no loans.
The same protocol is applied to each.

| | calm | crisis |
|---|---|---|
| origination years | 2013 to 2018 | 2005 to 2009 |
| loans | 55,000 (9,167 per year) | 55,000 (11,000 per year) |
| default | 3.97% | 8.71% |
| prepaid | 60.65% | 65.20% |
| censored at 72 months | 35.38% | 26.09% |
| DTI not reported | 10.2% | 4.1% |
| file | `financial_calm_2013_2018.csv` | `financial_crisis_2005_2009.csv` |

Why two: the claim under test is that the evaluation protocol matters *because the
population shifts between training and test*. That predicts a larger effect when the
shift is larger, so the two cohorts change the size of the shift while everything
else is held fixed: same models, metrics, window, sample size and equal draw per
year.

Default rate by origination year shows the contrast:

```
calm     2013 1.7%   2014 1.6%   2015 2.9%   2016 4.1%   2017 5.3%   2018 5.2%
crisis   2005 6.4%   2006 10.2%  2007 12.9%  2008 7.6%   2009 1.8%
```

The calm cohort drifts steadily over roughly a 3x range. The crisis cohort swings over
a 7x range and reverses: 2009 loans default less often than 2005 loans, because they
were underwritten after the collapse under much tighter standards. A time-ordered
split on the crisis cohort therefore tests on an easier population. That is the
opposite of the usual intuition and is not a bug.

Both cohorts use 55,000 loans, a 72-month window and seed 0, and draw an equal number
of loans from each origination year so no year dominates. The cleaning report and a
per-column value summary are printed before the file is written. The value summary is
what catches a category that exists in one cohort and not the other: channel `T`
(third-party origination) appears in the early years and is absent from 2013 onward,
which matters when the feature is encoded.

**Columns** of the CSV in `data/financial/1_merged/`:

| column | meaning |
|---|---|
| `loan_id` | Freddie Mac loan sequence number |
| `vintage` | origination year |
| `time` | loan age in months when the outcome was observed |
| `event` | `0` censored, `1` default, `2` prepaid |
| `default_zb_alt` | `1` if a credit-loss disposition code ever appeared: the alternative, stricter default definition, kept so the choice can be tested |
| `fico` | credit score at origination |
| `ltv`, `cltv` | loan-to-value, combined loan-to-value |
| `dti` | debt-to-income |
| `dti_missing` | `1` if DTI was not reported |
| `orig_upb` | original loan amount |
| `orig_rate` | interest rate at origination |
| `orig_term` | loan term in months |
| `mi_pct` | mortgage insurance coverage in percent (`0` is none) |
| `n_units`, `n_borrowers` | property units, number of borrowers |
| `first_time_buyer` | `Y`, `N` or `9` unknown |
| `occupancy` | `P` owner-occupied, `S` second home, `I` investment |
| `channel` | `R` retail, `B` broker, `C` correspondent, `T` third party |
| `loan_purpose` | `P` purchase, `C` cash-out refinance, `N` no cash-out refinance |
| `property_type` | `SF` single family, `CO` condo, `PU` planned unit, `MH` manufactured, `CP` co-op |
| `state` | US state code |

`time` is measured from origination, not from a calendar date. That makes it
comparable across origination years and makes a train-early, test-later split
meaningful.

**Events.**

| event | code | definition |
|---|---|---|
| censored | 0 | neither event observed within the follow-up window |
| default | 1 | first month with delinquency status of 3 or more (90 or more days past due), or `RA` (property repossessed) |
| prepaid | 2 | zero-balance code `01`, if it occurs before any default |

Default is defined by delinquency, not by the final disposition, and the threshold is
**90 days**. That is the data owner's own convention for this product: FHFA and
Freddie Mac report "seriously delinquent" loans as those 90 or more days past due or in
foreclosure, and it is the headline credit metric in their monthly volume summaries.
It also matches the general Basel and CRR reference definition of default.

Two thresholds we did not use:

- **180 days.** US capital regulation
  ([12 CFR 3.101](https://www.law.cornell.edu/cfr/text/12/3.101)) puts a residential
  mortgage exposure in default at 180 days, and Basel's IRB approach allows 180 days
  for retail exposures secured by residential real estate. That threshold exists for
  sizing regulatory capital, which is not what we model. A reader from a credit-risk
  background will expect it, so state the choice in any write-up.
- **Any delinquency.** Peng and Lessmann (2026, arXiv:2601.20533) use this on the same
  sample files and call it "an early-delinquency definition", against "a conventional
  default definition based on the Basel criterion". That is a different question.

The choice moves the default rate by a factor of 12: 1.7% to 5.3% per origination year
at 90 days, against 0.1% to 0.6% if you wait for a credit-loss disposition code. Only
about 7% of 90-day delinquencies reach such a disposition within the window. The rest
cure or are still in process. If anyone asks why your default rate differs from a
published figure, look here first.

Blumenstock et al. (2022) use the same definition on this dataset: "Default is
defined as a loan turning into 3-month delinquency for the first time" (section 3.1).
So the 90-day rule is citable, and our definition matches the study whose conclusions
we are testing. One number in their paper looks like it contradicts this and does not.
Their Table 1 reports default shares of 0.1% for 2013 and 0.0% for 2016 and 2017, far
below our 1.7% to 5.3%. That table is a per-calendar-year distribution of event
occurrences, not a per-origination-year cumulative default rate, so the two are not
comparable. An earlier version of these notes wrongly inferred from it that
Blumenstock used a stricter definition.

**Event times.** The monthly files carry a loan age field that Freddie Mac resets
when a loan is modified, about 1% of loans per year. Reading event times from it makes
events look earlier than they were. The script instead tracks calendar months and
converts to loan age at the end. Measured on the 2013 and 2018 files, this affected
0.18% of loans and 4.4% of all defaults, which would otherwise be recorded as early
prepayments.

**Covariates in the models.** The normalization step drops three columns and codes
the rest:

- `ltv` is dropped in favour of `cltv`, `state` is coarsened to the nine Census
  divisions (plus territories), and `default_zb_alt` is excluded because it is an
  alternative coding of the outcome.
- The nine numeric columns (`fico`, `cltv`, `dti`, `orig_upb`, `orig_rate`,
  `orig_term`, `mi_pct`, `n_units`, `n_borrowers`) are standardized.
- The categorical columns are one-hot encoded, and the "not available" codes are
  folded into the reference level.

**Limitations.**

1. **Origination-time covariates only.** Nothing time-varying: no unemployment, house
   price index or current interest rate. Prepayment in particular is driven by the
   gap between the loan's rate and the current market rate, which is not here.
2. **The files are a 50,000-loan random sample per year**, not the full population.
   Freddie Mac draws them and we do not control the draw.
3. **Agency loans only.** They conform to Freddie Mac's underwriting standards, so the
   credit quality range is narrower than the whole mortgage market. No subprime.
4. **In the crisis cohort both competing events shift, not only default.** Interest
   rates collapsed from 2008, so prepayment behaviour changes as much as default risk.
   That makes the crisis cohort a larger distribution shift, but no result from it can
   be attributed to default risk alone.
5. **Censoring is light at 72 months**, because most loans prepay first. The competing
   risk, not censoring, is what shapes this data.

### Follow-up window (important)

Every person and every loan is **censored at 72 months**: anything that had not
happened by month 72 is recorded as censored at 72. This is ordinary administrative
censoring, set by `HORIZON_MONTHS` (medical) and `MAX_FOLLOW_UP` (financial). Without
a common cut, `time` would mean something different for each group, because a 2013
loan has been observed far longer than a 2018 one and a 1999 survey participant far
longer than one from 2017. Set `MAX_FOLLOW_UP` to `0` in the financial pipeline to
keep full follow-up.

**Why 72.** Three reasons, none of which is "it is a round number":

1. It captures most of what ever happens. Within the available follow-up, 72 months
   covers 87% of defaults and 86% of prepayments, against 76% and 76% at 60 months.
   Going further gives diminishing returns.
2. It fits inside the follow-up of the financial data: even the newest mortgage year
   (2018) can be observed for 99 months. It does **not** fully fit the medical data,
   see below.
3. Blumenstock et al. (2022) evaluate at 24, 48 and 72 months on this dataset, so our
   horizons line up with the study whose conclusions we are testing. They do not
   justify the choice themselves, so reasons 1 and 2 are ours, but matching them makes
   the results directly comparable.

**Medical data.** The 72-month cut moves 4,402 deaths after month 72 to censored.
Within 72 months there are 1,033 heart disease, 923 cancer and 1,937 other deaths. At
60 months there would be 874, 769 and 1,582, so 72 gives about 20% more events per
cause. The cost is that the later cycles cannot be followed for 72 months.
Share of people in each cycle who died or were followed for at least the horizon:

| cycle | 60 months | 72 months |
|---|---|---|
| 1999 to 2012 | 100% | 100% |
| 2013 to 2014 | 99% | 57% |
| 2015 to 2016 | 6% | 5% |
| 2017 to 2018 | 2% | 2% |

So in 2013 to 2014, 43% of people are censored before month 72, and in 2015 to 2018
nearly everyone is. The models treat this as censoring, but the last test folds rest
more on the censoring estimate, and the 2017 to 2018 fold has only about 30 heart
disease and 30 cancer events. Read their 72-month metrics with care and also look at
shorter horizons (24 and 48 months).

### Missing values and leakage

The rule everywhere: anything learned from the data (a mean, a standard deviation, a
median, a list of categories) is fitted on the **training rows of one fold only**, then
applied to its test rows. Fitting on the whole file uses test information to build the
training data, which leaks and inflates results.

**DTI is missing on purpose.** Freddie Mac did not report it consistently in the early
years: it is missing for 28.8% of 2013 loans against 1.0% of 2018 loans, and for 10.2%
of the calm cohort against 4.1% of the crisis cohort. Dropping those loans would filter
the early years far harder than the late ones and confound any train-early, test-later
comparison with the cleaning itself. So the prepare step keeps the row, leaves `dti`
empty and sets `dti_missing`. Setting `DROP_MISSING_DTI` to `True` in the financial pipeline
overrides this, but it removes about 10% of loans, mostly from the early years.

**Where the empty values are filled.** The prepare step never imputes: it sees one
cohort with no train and test boundary, so any median it computed would run over rows
that become test rows in every fold. The folds do not exist until the folds step, so
the normalize step is the earliest point where a fill can be computed from training
rows alone, and that is where it happens. It fills `dti`, `cltv` and `n_borrowers`
with the median of each fold's training rows and keeps `dti_missing` as a feature,
because whether a lender recorded DTI is informative in itself. `cltv` is empty on 9
calm loans and `n_borrowers` on 19 crisis loans. They get no flag, because a flag set
on so few rows out of 55,000 carries nothing. The medians are saved in
`scaling.json`, one set per fold and arm.

Folds are saved row positions, so they only fit the table they were made from. The
normalize step stops with an error if the row count differs. Rebuild the folds
whenever the table changes.

### Vendored model code (`vendor/`)

Two of the four models have no usable PyPI release, so their source is committed
here and not installed.

| | upstream | pinned commit |
|---|---|---|
| Deep Survival Machines | `github.com/Jeanselme/DeepSurvivalMachines` | `b29ab6dda51066e5c0b1b1f02e8bd7baca875094` |
| Neural Fine-Gray | `github.com/Jeanselme/NeuralFineGray` | `9ec6e3cacf7ddda09b02ee3af06a9e3b430ad8a1` |

Both are MIT licensed and the `LICENSE` files are kept. Removed from the vendored
copies: the upstream `.git` directories, DSM's example datasets (`dsm/datasets/*.csv`,
4.6 MB of Framingham, PBC and SUPPORT data we never load) and its generated HTML docs.
Nothing on the import path touches any of them.

```python
import sys
sys.path.insert(0, "vendor/nfg/DeepSurvivalMachines")   # DSM
sys.path.insert(0, "vendor/nfg")                        # Neural Fine-Gray
from dsm import DeepSurvivalMachines
```

**Both require float64 input.** Passing float32 fails with `RuntimeError: mat1 and
mat2 must have the same dtype`, which upstream does not document. They share a base
class, so the same applies to both.

**Do not `pip install auton-survival` to get DSM.** That package is a 2023 snapshot
pinned to `pandas<2.0` and `torch<2.0`. It cannot coexist with `comprisk` (which needs
`pandas>=2.0`), and installing it drags torch back two major versions and breaks
DeepHit. The vendored source is the same model.

Neural Fine-Gray is vendored but not currently used. It came with the same clone and
is kept in case the model set changes.

### Data licensing

The Freddie Mac dataset is free for academic use but **may not be redistributed**, so
`data/` is gitignored. The processed CSVs are derived works of licensed data, so they
must not be committed or posted publicly either. Pass them to the group directly. The
NHANES files are public, but they are also kept out of git because of their size.
