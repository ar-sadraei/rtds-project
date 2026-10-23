# Financial dataset — competing risks

Dataset preparation for the group project in *Research Topics in Data Science*
(DSV, Stockholm University, 2026). Ali Emre Nebiler, Amirreza Sadraei,
Filip Becic.

This repository holds **only the financial dataset pipeline**. It turns the raw
Freddie Mac mortgage files into one clean CSV with a competing-risks outcome,
for use alongside the medical dataset (NHANES).

## What a competing-risks dataset is

Every loan ends in one of three states, and they exclude each other:

| `event` | meaning |
|---|---|
| `0` | **censored** — still performing when we stopped looking |
| `1` | **default** — borrower stopped paying |
| `2` | **prepaid** — paid off early (refinanced or house sold) |

`time` is the loan's age **in months** when that happened. Once a loan defaults
it can never prepay, and the reverse — that mutual exclusion is the competing
risk. Prepayment is roughly 18× more common than default here, so treating
prepaid loans as merely "censored" would discard most of the outcomes.

## Environment

**Keep the virtualenv outside iCloud.** A venv inside the iCloud-synced project
folder is ~36,000 files; syncing them makes Python startup take minutes.

```bash
python3 -m venv ~/venvs/rtds
~/venvs/rtds/bin/pip install -r requirements.txt comprisk \
  "git+https://github.com/soda-inria/hazardous.git@26e64323fb462cc283bf145f3e37af4f78e1a5d4"
```

That hazardous commit is **PR #100** (`soda-inria/hazardous`), which adds
CR D-calibration — the proper distributional calibration measure for competing
risks. It is not in any release. It reports its version as `0.2.0`, identical to
the released package, so **pin the commit or nobody can tell which one they
have**. Everything in the release is also present in it.

Note that `comprisk` caps `numba<0.66`, so installing it moves numba and numpy
back a version from what `requirements.txt` pins. After installing, check
nothing broke:

```bash
~/venvs/rtds/bin/python -c "import torch, numpy; from pycox.models import DeepHit; from comprisk import FineGrayRegression; from hazardous.metrics import d_cr_calibration; print('ok', torch.__version__, numpy.__version__)"
```

## Build it

The financial data is **two disjoint cohorts**, not one dataset. They share no
loans and the same protocol is applied to each.

```bash
~/venvs/rtds/bin/python src/0_prepare_financial_data.py --cohort calm    # ordinary drift
~/venvs/rtds/bin/python src/0_prepare_financial_data.py --cohort crisis  # a structural break
```

`--cohort` is required and sets the origination years, the row cap and the
output path together, since those three are what distinguish one cohort from
the other rather than three independent settings. `--years`, `--max-rows` and
`--out` still override it for a one-off check.

| | calm 2013–2018 | crisis 2005–2009 |
|---|---|---|
| loans | 55,000 | 55,000 |
| default | 3.97% | 8.71% |
| prepaid | 60.65% | 65.20% |
| censored | 35.38% | 26.09% |

Both need `data/banking/0_raw/sample_YYYY.zip` for every year in range;
`DATA.md` says where to get them. Both use a 72-month window, seed 0, and draw
an equal number of loans per origination year.

**Why two cohorts.** The claim under test is that the evaluation protocol
matters *because the population shifts between training and test*. That
predicts a bigger effect when the shift is bigger, so the two cohorts
manipulate the size of the shift while holding everything else fixed. Default
rates drift over roughly a 3x range in the calm cohort and a 7x range in the
crisis one.

The script prints a cleaning report and a per-column value summary before
writing. Read them rather than trusting the file.

## The pipeline

Both domains run the same three numbered stages, and every stage writes to disk
so the next one reads a file rather than repeating the work. Run them in order.

```bash
~/venvs/rtds/bin/python src/0_prepare_financial_data.py --cohort calm
~/venvs/rtds/bin/python src/0_prepare_financial_data.py --cohort crisis
~/venvs/rtds/bin/python src/0_prepare_medical_data.py
~/venvs/rtds/bin/python src/1_make_folds.py               # all three datasets at once
~/venvs/rtds/bin/python src/2_normalize_financial.py
~/venvs/rtds/bin/python src/2_normalize_medical.py
```

| stage | financial | medical |
|---|---|---|
| 0 raw archives | `data/banking/0_raw/` | `data/medical/0_raw/` |
| 0 prepared table | `data/banking/1_merged/` | `data/medical/1_merged/`, then `2_preprocessed/` |
| 1 folds | `data/banking/2_folds/` | `data/medical/3_folds/` |
| 2 per-fold train/test | `data/banking/3_normalized/{cohort}/{arm}/` | `data/medical/4_normalized/{arm}/` |

Stage 1 builds the folds **once** and writes row positions, so every model sees
identical splits and comparisons between models are paired. Stage 2 then
materialises each fold as a `fold_N_train.csv` / `fold_N_test.csv` pair, with
every statistic — the scaling, the imputation, the category lists — derived from
the training rows of that fold alone. The model runner therefore reads a file
and fits; nothing downstream can re-split or re-scale, which is the only way a
leak can be ruled out rather than hoped for. The numbers used are kept in
`scaling.json` next to the folds.

The financial stage 2 does two things its medical counterpart does not need, for
the same train-only reason: it fills missing `dti` with the training median
(keeping `dti_missing` as the signal) and one-hot encodes the categorical
covariates. Its header explains the three covariate decisions it encodes —
`ltv` dropped for `cltv`, `state` coarsened to the nine Census divisions, and
`default_zb_alt` excluded as an alternative coding of the outcome.

Stage 2 writes about 160 MB of financial folds. Nothing under `data/` is
tracked, so this stays local.

## Vendored model code (`vendor/`)

Two of the four models have no usable PyPI release, so their source is
committed here rather than installed.

| | upstream | pinned commit |
|---|---|---|
| Deep Survival Machines | `github.com/Jeanselme/DeepSurvivalMachines` | `b29ab6dda51066e5c0b1b1f02e8bd7baca875094` |
| Neural Fine-Gray | `github.com/Jeanselme/NeuralFineGray` | `9ec6e3cacf7ddda09b02ee3af06a9e3b430ad8a1` |

Both MIT licensed; the `LICENSE` files are retained. Removed from the vendored
copies: the upstream `.git` directories, DSM's example datasets
(`dsm/datasets/*.csv`, 4.6 MB of Framingham/PBC/SUPPORT data we never load) and
its generated HTML docs. Nothing on the import path touches any of them.

```python
import sys
sys.path.insert(0, "vendor/nfg/DeepSurvivalMachines")   # DSM
sys.path.insert(0, "vendor/nfg")                        # Neural Fine-Gray
from dsm import DeepSurvivalMachines
```

**Both require float64 input.** Passing float32 fails with
`RuntimeError: mat1 and mat2 must have the same dtype`, which is not documented
upstream. They share a base class, so the same applies to both.

**Do not `pip install auton-survival` to get DSM.** That package is a 2023
snapshot pinned to `pandas<2.0` and `torch<2.0`. It cannot coexist with
`comprisk` (which needs `pandas>=2.0`), and installing it drags torch back two
major versions and breaks DeepHit. The vendored source above is the same model.

Neural Fine-Gray is vendored but not currently used; it came with the same
clone and is kept in case the model set changes.

## The data is not in this repository

The Freddie Mac dataset is free for academic use but **may not be
redistributed**, so `data/` is gitignored. The processed CSV is a derived work
of licensed data, so it must not be committed or posted publicly either. Pass it
to the group directly.

## Two things to know before modelling

**DTI has missing values on purpose.** Freddie Mac did not report it
consistently in the earlier years — missing for 28.8% of 2013 loans against
1.0% of 2018 loans, and for 10.2% of the calm cohort against 4.1% of the crisis
cohort. Those rows are kept with `dti` empty and `dti_missing` set, rather than
dropped, because dropping them would filter the early years far harder than the
late ones and confound any train-early / test-later comparison. If your model
cannot take an empty value, impute from the **training fold only** — imputing
from the whole file leaks test information into training.

**Follow-up is cut at 72 months.** A 2013 loan has been observed far longer than
a 2018 one. Without a common cut, `time` means something different for each
year. `--max-follow-up 0` keeps full follow-up.
