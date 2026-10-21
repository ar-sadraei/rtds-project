# Financial dataset — competing risks

Dataset preparation for the group project in *Research Topics in Data Science*
(DSV, Stockholm University, 2026). Ali Emre Nebiler, Amirreza Sadraei,
Filip Becic.

This repository holds **only the financial dataset pipeline**. It turns the raw
Freddie Mac mortgage files into one clean CSV with a competing-risks outcome,
for use alongside the clinical dataset (NHANES).

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

```bash
~/venvs/rtds/bin/python src/prepare_financial_data.py
```

Needs `data/banking/0_raw/sample_YYYY.zip` for each year. `DATA.md` says where
to get them. Output: `data/banking/1_merged/financial_competing_risks.csv`.
Both paths are set at the top of the script.

Defaults: 2013–2018, 50,000 loans, follow-up cut at 60 months, seed 0.

```bash
~/venvs/rtds/bin/python src/prepare_financial_data.py --max-rows 250000
~/venvs/rtds/bin/python src/prepare_financial_data.py --years 2016 2017 2018 --max-follow-up 36
```

The script prints a cleaning report before it writes. That report is the audit
trail — read it rather than trusting the file.

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
consistently before 2016 — missing for 28.8% of 2013 loans against 1.0% of 2018
loans. Those rows are kept with `dti` empty and `dti_missing` set, rather than
dropped, because dropping them would filter the early years far harder than the
late ones and confound any train-early / test-later comparison. If your model
cannot take an empty value, impute from the **training fold only** — imputing
from the whole file leaks test information into training.

**Follow-up is cut at 60 months.** A 2013 loan has been observed far longer than
a 2018 one. Without a common cut, `time` means something different for each
year. `--max-follow-up 0` keeps full follow-up.
