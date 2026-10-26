# Competing-risks survival models on medical and financial data

## Quick start

**1. Environment.** Create an environment and install dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Input.** The data is not in the repository. Download it and put it in the `0_raw`
folders (see [Data](#data)):

| dataset   | folder                           | content                                        |
| --------- | -------------------------------- | ---------------------------------------------- |
| medical   | `data/medical/0_raw/demography/` | one `.xpt` file per cycle                      |
| medical   | `data/medical/0_raw/smoking/`    | one `.xpt` file per cycle                      |
| medical   | `data/medical/0_raw/mortality/`  | one `.dat` file per cycle                      |
| financial | `data/financial/0_raw/`          | `sample_YYYY.zip` for every year, not unzipped |

- NHANES survey files: <https://wwwn.cdc.gov/nchs/nhanes/>
- NCHS linked mortality files: <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/>
- Freddie Mac loan-level data (free, register first): <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>

The Freddie Mac data may not be redistributed, so `data/` is not tracked by git.

**3. Run.** Run the pipelines in this order:

```bash
python src/pipelines/0_prepare_medical.py
python src/pipelines/0_prepare_financial.py
python src/pipelines/1_tune_medical.py
python src/pipelines/1_tune_financial.py
python src/pipelines/2_benchmark_medical.py
python src/pipelines/2_benchmark_financial.py
python src/pipelines/3_analyze.py
```

Or run all of them in this order with `python src/main.py`.

**4. Output.** Every step writes into the numbered folders of `data/` (see
[Data folder](#data-folder)), and the final tables are in `data/analysis/`. Each run
also writes what it prints to `logs/<pipeline name>_<date>_<time>.log`. The slow steps
(tune, benchmark and the analysis) skip the work already saved, so an interrupted run
continues where it stopped.

## Project structure

```
src/
  main.py                     runs all the pipelines in order
  pipelines/                  runnable scripts, numbered in run order
    0_prepare_*.py            1_tune_*.py          2_benchmark_*.py       3_analyze.py
  utils/                      the functions the pipelines call
data/                         input and output, not in git
logs/                         one log file per run, not in git
vendor/                       other people's model code
requirements.txt              Python packages
AGENTS.md                     rules for people and agents working in the repo
```

Pipelines are split by dataset (`*_medical.py` and `*_financial.py`), except the
analysis, which reads both.

### Data

Each dataset has its own folder, and every step writes to the next numbered folder.

```
data/
  medical/     0_raw  1_merged  2_folds  3_normalized  4_tuning  5_benchmark
  financial/   0_raw  1_merged  2_folds  3_normalized  4_tuning  5_benchmark
  analysis/    result tables, one folder per question
```

| folder         | content                          | written by    |
| -------------- | -------------------------------- | ------------- |
| `0_raw`        | the downloaded files (the input) | you           |
| `1_merged`     | one clean table per dataset      | `0_prepare`   |
| `2_folds`      | the saved train and test splits  | `0_prepare`   |
| `3_normalized` | the tables used by the models    | `0_prepare`   |
| `4_tuning`     | the best settings per model      | `1_tune`      |
| `5_benchmark`  | scores and predictions per model | `2_benchmark` |
| `analysis/`    | the result tables                | `3_analyze`   |

### Notes

- `hazardous` is installed from a git commit, not a version. CR D-calibration is in
  no release yet, and the commit reports the same version as the release.
- `numba`, `numpy` and `pandas` are pinned below the latest because `comprisk` and
  `lifelines` require it.
- Deep Survival Machines (DSM) has no usable PyPI release, so its source is in
  `vendor/DeepSurvivalMachines`. Do not edit it, and do not install `auton-survival`
  (it pins old `pandas` and `torch` and breaks the other packages).
