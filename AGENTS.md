# Rules for agents working in this repo

Project: benchmark of deep and classical competing-risks survival models on two
datasets, medical (NHANES) and financial (Freddie Mac). Course: Research Topics in
Data Science, Stockholm University.

## Pipeline

- No Jupyter notebooks. Everything is Python files.
- Code lives in `src/`. Functions go in `src/utils/`, grouped by content:
  `medical_utils.py`, `financial_utils.py`, `fold_utils.py`.
- Runnable scripts go in `src/pipelines/`, one per process and domain, numbered in
  run order from 0. Each one runs all the steps of its process. For example
  `0_prepare_medical.py` merges, builds the folds and normalizes.
- Later processes are separate pipelines in the same folder, still split by domain
  and numbered in order: tune, then benchmark, then analyze.
- A pipeline holds the folders and settings and chains calls to the utils. Keep the
  logic in the utils.
- The medical and the financial data follow the same steps and the same table
  contract: an id, a group column (`cycle` or `vintage`), `time`, `event` and the
  features. Only the prepare step is specific to a dataset.
- Run Python with the project environment, from the project folder, for example
  `venv/bin/python src/pipelines/0_prepare_medical.py`.
- Do not use command line arguments (no `argparse`). Pipelines run with no options.
- Format all Python code with `black`.

## Data

- Everything a script writes goes under `data/<dataset>/`, in numbered folders such
  as `0_raw`, `1_merged`, `2_folds`, `3_normalized`. Do not write to
  a `results/` folder.
- Put folders and settings in a section at the top of each pipeline, as constants.
  Paths are relative to the project folder, and the seed is fixed in the code.
- Create output folders in the script, so it works on a fresh checkout.
- Anything learned from the data (scaling, imputation) is fitted on the training
  rows of each fold only, then applied to the test rows.
- Folds are saved row positions, so rebuild them whenever the table changes.
- Call the medical dataset "medical", not "clinical".
- Every cleaning or preprocessing step prints what it did and how many rows it
  removed.

## Comments

- Few comments. One short sentence.
- Put the comment on the line above the code, never at the end of a line.
- Do not explain what the code already says.

## Writing

- Short, direct sentences. Define a term when it is first used.
- No long dashes (em dash or en dash). Use commas, periods or parentheses.
  Write ranges as "1999 to 2000".

## Git

- Do not commit or push unless asked. Upstream may change.
- Do not edit `vendor/`. It is other people's code.
