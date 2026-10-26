#!/usr/bin/env python3
"""
Prepare the financial data (Freddie Mac single-family mortgages), both cohorts.

For each cohort (calm and crisis) it runs three steps in order, each one printing what
it does:
  1. prepare: read the raw archives, clean them and save one CSV,
  2. folds: build the kfold, temporal and matched folds and check them for leaks,
  3. normalize: impute, scale and encode inside each fold, training rows only.

Edit the folders and settings below, then run from the project folder:
    venv/bin/python src/pipelines/0_prepare_financial.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import financial_utils, fold_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
# Raw archives, named sample_YYYY.zip
RAW_DIR = Path("data/financial/0_raw")
NORMALIZED_DIR = Path("data/financial/3_normalized")

# The two cohorts share no loans and get the same protocol
COHORTS = {
    "calm": {
        "years": [2013, 2014, 2015, 2016, 2017, 2018],
        "max_rows": 55_000,
        "table": Path("data/financial/1_merged/financial_calm_2013_2018.csv"),
        "folds": Path("data/financial/2_folds/financial-calm.json"),
    },
    "crisis": {
        "years": [2005, 2006, 2007, 2008, 2009],
        "max_rows": 55_000,
        "table": Path("data/financial/1_merged/financial_crisis_2005_2009.csv"),
        "folds": Path("data/financial/2_folds/financial-crisis.json"),
    },
}

# Censor every loan at this age in months; 0 keeps full follow-up
FOLLOW_UP_MONTHS = 72

# True drops loans with no DTI instead of flagging them
DROP_MISSING_DTI = False

# Seed for the subsample and the folds
SEED = 0


@report_utils.logged
def prepare_financial() -> None:
    """Run every step of this process."""
    for cohort, cfg in COHORTS.items():
        report_utils.h1(f"{cohort} cohort: prepare")
        report_utils.h2("Read and clean")
        financial_utils.prepare_cohort(
            cohort,
            RAW_DIR,
            cfg["years"],
            cfg["max_rows"],
            FOLLOW_UP_MONTHS,
            SEED,
            DROP_MISSING_DTI,
            cfg["table"],
        )

        report_utils.h2("Folds")
        fold_utils.make_folds(
            cfg["table"],
            financial_utils.GROUP_COLUMN,
            financial_utils.ID_COLUMN,
            "event",
            f"financial-{cohort}",
            cfg["folds"],
            SEED,
        )

        report_utils.h2("Normalize inside each fold")
        financial_utils.normalize_cohort(
            cohort, cfg["table"], cfg["folds"], NORMALIZED_DIR
        )

        report_utils.h2("Fold Summary")
        fold_utils.show_folds(
            cfg["table"], financial_utils.ID_COLUMN, "event", cfg["folds"]
        )


if __name__ == "__main__":
    prepare_financial()
