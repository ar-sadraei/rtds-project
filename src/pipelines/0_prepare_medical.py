#!/usr/bin/env python3
"""
Prepare the medical data (NHANES demography, smoking and linked mortality, 2019),
as one cohort.

Runs three steps in order, each one printing what it does:
  1. merge the raw files, clean them and build the time and event labels,
  2. build the folds (kfold, temporal and matched) and check them for leaks,
  3. recode the columns and normalize inside each fold, using that fold's training
     rows only.

Edit the folders and settings below, then run from the project folder:
    venv/bin/python src/pipelines/0_prepare_medical.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import fold_utils, medical_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
DEMOGRAPHY_DIR = Path("data/medical/0_raw/demography")
SMOKING_DIR = Path("data/medical/0_raw/smoking")
MORTALITY_DIR = Path("data/medical/0_raw/mortality")
NORMALIZED_DIR = Path("data/medical/3_normalized")

# One cohort. Mortality is linked through 2019, so nobody interviewed in 2015 to 2018
# reaches 72 months of follow-up and those cycles are left out.
COHORTS = {
    "medical": {
        "cycles": [
            "1999-2000",
            "2001-2002",
            "2003-2004",
            "2005-2006",
            "2007-2008",
            "2009-2010",
            "2011-2012",
            "2013-2014",
        ],
        "table": Path("data/medical/1_merged/nhanes_merged.csv"),
        "folds": Path("data/medical/2_folds/medical.json"),
    },
}

# Mortality is only linked for adults
MIN_AGE_YEARS = 20

# Censor every person at this age in months; 0 keeps full follow-up
FOLLOW_UP_MONTHS = 72

# Seed for the folds
SEED = 0


@report_utils.logged
def prepare_medical() -> None:
    """Run every step of this process."""
    for cohort, cfg in COHORTS.items():
        report_utils.h1(f"{cohort} cohort: prepare")
        report_utils.h2("Merge, clean and label")
        df = medical_utils.merge_and_clean(
            DEMOGRAPHY_DIR, SMOKING_DIR, MORTALITY_DIR, MIN_AGE_YEARS
        )
        df = medical_utils.keep_cycles(df, cfg["cycles"])
        df = medical_utils.add_labels(df, FOLLOW_UP_MONTHS)
        cfg["table"].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg["table"], index=False)
        report_utils.log(f"Saved {cfg['table']}")

        report_utils.h2("Folds")
        fold_utils.make_folds(
            cfg["table"],
            medical_utils.GROUP_COLUMN,
            medical_utils.ID_COLUMN,
            "event",
            cohort,
            cfg["folds"],
            SEED,
        )

        report_utils.h2("Recode and normalize inside each fold")
        medical_utils.normalize(cfg["table"], cfg["folds"], NORMALIZED_DIR)

        report_utils.h2("Fold Summary")
        fold_utils.show_folds(
            cfg["table"], medical_utils.ID_COLUMN, "event", cfg["folds"]
        )


if __name__ == "__main__":
    prepare_medical()
