#!/usr/bin/env python3
"""
Prepare the medical data (NHANES demography, smoking and linked mortality, 2019).

Runs three steps in order, each one printing what it does:
  1. merge the raw files, clean them and build the time and event labels,
  2. build the folds (temporal and random) and check them for leaks,
  3. recode the columns and normalize inside each fold, using that fold's training
     rows only.

Edit the folders and settings below, then run from the project folder:
    venv/bin/python src/pipelines/0_prepare_medical.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import fold_utils, medical_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
DEMOGRAPHY_DIR = Path("data/medical/0_raw/demography")
SMOKING_DIR = Path("data/medical/0_raw/smoking")
MORTALITY_DIR = Path("data/medical/0_raw/mortality")
MERGED_FILE = Path("data/medical/1_merged/nhanes_merged.csv")
FOLDS_FILE = Path("data/medical/2_folds/medical.json")
NORMALIZED_DIR = Path("data/medical/3_normalized")

# Mortality is only linked for adults
MIN_AGE_YEARS = 20

# Cut follow-up at this many months
HORIZON_MONTHS = 72

SEED = 0


def main() -> None:
    print("=== Step 1: merge, clean and label ===")
    df = medical_utils.merge_and_clean(
        DEMOGRAPHY_DIR, SMOKING_DIR, MORTALITY_DIR, MIN_AGE_YEARS
    )
    medical_utils.show_preview(df)
    df = medical_utils.add_labels(df, HORIZON_MONTHS)
    MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_FILE, index=False)
    print(f"\nSaved {MERGED_FILE.resolve()}")

    print("\n=== Step 2: folds ===")
    fold_utils.make_folds(
        MERGED_FILE,
        medical_utils.GROUP_COLUMN,
        medical_utils.ID_COLUMN,
        "event",
        "medical",
        FOLDS_FILE,
        SEED,
    )

    print("\n=== Step 3: recode and normalize inside each fold ===")
    medical_utils.normalize(MERGED_FILE, FOLDS_FILE, NORMALIZED_DIR)


if __name__ == "__main__":
    main()
