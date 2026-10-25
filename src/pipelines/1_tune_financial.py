#!/usr/bin/env python3
"""
Choose the settings of every model inside every fold of every arm, with that fold's
training rows only. Resumable: tuned cells are skipped.

Run after the prepare pipeline:
    venv/bin/python src/pipelines/1_tune_financial.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import benchmark_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
FOLDS_DIR = Path("data/financial/2_folds")
NORMALIZED_DIR = Path("data/financial/3_normalized")
TUNING_DIR = Path("data/financial/4_tuning")

COHORTS = ("calm", "crisis")

ID_COLUMN = "loan_id"
GROUP_COLUMN = "vintage"

# default and payoff
N_CAUSES = 2

# Same cap for every model so none is tuned harder than another
BUDGET = 10


def main() -> None:
    report_utils.start_log(Path(__file__).stem)
    for cohort in COHORTS:
        report_utils.h1(f"{cohort.upper()} COHORT: TUNING")
        benchmark_utils.tune(
            FOLDS_DIR / f"financial-{cohort}.json",
            NORMALIZED_DIR / cohort,
            ID_COLUMN,
            GROUP_COLUMN,
            N_CAUSES,
            BUDGET,
            TUNING_DIR / f"{cohort}.json",
        )


if __name__ == "__main__":
    main()
