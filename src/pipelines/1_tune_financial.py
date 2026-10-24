#!/usr/bin/env python3
"""
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
NORMALIZED_DIR = Path("data/financial/3_normalized")
TUNING_DIR = Path("data/financial/4_tuning")

# (cohort, number of temporal folds)
COHORTS = (("calm", 5), ("crisis", 4))

ID_COLUMN = "loan_id"
GROUP_COLUMN = "vintage"

# default and payoff
N_CAUSES = 2

# Same cap for every model so none is tuned harder than another
BUDGET = 10


def main() -> None:
    for cohort, n_folds in COHORTS:
        report_utils.h1(f"{cohort.upper()} COHORT: TUNING")
        benchmark_utils.tune(
            NORMALIZED_DIR / cohort,
            "temporal",
            n_folds,
            ID_COLUMN,
            GROUP_COLUMN,
            N_CAUSES,
            BUDGET,
            TUNING_DIR / f"{cohort}.json",
        )


if __name__ == "__main__":
    main()
