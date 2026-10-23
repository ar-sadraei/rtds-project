#!/usr/bin/env python3
"""
Benchmark the four models on the financial cohorts, both arms, every fold.

Reads the folds that 0_prepare_financial.py materialized and the hyperparameters
that 1_tune_financial.py chose. Writes one row per number, so the analysis can
group by arm, model, cause, horizon or metric without reshaping.

The run is resumable: finished cells are skipped.

    venv/bin/python src/pipelines/2_benchmark_financial.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from utils import benchmark_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
NORMALIZED_DIR = Path("data/financial/3_normalized")
TUNING_DIR = Path("data/financial/4_tuning")
BENCHMARK_DIR = Path("data/financial/5_benchmark")

# (cohort, number of temporal folds)
COHORTS = (("calm", 5), ("crisis", 4))

ID_COLUMN = "loan_id"
GROUP_COLUMN = "vintage"

# default and payoff
N_CAUSES = 2


def main() -> None:
    for cohort, n_folds in COHORTS:
        print("=" * 92)
        print(f"{cohort.upper()} COHORT, {n_folds} folds per arm")
        print("=" * 92)
        params = benchmark_utils.read_params(TUNING_DIR / f"{cohort}.json")
        benchmark_utils.run(
            f"financial-{cohort}",
            NORMALIZED_DIR / cohort,
            n_folds,
            ID_COLUMN,
            GROUP_COLUMN,
            N_CAUSES,
            params,
            BENCHMARK_DIR / f"{cohort}.csv",
        )


if __name__ == "__main__":
    main()
