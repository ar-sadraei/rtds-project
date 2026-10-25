#!/usr/bin/env python3
"""
Benchmark the four models on the financial cohorts, all three arms, every fold.

Reads the folds that 0_prepare_financial.py materialized and the settings that
1_tune_financial.py chose. The deep models run once per seed. Writes one row per
number, so the analysis can group by arm, model, cause, horizon or metric without
reshaping, plus the calibration bins and the test predictions of every fit.

The run is resumable: finished fits are skipped.

    venv/bin/python src/pipelines/2_benchmark_financial.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from utils import benchmark_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
FOLDS_DIR = Path("data/financial/2_folds")
NORMALIZED_DIR = Path("data/financial/3_normalized")
TUNING_DIR = Path("data/financial/4_tuning")
BENCHMARK_DIR = Path("data/financial/5_benchmark")

COHORTS = ("calm", "crisis")

ID_COLUMN = "loan_id"
GROUP_COLUMN = "vintage"

# default and payoff
N_CAUSES = 2

# Initialization seeds of the deep models; the classical models run once
SEEDS = (0, 1, 2)


@report_utils.logged
def benchmark_financial() -> None:
    """Run every step of this process."""
    for cohort in COHORTS:
        report_utils.h1(f"{cohort} cohort: benchmark")
        params = benchmark_utils.read_params(TUNING_DIR / f"{cohort}.json")
        benchmark_utils.run(
            f"financial-{cohort}",
            FOLDS_DIR / f"financial-{cohort}.json",
            NORMALIZED_DIR / cohort,
            ID_COLUMN,
            GROUP_COLUMN,
            N_CAUSES,
            params,
            SEEDS,
            BENCHMARK_DIR,
        )


if __name__ == "__main__":
    benchmark_financial()
