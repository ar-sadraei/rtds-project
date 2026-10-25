#!/usr/bin/env python3
"""
Benchmark the four models on the medical data, all three arms, every fold.

Reads the folds that 0_prepare_medical.py materialized and the settings that
1_tune_medical.py chose. The deep models run once per seed. Writes one row per number,
the calibration bins and the test predictions of every fit.

The run is resumable: finished fits are skipped.

    venv/bin/python src/pipelines/2_benchmark_medical.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from utils import benchmark_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
FOLDS_FILE = Path("data/medical/2_folds/medical.json")
NORMALIZED_DIR = Path("data/medical/3_normalized")
TUNING_DIR = Path("data/medical/4_tuning")
BENCHMARK_DIR = Path("data/medical/5_benchmark")

ID_COLUMN = "participant_id"
GROUP_COLUMN = "cycle"

# cardiovascular death, cancer death, other causes
N_CAUSES = 3

# Initialization seeds of the deep models; the classical models run once
SEEDS = (0, 1, 2)


@report_utils.logged
def benchmark_medical() -> None:
    """Run every step of this process."""
    report_utils.h1("MEDICAL: BENCHMARK")
    params = benchmark_utils.read_params(TUNING_DIR / "medical.json")
    benchmark_utils.run(
        "medical",
        FOLDS_FILE,
        NORMALIZED_DIR,
        ID_COLUMN,
        GROUP_COLUMN,
        N_CAUSES,
        params,
        SEEDS,
        BENCHMARK_DIR,
    )


if __name__ == "__main__":
    benchmark_medical()
