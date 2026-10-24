#!/usr/bin/env python3
"""
The run is resumable: finished cells are skipped.

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
NORMALIZED_DIR = Path("data/medical/3_normalized")
TUNING_DIR = Path("data/medical/4_tuning")
BENCHMARK_DIR = Path("data/medical/5_benchmark")

N_FOLDS = 7

ID_COLUMN = "participant_id"
GROUP_COLUMN = "cycle"

# cardiovascular death, cancer death, other causes
N_CAUSES = 3


def main() -> None:
    report_utils.h1(f"MEDICAL: BENCHMARK, {N_FOLDS} FOLDS PER ARM")
    params = benchmark_utils.read_params(TUNING_DIR / "medical.json")
    benchmark_utils.run(
        "medical",
        NORMALIZED_DIR,
        N_FOLDS,
        ID_COLUMN,
        GROUP_COLUMN,
        N_CAUSES,
        params,
        BENCHMARK_DIR / "medical.csv",
    )


if __name__ == "__main__":
    main()
