#!/usr/bin/env python3
"""
Run after the prepare pipeline:
    venv/bin/python src/pipelines/1_tune_medical.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import benchmark_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
NORMALIZED_DIR = Path("data/medical/3_normalized")
TUNING_DIR = Path("data/medical/4_tuning")

N_FOLDS = 7

ID_COLUMN = "participant_id"
GROUP_COLUMN = "cycle"

# cardiovascular death, cancer death, other causes
N_CAUSES = 3

# Same cap for every model so none is tuned harder than another
BUDGET = 10


def main() -> None:
    report_utils.h1("MEDICAL: TUNING")
    benchmark_utils.tune(
        NORMALIZED_DIR,
        "temporal",
        N_FOLDS,
        ID_COLUMN,
        GROUP_COLUMN,
        N_CAUSES,
        BUDGET,
        TUNING_DIR / "medical.json",
    )


if __name__ == "__main__":
    main()
