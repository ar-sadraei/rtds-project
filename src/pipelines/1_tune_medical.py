#!/usr/bin/env python3
"""
Choose the settings of every model inside every fold of every arm, with that fold's
training rows only. Resumable: tuned cells are skipped.

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
FOLDS_FILE = Path("data/medical/2_folds/medical.json")
NORMALIZED_DIR = Path("data/medical/3_normalized")
TUNING_DIR = Path("data/medical/4_tuning")

ID_COLUMN = "participant_id"
GROUP_COLUMN = "cycle"

# cardiovascular death, cancer death, other causes
N_CAUSES = 3

# Same cap for every model so none is tuned harder than another
BUDGET = 10


def main() -> None:
    report_utils.start_log(Path(__file__).stem)
    report_utils.h1("MEDICAL: TUNING")
    benchmark_utils.tune(
        FOLDS_FILE,
        NORMALIZED_DIR,
        ID_COLUMN,
        GROUP_COLUMN,
        N_CAUSES,
        BUDGET,
        TUNING_DIR / "medical.json",
    )


if __name__ == "__main__":
    main()
