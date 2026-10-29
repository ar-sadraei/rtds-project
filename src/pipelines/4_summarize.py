#!/usr/bin/env python3
"""
Summarize the analysis as tables for the three hypotheses.

Reads the tables and the bootstrap draws that 3_analyze.py saved, so run that first.
It fits nothing and takes about a minute. The tables are printed and written to
data/analysis/6_summary/ as CSV files.

    venv/bin/python src/pipelines/4_summarize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import report_utils
from utils import summary_utils as S

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
ANALYSIS_DIR = Path("data/analysis")
BOOTSTRAP_DIR = ANALYSIS_DIR / "0_bootstrap"
MAIN_DIR = ANALYSIS_DIR / "2_main_question"
SQ1_DIR = ANALYSIS_DIR / "3_sq1_later_data"
SQ2_DIR = ANALYSIS_DIR / "4_sq2_ranking_vs_probability"
SUPPORT_DIR = ANALYSIS_DIR / "5_supporting"
OUTPUT_DIR = ANALYSIS_DIR / "6_summary"

# Dataset label and its saved folds
DATASETS = (
    ("medical", Path("data/medical/2_folds/medical.json")),
    ("financial-calm", Path("data/financial/2_folds/financial-calm.json")),
    ("financial-crisis", Path("data/financial/2_folds/financial-crisis.json")),
)


def show(title, frame, name):
    """Print a table under a sub title and write it to the output folder."""
    report_utils.h2(title)
    report_utils.table(frame, decimals=4)
    frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


@report_utils.logged
def summarize() -> None:
    """Run every step of this process."""
    report_utils.h1("SUMMARIZE")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = [S.model_level(label, folds, BOOTSTRAP_DIR) for label, folds in DATASETS]
    comparison = pd.concat([c for c, _ in parts], ignore_index=True)
    later = pd.concat([l for _, l in parts], ignore_index=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison_pvalues.csv", index=False)
    later.to_csv(OUTPUT_DIR / "h2_model_level_raw.csv", index=False)

    show(
        "Temporal split: advantage of the deep model and p-value",
        S.temporal_table(comparison),
        "model_comparison_temporal",
    )
    h1_pairs = S.winner_counts(comparison)
    h1_reversals = S.reversals(pd.read_csv(MAIN_DIR / "h1_reversal_cells.csv"))
    h2_model = S.later_data_table(later)
    h2_counts = S.later_data_counts(pd.read_csv(SQ1_DIR / "h2_arm_difference.csv"))
    h3 = S.ranking_table(pd.read_csv(SQ2_DIR / "ranking_agreement.csv"))
    friedman = S.friedman_table(pd.read_csv(SUPPORT_DIR / "friedman.csv"))
    show("H1: pairs significantly better", h1_pairs, "h1_pairs")
    show("H1: reversals", h1_reversals, "h1_reversals")
    show("H2: temporal minus matched, per model pair", h2_model, "h2_model_level")
    show("H2: cells with a smaller or larger advantage", h2_counts, "h2_counts")
    show("H3: ranking versus probability", h3, "h3_ranking")
    show("Friedman: best model by mean rank over folds", friedman, "friedman")

    report_utils.log(f"Saved tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    summarize()
