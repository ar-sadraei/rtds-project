#!/usr/bin/env python3
"""
Analyze the benchmark: the Friedman test for H1.

H0: the relative performance of DL and classical models does not change with the
evaluation metric.
H1: it does change.

The relative performance is the gap, DL group minus classical group, standardized per
metric. The test is a Friedman test over the metrics (10 blocks, the domain and fold
pairs). H1 is supported when its p value is below ALPHA, and then the Nemenyi test
compares the metrics in pairs. The folds share training data,
so the p value is approximate. A significant result does not say where the gap changes,
and the claim covers only these cells.

    venv/bin/python src/pipelines/3_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import report_utils
from utils.analysis_utils import (
    cell_table,
    fold_gaps,
    friedman_metric,
    load,
    missing_values,
    nemenyi,
    seed_mean,
)

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
BENCHMARK_FILES = {
    "medical": Path("data/medical/5_benchmark/medical.csv"),
    "financial-calm": Path("data/financial/5_benchmark/financial-calm.csv"),
}
OUTPUT_DIR = Path("data/analysis")
TESTS_FILE = OUTPUT_DIR / "h1_friedman.csv"
CELLS_FILE = OUTPUT_DIR / "h1_cells.csv"
NEMENYI_FILE = OUTPUT_DIR / "h1_nemenyi.csv"

ARM = "kfold"
DATASETS = ("medical", "financial-calm")
METRICS = ("scaled_brier", "concordance", "calibration_gap", "cr_d_calibration")
HORIZONS = (12.0, 36.0, 60.0)
ALPHA = 0.05


def save(frame: pd.DataFrame, path: Path) -> None:
    """Write a table as CSV with no newline after the last row."""
    path.write_text(frame.to_csv(index=False).rstrip("\n"))


@report_utils.logged
def analyze() -> None:
    """Run every step of this process."""
    report_utils.h1("ANALYZE")
    report_utils.log(f"Split: {ARM}")
    report_utils.log(f"Datasets: {', '.join(DATASETS)}")
    report_utils.log(f"Metrics: {', '.join(METRICS)}")
    report_utils.log(f"Horizons: {', '.join(f'{h:g}' for h in HORIZONS)}")

    raw = load(BENCHMARK_FILES[d] for d in DATASETS)
    missing = missing_values(raw)
    report_utils.log(f"Benchmark rows without a value: {int(missing.missing.sum())}")

    per_fold, dropped = fold_gaps(seed_mean(raw), ARM, DATASETS, METRICS, HORIZONS)
    for (dataset, metric), n in per_fold.groupby(["dataset", "metric"]).size().items():
        report_utils.log(f"{dataset} {metric}: {n} folds")
    report_utils.log(f"Folds dropped for missing values: {len(dropped)}")
    for r in dropped.itertuples():
        report_utils.log(f"Dropped {r.dataset} {r.metric} fold {r.fold}")

    report_utils.h2("H1: Does the Gap Change?")
    tests = pd.DataFrame([friedman_metric(per_fold, METRICS)])
    report_utils.table(tests, decimals=4)

    p = tests.p_value.iloc[0]
    pairs = None
    if p < ALPHA:
        report_utils.h2("Nemenyi Test: Which Metrics Differ?")
        pairs, mean_rank, critical = nemenyi(per_fold, METRICS, ALPHA)
        pairs = pairs.rename_axis("metric").reset_index()
        report_utils.table(pairs, decimals=4)
        ranks = ", ".join(f"{m} {r:.1f}" for m, r in mean_rank.items())
        report_utils.log(f"Nemenyi p values. Mean ranks: {ranks}.")
        report_utils.log(
            f"Two metrics differ when their mean ranks are further apart than the "
            f"critical difference, {critical:.4f}."
        )

    report_utils.h2("Gap per Cell")
    cells = cell_table(per_fold, DATASETS, METRICS)
    report_utils.table(cells, decimals=4)
    report_utils.log(
        "gap is positive when DL is better. mean_rank is 1 for the smallest "
        f"standardized gap in a fold and {len(METRICS)} for the largest."
    )

    if p < ALPHA:
        report_utils.log(
            f"H1 is supported: p = {p:.4f}, below {ALPHA}. The gap changes with the metric."
        )
    else:
        report_utils.log(f"H0 is not rejected: p = {p:.4f}, not below {ALPHA}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(tests, TESTS_FILE)
    save(cells, CELLS_FILE)
    if pairs is not None:
        save(pairs, NEMENYI_FILE)
    report_utils.log(f"Saved tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze()
