#!/usr/bin/env python3
"""
Analyze the benchmark: the tests for H1 and H2.

The relative performance is the gap, DL group minus classical group, positive when DL
is better.

H1: does the gap change with the evaluation metric? A Friedman test over the metrics
on K-fold (10 blocks, the domain and fold pairs). When its p value is below ALPHA, the
Nemenyi test compares the metrics in pairs.

H2: is the DL advantage smaller on later data than on a standard split? Each temporal
fold is paired with its matched random fold (same train and test sizes). The difference
is averaged over the metrics, and one exact one sided Wilcoxon signed-rank test pools
the fold pairs of the three domains. K-fold is not used.

The folds share training data, so all p values are approximate. The claims cover only
these cells.

    venv/bin/python src/pipelines/3_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import report_utils
from utils.analysis_utils import (
    agreement_table,
    arm_differences,
    cell_table,
    difference_table,
    fold_agreement,
    fold_gaps,
    friedman_metric,
    load,
    missing_values,
    model_values,
    nemenyi,
    seed_mean,
    sign_test_tau,
    wilcoxon_overall,
)

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
BENCHMARK_FILES = {
    "medical": Path("data/medical/5_benchmark/medical.csv"),
    "financial-calm": Path("data/financial/5_benchmark/financial-calm.csv"),
    "financial-crisis": Path("data/financial/5_benchmark/financial-crisis.csv"),
}
OUTPUT_DIR = Path("data/analysis")
H1_TESTS_FILE = OUTPUT_DIR / "h1_friedman.csv"
H1_CELLS_FILE = OUTPUT_DIR / "h1_cells.csv"
H1_NEMENYI_FILE = OUTPUT_DIR / "h1_nemenyi.csv"
H2_TESTS_FILE = OUTPUT_DIR / "h2_wilcoxon.csv"
H2_CELLS_FILE = OUTPUT_DIR / "h2_cells.csv"
H3_TESTS_FILE = OUTPUT_DIR / "h3_sign_test.csv"
H3_FOLDS_FILE = OUTPUT_DIR / "h3_folds.csv"
H3_CELLS_FILE = OUTPUT_DIR / "h3_cells.csv"

METRICS = ("scaled_brier", "concordance", "calibration_gap", "cr_d_calibration")
HORIZONS = (12.0, 36.0, 60.0)
ALPHA = 0.05

H1_ARM = "kfold"
H1_DATASETS = ("medical", "financial-calm")

# Later data first, then the random split it is compared with
H2_ARMS = ("temporal", "matched")
H2_DATASETS = ("medical", "financial-calm", "financial-crisis")

# The matched arm is the control for the temporal arm in H2, so it is not used here
H3_ARMS = ("kfold", "temporal")
H3_DATASETS = ("medical", "financial-calm", "financial-crisis")
H3_RANK_METRIC = "concordance"
H3_PROBABILITY_METRIC = "cr_d_calibration"


def save(frame: pd.DataFrame, path: Path) -> None:
    """Write a table as CSV with no newline after the last row."""
    path.write_text(frame.to_csv(index=False).rstrip("\n"))


def log_counts(frame: pd.DataFrame, unit: str) -> None:
    counts = frame.groupby(["dataset", "metric"]).size().reset_index(name="n")
    for dataset, metric, n in counts.itertuples(index=False):
        report_utils.log(f"{dataset} {metric}: {n} {unit}")


def log_dropped(dropped: pd.DataFrame) -> None:
    report_utils.log(f"Folds dropped for missing values: {len(dropped)}")
    for r in dropped.itertuples():
        report_utils.log(f"Dropped {r.dataset} {r.metric} fold {r.fold}")


def test_h1(folds: pd.DataFrame) -> None:
    """Friedman over the metrics, then Nemenyi, then the gap per cell."""
    report_utils.h2("Does the Gap Change With the Metric?")
    report_utils.log(f"Split: {H1_ARM}")
    report_utils.log(f"Datasets: {', '.join(H1_DATASETS)}")
    per_fold, dropped = fold_gaps(folds, H1_ARM, H1_DATASETS, METRICS, HORIZONS)
    log_counts(per_fold, "folds")
    log_dropped(dropped)

    tests = pd.DataFrame([friedman_metric(per_fold, METRICS)])
    report_utils.table(tests, decimals=4)
    save(tests, H1_TESTS_FILE)

    p = tests.p_value.iloc[0]
    if p < ALPHA:
        report_utils.h2("Nemenyi Test, Which Metrics Differ?")
        pairs, mean_rank, critical = nemenyi(per_fold, METRICS, ALPHA)
        pairs = pairs.rename_axis("metric").reset_index()
        report_utils.table(pairs, decimals=4)
        save(pairs, H1_NEMENYI_FILE)
        ranks = ", ".join(f"{m} {r:.1f}" for m, r in mean_rank.items())
        report_utils.log(f"Nemenyi p values. Mean ranks: {ranks}.")
        report_utils.log(
            f"Two metrics differ when their mean ranks are further apart than the "
            f"critical difference, {critical:.4f}."
        )

    report_utils.h2("Gap per Cell")
    cells = cell_table(per_fold, H1_DATASETS, METRICS)
    report_utils.table(cells, decimals=4)
    save(cells, H1_CELLS_FILE)
    report_utils.log(
        "gap is positive when DL is better. mean_rank is 1 for the smallest "
        f"standardized gap in a fold and {len(METRICS)} for the largest."
    )
    if p < ALPHA:
        report_utils.log(
            f"H1 is supported: p = {p:.4f}, below {ALPHA}. The gap changes with the "
            "metric."
        )
    else:
        report_utils.log(f"H1: H0 is not rejected, p = {p:.4f}, not below {ALPHA}.")


def test_h2(folds: pd.DataFrame) -> None:
    """One sided Wilcoxon on the later minus matched differences, averaged over metrics."""
    report_utils.h2("Is the DL Advantage Smaller on Later Data?")
    report_utils.log(f"Splits: {H2_ARMS[0]} against {H2_ARMS[1]}, fold i with fold i")
    report_utils.log(f"Datasets: {', '.join(H2_DATASETS)}")
    pairs, dropped = arm_differences(folds, H2_ARMS, H2_DATASETS, METRICS, HORIZONS)
    log_counts(pairs, "fold pairs")
    log_dropped(dropped)

    tests = pd.DataFrame([wilcoxon_overall(pairs, METRICS)])
    report_utils.table(tests, decimals=4)
    save(tests, H2_TESTS_FILE)
    report_utils.log(
        "d is the gap on later data minus the gap on the matched split, standardized "
        "and averaged over the metrics. The test asks whether it is below 0."
    )

    report_utils.h2("Difference per Domain and Metric")
    cells = difference_table(pairs, H2_DATASETS, METRICS)
    report_utils.table(cells, decimals=4)
    save(cells, H2_CELLS_FILE)
    report_utils.log(
        "mean_d below 0 means the DL advantage is smaller on later data. "
        "pairs_dl_lead_smaller counts the fold pairs where it is. No test between "
        "domains."
    )

    p = tests.p_value.iloc[0]
    if p < ALPHA:
        report_utils.log(
            f"H2 is supported: p = {p:.4f}, below {ALPHA}. The DL advantage is smaller "
            "on later data."
        )
    else:
        report_utils.log(f"H2: H0 is not rejected, p = {p:.4f}, not below {ALPHA}.")


def test_h3(folds: pd.DataFrame) -> None:
    """Kendall's tau per fold between the two orderings, then a sign test."""
    report_utils.h2("Do the Rankings Agree?")
    report_utils.log(f"Splits: {', '.join(H3_ARMS)}")
    report_utils.log(f"Datasets: {', '.join(H3_DATASETS)}")
    report_utils.log(f"{H3_RANK_METRIC} against {H3_PROBABILITY_METRIC}")

    values = model_values(
        folds, H3_ARMS, H3_DATASETS, (H3_RANK_METRIC, H3_PROBABILITY_METRIC), HORIZONS
    )
    per_fold = fold_agreement(values, H3_RANK_METRIC, H3_PROBABILITY_METRIC)
    save(per_fold, H3_FOLDS_FILE)

    report_utils.h2("Agreement per Cell")
    cells = agreement_table(per_fold)
    report_utils.table(cells, decimals=4)
    save(cells, H3_CELLS_FILE)

    tests = pd.DataFrame([sign_test_tau(per_fold, ALPHA)])
    report_utils.table(tests, decimals=4)
    save(tests, H3_TESTS_FILE)

    row = tests.iloc[0]
    report_utils.log(
        "tau is negative when the model that ranks best is not the model with the "
        "best probabilities. Tied folds are left out of the sign test."
    )
    if row.rejected:
        report_utils.log(
            f"H3 is supported: p = {row.p_value:.4f}, below {ALPHA}. The two "
            "orderings disagree."
        )
    else:
        report_utils.log(
            f"H3: H0 is not rejected, p = {row.p_value:.4f}. The smallest p value "
            f"possible with {int(row.untied)} untied folds is {row.smallest_p:.4f}."
        )


@report_utils.logged
def analyze() -> None:
    """Run every step of this process."""
    report_utils.h1("ANALYZE: H1")
    report_utils.log(f"Metrics: {', '.join(METRICS)}")
    report_utils.log(f"Horizons: {', '.join(f'{h:g}' for h in HORIZONS)}")
    raw = load(BENCHMARK_FILES.values())
    missing = missing_values(raw)
    report_utils.log(f"Benchmark rows without a value: {int(missing.missing.sum())}")
    folds = seed_mean(raw)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_h1(folds)
    report_utils.h1("ANALYZE: H2")
    test_h2(folds)
    report_utils.h1("ANALYZE: H3")
    test_h3(folds)
    report_utils.log(f"Saved tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze()
