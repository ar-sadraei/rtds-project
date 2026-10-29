#!/usr/bin/env python3
"""
Analyze the benchmark: bootstrap, then the tables.

Every table is per dataset and arm, nothing is pooled. The bootstrap of the test rows
is saved per fold and resumable, and it is the slow part. Everything printed here is
also written to data/analysis/, in one folder per context.

    venv/bin/python src/pipelines/3_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import analysis_utils as A
from utils import benchmark_utils, bootstrap_utils, report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
DATASETS = (
    {
        "label": "financial-calm",
        "folds": Path("data/financial/2_folds/financial-calm.json"),
        "normalized": Path("data/financial/3_normalized/calm"),
        "benchmark": Path("data/financial/5_benchmark"),
        "id": "loan_id",
        "group": "vintage",
        "causes": 2,
    },
    {
        "label": "financial-crisis",
        "folds": Path("data/financial/2_folds/financial-crisis.json"),
        "normalized": Path("data/financial/3_normalized/crisis"),
        "benchmark": Path("data/financial/5_benchmark"),
        "id": "loan_id",
        "group": "vintage",
        "causes": 2,
    },
    {
        "label": "medical",
        "folds": Path("data/medical/2_folds/medical.json"),
        "normalized": Path("data/medical/3_normalized"),
        "benchmark": Path("data/medical/5_benchmark"),
        "id": "participant_id",
        "group": "cycle",
        "causes": 3,
    },
)
OUTPUT_DIR = Path("data/analysis")

# One folder per context, numbered in run order
BOOTSTRAP_DIR = OUTPUT_DIR / "0_bootstrap"
OVERVIEW_DIR = OUTPUT_DIR / "1_overview"
MAIN_DIR = OUTPUT_DIR / "2_main_question"
SQ1_DIR = OUTPUT_DIR / "3_sq1_later_data"
SQ2_DIR = OUTPUT_DIR / "4_sq2_ranking_vs_probability"
SUPPORT_DIR = OUTPUT_DIR / "5_supporting"

# Same seeds as the benchmark pipelines
SEEDS = (0, 1, 2)

# Resamples of the test rows in every fold
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 0


def show(title, frame, folder, name, decimals=3, **kwargs):
    """Print a table under a sub title and write it to a folder."""
    report_utils.h2(title)
    if len(frame):
        report_utils.table(frame, decimals=decimals, **kwargs)
    else:
        report_utils.log("Nothing to show")
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{name}.csv", index=False)


@report_utils.logged
def analyze() -> None:
    """Run every step of this process."""
    report_utils.h1("ANALYZE")
    raw = A.load(d["benchmark"] / f"{d['label']}.csv" for d in DATASETS)

    missing = A.missing_values(raw)
    if len(missing):
        show("Missing metric values", missing, OVERVIEW_DIR, "missing_values")
    else:
        report_utils.h2("Missing Metric Values")
        report_utils.log("None: every metric of every fit has a value")

    report_utils.h2("Bootstrap of the test rows")
    for d in DATASETS:
        bootstrap_utils.run(
            d["label"],
            d["folds"],
            d["normalized"],
            d["benchmark"] / "predictions" / d["label"],
            d["id"],
            d["group"],
            d["causes"],
            SEEDS,
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_DIR,
            BOOTSTRAP_SEED,
        )

    folds = A.seed_mean(raw)
    full = A.headline(folds)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OVERVIEW_DIR / "headline.csv", index=False)
    show(
        "Headline: mean over folds and seeds",
        A.headline_preview(folds),
        OVERVIEW_DIR,
        "headline_preview",
        decimals=4,
    )
    report_utils.log(
        f"Averaged over causes and the horizons 12, 36 and 60. The full table, per cause "
        f"and horizon with a t interval over folds, is {OVERVIEW_DIR}/headline.csv."
    )

    pairs, differences, agreement = [], [], []
    for d in DATASETS:
        counts = benchmark_utils.fold_counts(benchmark_utils.read_folds(d["folds"]))
        arrays = bootstrap_utils.load(d["label"], counts, BOOTSTRAP_DIR)
        advantage = {}
        for arm, (point, boot) in arrays.items():
            frame, adv_p, adv_b = A.deep_vs_classical(d["label"], arm, point, boot)
            pairs.append(frame)
            advantage[arm] = (adv_p, adv_b)
            agreement.append(A.ranking_agreement(d["label"], arm, point, boot))
        if "temporal" in advantage and "matched" in advantage:
            differences.append(
                A.arm_difference(
                    d["label"], advantage["temporal"], advantage["matched"]
                )
            )

    if pairs:
        pairs = pd.concat(pairs, ignore_index=True)
        MAIN_DIR.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(MAIN_DIR / "deep_vs_classical.csv", index=False)
        labels = A.domain_labels(pairs)
        labels.to_csv(MAIN_DIR / "domain_labels.csv", index=False)
        reversal_cells = A.reversals(pairs)
        reversal_cells.to_csv(MAIN_DIR / "h1_reversal_cells.csv", index=False)
        show(
            "H1: does the winner change with metric, period or domain?",
            A.reversal_counts(reversal_cells),
            MAIN_DIR,
            "h1_reversals",
        )
        report_utils.log(
            f"A reversal is a pair of cells where the deep advantage has opposite signs "
            f"and both p-values are below 0.05. Cells are in "
            f"{MAIN_DIR}/h1_reversal_cells.csv."
        )
        show(
            "Main question and SQ1: deep versus classical",
            A.label_counts(labels, ["dataset", "arm", "measure"]),
            MAIN_DIR,
            "domain_label_counts",
        )
        report_utils.log(
            f"Cells are cause x horizon. A cell is labelled when at least "
            f"{A.MIN_PAIRS} of the 4 pairs agree. Pairs, intervals and labels per "
            f"cell are in {MAIN_DIR}/deep_vs_classical.csv."
        )
    if differences:
        differences = pd.concat(differences, ignore_index=True)
        SQ1_DIR.mkdir(parents=True, exist_ok=True)
        differences.to_csv(SQ1_DIR / "h2_arm_difference.csv", index=False)
        show(
            "H2: deep advantage on later data minus on the matched random split",
            differences.groupby(["dataset", "measure", "label"])
            .size()
            .unstack(fill_value=0)
            .reset_index(),
            SQ1_DIR,
            "h2_summary",
        )
        report_utils.log(
            f"Cells are pair x cause x horizon. Differences and intervals are in "
            f"{SQ1_DIR}/h2_arm_difference.csv."
        )
    if agreement:
        cases = pd.concat(agreement, ignore_index=True)
        SQ2_DIR.mkdir(parents=True, exist_ok=True)
        cases.to_csv(SQ2_DIR / "ranking_cases.csv", index=False)
        show(
            "SQ2: does the best ranker give the best probabilities?",
            A.agreement_summary(cases),
            SQ2_DIR,
            "ranking_agreement",
        )
        report_utils.log(
            f"A case is a fold x cause x horizon. It is stable when both winners win "
            f"at least {A.STABLE_SHARE:.0%} of the bootstrap replicates."
        )

    show(
        "Friedman test across the four models (secondary, low power)",
        A.friedman(folds),
        SUPPORT_DIR,
        "friedman",
    )
    report_utils.log(
        "Folds are the blocks, so power is low. Read it as supporting evidence only."
    )

    show(
        "Cost: seconds to fit and to predict (diagnostic)",
        A.cost(raw),
        SUPPORT_DIR,
        "cost",
        2,
    )
    report_utils.log(f"Saved tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze()
