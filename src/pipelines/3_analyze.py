#!/usr/bin/env python3
"""
venv/bin/python src/pipelines/3_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import analysis_utils as A
from utils import report_utils

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
BENCHMARK_FILES = (
    Path("data/financial/5_benchmark/calm.csv"),
    Path("data/financial/5_benchmark/crisis.csv"),
    Path("data/medical/5_benchmark/medical.csv"),
)
OUTPUT_DIR = Path("data/analysis")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_utils.h1("ANALYZE")
    raw = A.load(BENCHMARK_FILES)
    folds = A.per_fold(raw)
    report_utils.log(
        f"Read {len(raw):,} numbers covering "
        f"{folds.groupby(['dataset','arm','fold','model']).ngroups} fitted models"
    )

    report_utils.h2("Headline: every metric by dataset, arm and model")
    table = A.summary(folds)
    report_utils.table(table)
    table.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    report_utils.h2("Main question: do the four models differ?")
    tests = []
    for metric in ("concordance", "ibs", "aj_calibration", "cr_d_calibration"):
        r = A.friedman(folds, metric)
        if r is None:
            report_utils.log(f"{metric}: not enough complete blocks")
            continue
        tests.append(
            {
                "metric": metric,
                "blocks": r["blocks"],
                "chi2": f"{r['statistic']:.2f}",
                "p": f"{r['p_value']:.2e}",
                "mean rank, best first": ", ".join(
                    f"{m} {v:.2f}" for m, v in r["mean_ranks"].items()
                ),
            }
        )
    if tests:
        report_utils.table(pd.DataFrame(tests), right=("chi2", "p"))

    report_utils.h2("Main question: deep minus classical on the same folds")
    gaps = pd.concat(
        [
            A.family_gap(folds, m)
            for m in ("concordance", "ibs", "aj_calibration", "cr_d_calibration")
        ],
        ignore_index=True,
    )
    gaps["favours"] = ["deep" if v > 0 else "classical" for v in gaps["mean"]]
    report_utils.table(gaps)
    gaps.to_csv(OUTPUT_DIR / "family_gap.csv", index=False)

    report_utils.h2("SQ1: temporal minus random, per model and metric")
    eff = A.arm_effect(folds)
    if len(eff):
        report_utils.table(eff)
        eff.to_csv(OUTPUT_DIR / "arm_effect.csv", index=False)

    report_utils.h2(
        "SQ2: does the discrimination ranking agree with the calibration ranking?"
    )
    agree = A.rank_agreement(folds)
    if len(agree):
        report_utils.table(agree, decimals=3)
        agree.to_csv(OUTPUT_DIR / "rank_agreement.csv", index=False)

    report_utils.h2("Cost: seconds to fit and to predict")
    cost = (
        raw.groupby(["dataset", "model"])[["fit_seconds", "predict_seconds"]]
        .mean()
        .round(2)
        .reset_index()
    )
    report_utils.table(cost, decimals=2)
    cost.to_csv(OUTPUT_DIR / "cost.csv", index=False)

    report_utils.log(f"Saved tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
