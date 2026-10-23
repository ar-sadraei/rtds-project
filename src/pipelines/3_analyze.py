#!/usr/bin/env python3
"""
    venv/bin/python src/pipelines/3_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import analysis_utils as A

# ----------------------------------------------------------------------------------------
# 1. Folders and settings
# ----------------------------------------------------------------------------------------
BENCHMARK_FILES = (
    Path("data/financial/5_benchmark/calm.csv"),
    Path("data/financial/5_benchmark/crisis.csv"),
    Path("data/medical/5_benchmark/medical.csv"),
)
OUTPUT_DIR = Path("data/analysis")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def banner(text):
    print("\n" + "=" * 92)
    print(text)
    print("=" * 92)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = A.load(BENCHMARK_FILES)
    folds = A.per_fold(raw)
    print(
        f"\nread {len(raw):,} numbers covering "
        f"{folds.groupby(['dataset','arm','fold','model']).ngroups} fitted models"
    )

    banner("HEADLINE: every metric by dataset, arm and model")
    table = A.summary(folds)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    table.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    banner("MAIN QUESTION: do the four models differ?")
    for metric in ("concordance", "ibs", "aj_calibration", "cr_d_calibration"):
        r = A.friedman(folds, metric)
        if r is None:
            print(f"  {metric}: not enough complete blocks")
            continue
        print(
            f"\n  {metric} (Friedman, {r['blocks']} blocks): "
            f"chi2 = {r['statistic']:.2f}, p = {r['p_value']:.2e}"
        )
        print(
            "    mean rank, best first: "
            + ", ".join(f"{m} {v:.2f}" for m, v in r["mean_ranks"].items())
        )

    banner("MAIN QUESTION: deep minus classical on the same folds")
    gaps = pd.concat(
        [
            A.family_gap(folds, m)
            for m in ("concordance", "ibs", "aj_calibration", "cr_d_calibration")
        ],
        ignore_index=True,
    )
    gaps["favours"] = ["deep" if v > 0 else "classical" for v in gaps["mean"]]
    print(gaps.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    gaps.to_csv(OUTPUT_DIR / "family_gap.csv", index=False)

    banner("SQ1: temporal minus random, per model and metric")
    eff = A.arm_effect(folds)
    if len(eff):
        print(eff.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        eff.to_csv(OUTPUT_DIR / "arm_effect.csv", index=False)

    banner("SQ2: does the discrimination ranking agree with the calibration ranking?")
    agree = A.rank_agreement(folds)
    if len(agree):
        print(agree.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        agree.to_csv(OUTPUT_DIR / "rank_agreement.csv", index=False)

    banner("COST: seconds to fit and to predict")
    cost = (
        raw.groupby(["dataset", "model"])[["fit_seconds", "predict_seconds"]]
        .mean()
        .round(2)
        .reset_index()
    )
    print(cost.to_string(index=False))
    cost.to_csv(OUTPUT_DIR / "cost.csv", index=False)

    print(f"\nsaved tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
