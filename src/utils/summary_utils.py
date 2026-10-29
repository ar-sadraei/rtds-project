#!/usr/bin/env python3
"""
Summary tables for the three hypotheses, built from the analysis outputs.

H1: does the winner change with the metric, the period or the domain?
H2: is the deep advantage smaller on later data than on the matched random split?
H3: do the rankings by concordance and by probability accuracy agree?

Model-level p-values come from the saved bootstrap draws: the advantage of a deep
model is averaged over causes, horizons and folds, and the p-value is two-sided.
"""

import pandas as pd
from scipy import stats

from . import analysis_utils as A
from . import benchmark_utils, bootstrap_utils

MEASURES = ("concordance", "scaled_brier", "calibration_gap")
DATASET_ORDER = ("medical", "financial-calm", "financial-crisis")
SPLITS = {"kfold": "K-fold", "temporal": "Temporal", "matched": "Matched"}
ALPHA = 0.05


def p_text(p, floor):
    """The p-value as text, with the smallest possible value shown as a bound."""
    return f"<{floor:.3f}" if p <= floor else f"{p:.3f}"


def sort_rows(df, split_column=None):
    """Datasets in a fixed order, then splits in a fixed order."""
    df = df.assign(_d=df.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)}))
    keys = ["_d"]
    if split_column:
        order = {s: i for i, s in enumerate(SPLITS)}
        df = df.assign(_s=df[split_column].map(order))
        keys.append("_s")
    return df.sort_values(keys, kind="stable").drop(
        columns=["_d", "_s"], errors="ignore"
    )


# ----------------------------------------------------------------------------------------
# Model level, from the bootstrap draws
# ----------------------------------------------------------------------------------------
def model_level(label, folds_path, bootstrap_dir):
    """
    One row per split, model pair and measure, and the H2 rows (temporal minus matched).

    Both are averaged over causes and horizons. Returns (comparison, later).
    """
    counts = benchmark_utils.fold_counts(benchmark_utils.read_folds(folds_path))
    arrays = bootstrap_utils.load(label, counts, bootstrap_dir)
    comparison, advantage = [], {}
    for split, (point, boot) in arrays.items():
        adv_p, adv_b = A.advantages(point, boot)
        adv_p, adv_b = adv_p.mean(axis=(2, 3)), adv_b.mean(axis=(3, 4))
        advantage[split] = (adv_p, adv_b)
        low, high = A.interval(adv_b)
        p = A.p_value(adv_b)
        floor = 2 / (adv_b.shape[0] + 1)
        for i, (deep, classical) in enumerate(A.PAIRS):
            for j, measure in enumerate(A.MEASURES):
                if measure not in MEASURES:
                    continue
                better = (
                    "deep"
                    if low[i, j] > 0
                    else "classical" if high[i, j] < 0 else "none"
                )
                comparison.append(
                    {
                        "dataset": label,
                        "split": split,
                        "measure": measure,
                        "deep": deep,
                        "classical": classical,
                        "advantage": adv_p[i, j],
                        "ci_low": low[i, j],
                        "ci_high": high[i, j],
                        "p_value": p[i, j],
                        "p_floor": floor,
                        "better": better,
                    }
                )
    later = []
    if "temporal" in advantage and "matched" in advantage:
        diff_p = advantage["temporal"][0] - advantage["matched"][0]
        diff_b = advantage["temporal"][1] - advantage["matched"][1]
        low, high = A.interval(diff_b)
        p_smaller = A.p_below(diff_b)
        floor = 1 / (diff_b.shape[0] + 1)
        for i, (deep, classical) in enumerate(A.PAIRS):
            for j, measure in enumerate(A.MEASURES):
                if measure in MEASURES:
                    later.append(
                        {
                            "dataset": label,
                            "deep": deep,
                            "classical": classical,
                            "measure": measure,
                            "difference": diff_p[i, j],
                            "ci_low": low[i, j],
                            "ci_high": high[i, j],
                            "p_smaller": p_smaller[i, j],
                            "p_floor": floor,
                        }
                    )
    return pd.DataFrame(comparison), pd.DataFrame(later)


def temporal_table(comparison):
    """Advantage and p-value per model pair on the temporal split, measures side by side."""
    use = comparison[comparison.split == "temporal"].copy()
    use["text"] = [
        f"{r.advantage:+.4f} (p={p_text(r.p_value, r.p_floor)})"
        for r in use.itertuples()
    ]
    wide = use.pivot(
        index=["dataset", "deep", "classical"], columns="measure", values="text"
    )
    wide = wide[list(MEASURES)].reset_index()
    wide = sort_rows(wide)
    wide.columns = ["Dataset", "Deep", "Classical", *MEASURES]
    return wide


# ----------------------------------------------------------------------------------------
# H1
# ----------------------------------------------------------------------------------------
def winner_counts(comparison):
    """Pairs (of 4) where one side is significantly better, per dataset, split, measure."""
    rows = []
    for (dataset, split), g in comparison.groupby(["dataset", "split"]):
        row = {"dataset": dataset, "split": split, "Split": SPLITS[split]}
        for measure in MEASURES:
            m = g[g.measure == measure]
            row[measure] = (
                f"deep {(m.better == 'deep').sum()} : "
                f"classical {(m.better == 'classical').sum()}"
            )
        rows.append(row)
    out = sort_rows(pd.DataFrame(rows), "split")
    out = out.drop(columns="split")
    return out[["dataset", "Split", *MEASURES]].rename(columns={"dataset": "Dataset"})


def reversals(cells):
    """Cells compared and reversals (opposite signs, both p below ALPHA) per factor."""
    cells = cells[
        (cells.factor != "metric") | (cells.a.isin(MEASURES) & cells.b.isin(MEASURES))
    ]
    grouped = cells.groupby(["factor", "a", "b"]).p_value
    out = grouped.agg(
        cells="size", reversals=lambda p: int((p < ALPHA).sum())
    ).reset_index()
    out["share"] = (100 * out.reversals / out.cells).round(1).astype(str) + "%"
    out.columns = [
        "Factor",
        "Level A",
        "Level B",
        "Cells compared",
        "Reversals (p<0.05)",
        "Share",
    ]
    return out


# ----------------------------------------------------------------------------------------
# H2
# ----------------------------------------------------------------------------------------
def later_data_table(later):
    """Temporal minus matched advantage per model pair and measure."""
    out = later.assign(
        **{
            "Temporal minus matched": [f"{v:+.4f}" for v in later.difference],
            "95% interval": [
                f"{lo:+.4f} to {hi:+.4f}" for lo, hi in zip(later.ci_low, later.ci_high)
            ],
            "p (smaller on later data)": [
                p_text(p, f) for p, f in zip(later.p_smaller, later.p_floor)
            ],
        }
    )
    out = sort_rows(out)
    return out[
        [
            "dataset",
            "deep",
            "classical",
            "measure",
            "Temporal minus matched",
            "95% interval",
            "p (smaller on later data)",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "deep": "Deep",
            "classical": "Classical",
            "measure": "Measure",
        }
    )


def later_data_counts(cells):
    """Cells with a smaller or a larger advantage on later data, per dataset and measure."""
    cells = cells[cells.measure.isin(MEASURES)]
    rows = []
    for dataset in DATASET_ORDER:
        for measure in MEASURES:
            g = cells[(cells.dataset == dataset) & (cells.measure == measure)]
            if len(g):
                rows.append(
                    {
                        "Dataset": dataset,
                        "Measure": measure,
                        "Cells": len(g),
                        "Smaller (p<0.05)": int((g.p_smaller < ALPHA).sum()),
                        "Larger (p<0.05)": int(
                            ((g.p_value < ALPHA) & (g.difference > 0)).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# H3
# ----------------------------------------------------------------------------------------
def ranking_table(agreement):
    """Stable cases where the best ranker is not the best on probabilities."""
    use = agreement[agreement.scope == "all causes"]
    rows = []
    for r in use.to_dict("records"):
        stable = int(r["stable"])
        share = r["differ (stable)"]
        differ = int(round(share * stable)) if stable and share == share else 0
        p = (
            stats.binomtest(differ, stable, 0.5, alternative="greater").pvalue
            if stable
            else float("nan")
        )
        rows.append(
            {
                "dataset": r["dataset"],
                "arm": r["arm"],
                "Split": SPLITS[r["arm"]],
                "Cases": int(r["cases"]),
                "Stable cases": stable,
                "Best ranker not best on probabilities": f"{differ} of {stable}",
                "p (share above 50%)": (
                    ("<0.001" if p < 0.001 else f"{p:.3f}") if stable else "n/a"
                ),
            }
        )
    out = sort_rows(pd.DataFrame(rows), "arm")
    return out.drop(columns="arm").rename(columns={"dataset": "Dataset"})


# ----------------------------------------------------------------------------------------
# Friedman
# ----------------------------------------------------------------------------------------
def friedman_table(friedman):
    """Best model by mean rank over folds, with the Friedman p-value, per measure."""
    use = friedman[friedman.metric.isin(MEASURES)]
    rows = []
    for (dataset, arm), g in use.groupby(["dataset", "arm"]):
        row = {"dataset": dataset, "arm": arm, "Split": SPLITS[arm]}
        for r in g.to_dict("records"):
            best = r["mean rank, best first"].split(",")[0].rsplit(" ", 1)[0]
            p = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            row[r["metric"]] = f"{best} (p={p})"
        rows.append(row)
    out = sort_rows(pd.DataFrame(rows), "arm").drop(columns="arm")
    return out[["dataset", "Split", *MEASURES]].rename(columns={"dataset": "Dataset"})
