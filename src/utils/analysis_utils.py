#!/usr/bin/env python3
"""
Test H1 on the benchmark rows: does the gap between DL and classical models change?

The gap is the DL group (DeepHit and DSM) minus the classical group (Fine-Gray and
cs-RSF) on one metric in one domain, signed so that a positive value means DL is better.
The models are never ranked. Because the metrics have different units, each gap is
divided by the standard deviation of that metric's fold values over all four models.

The test is a Friedman test over the metrics. The blocks are the (domain, fold) pairs.

LIMITS
    There are 5 folds per domain and the folds share training data, so they are not
    independent. All p values are approximate. Read them as supporting evidence.
    The test uses 10 blocks.
    A significant result says the gap differs between metrics. It does not say where.
    The cell table shows where.
    The Nemenyi test follows a significant Friedman test and compares the metrics in pairs.
    The test detects any change in the size of the gap, not only a change of winner.
    The domains are not compared with each other.
    The claim covers only the cells that were tested, not every metric and domain.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

DEEP = ("DeepHit", "DSM")
CLASSICAL = ("Fine-Gray", "cs-RSF")
KEYS = ["dataset", "arm", "fold", "model"]

# Metrics where a larger value is better; the others are better when smaller
HIGHER_IS_BETTER = {"scaled_brier", "concordance"}


# ----------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------
def load(paths):
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise SystemExit("No benchmark files found, run the benchmark pipelines first")
    return pd.concat(frames, ignore_index=True)


def missing_values(df):
    """Rows of the benchmark that hold no value, per dataset, arm, model and metric."""
    bad = df.value.isna() | (df.error.fillna("") != "")
    out = (
        df.assign(missing=bad)
        .groupby(["dataset", "arm", "model", "metric"], as_index=False)
        .agg(missing=("missing", "sum"), values=("missing", "size"))
    )
    return out[out.missing > 0]


def seed_mean(df):
    """One value per fold: the seeds of a deep model are averaged."""
    ok = df[df.value.notna()]
    return ok.groupby(
        KEYS + ["metric", "cause", "horizon"], as_index=False, dropna=False
    ).value.mean()


# ----------------------------------------------------------------------------------------
# H1
# ----------------------------------------------------------------------------------------
def fold_gaps(folds, arm, datasets, metrics, horizons):
    """
    The group values, the signed gap and the standardized gap of every fold of every cell.

    `folds` is the output of seed_mean. Each model is averaged over the causes and the
    horizons (a metric with no horizon, such as cr_d_calibration, has one value per
    fit). The scale of a cell is the standard deviation of the fold values of its
    metric over all models and folds. Returns (per_fold, dropped): dropped lists the
    folds left out because a model had no value.
    """
    use = folds[
        (folds.arm == arm)
        & folds.dataset.isin(datasets)
        & folds.metric.isin(metrics)
        & (folds.horizon.isin(horizons) | folds.horizon.isna())
    ]
    values = use.groupby(KEYS + ["metric"], as_index=False).value.mean()
    scale = (
        values.groupby(["dataset", "metric"]).value.std().rename("scale").reset_index()
    )
    wide = values.pivot_table(
        index=["dataset", "arm", "metric", "fold"],
        columns="model",
        values="value",
        dropna=False,
    ).reindex(columns=[*DEEP, *CLASSICAL])
    complete = wide.notna().all(axis=1)
    dropped = wide[~complete].reset_index()[["dataset", "arm", "metric", "fold"]]
    wide = wide[complete]
    per_fold = wide.reset_index()[["dataset", "arm", "metric", "fold"]].copy()
    per_fold["dl_value"] = wide[list(DEEP)].mean(axis=1).to_numpy()
    per_fold["classical_value"] = wide[list(CLASSICAL)].mean(axis=1).to_numpy()
    sign = per_fold.metric.map(lambda m: 1.0 if m in HIGHER_IS_BETTER else -1.0)
    per_fold["gap"] = sign * (per_fold.dl_value - per_fold.classical_value)
    per_fold = per_fold.merge(scale, on=["dataset", "metric"])
    per_fold["z"] = per_fold.gap / per_fold.scale
    return per_fold, dropped


def metric_blocks(per_fold, metrics):
    """The standardized gaps with one row per (dataset, fold) and one column per metric."""
    wide = per_fold.pivot_table(index=["dataset", "fold"], columns="metric", values="z")
    return wide.reindex(columns=list(metrics)).dropna()


def friedman_metric(per_fold, metrics):
    """Does the gap change with the metric? The Friedman chi-square p value."""
    wide = metric_blocks(per_fold, metrics)
    _, p = stats.friedmanchisquare(*[wide[m] for m in wide])
    return {"p_value": float(p)}


def metric_ranks(per_fold, metrics):
    """The rank of each metric inside each (dataset, fold) block, 1 for the smallest gap."""
    wide = metric_blocks(per_fold, metrics)
    return pd.DataFrame(
        np.apply_along_axis(stats.rankdata, 1, wide.to_numpy()),
        index=wide.index,
        columns=wide.columns,
    )


def nemenyi(per_fold, metrics, alpha):
    """
    Nemenyi post hoc test: which pairs of metrics differ in the size of the gap?

    Two metrics differ when their mean ranks over the blocks are further apart than the
    critical difference. The p values come from the studentized range distribution.
    Returns (p, mean_rank, critical): p is a metric by metric table, 1 on the diagonal.
    """
    ranks = metric_ranks(per_fold, metrics)
    n, k = ranks.shape
    se = np.sqrt(k * (k + 1) / (6 * n))
    critical = stats.studentized_range.isf(alpha, k, np.inf) / np.sqrt(2) * se
    mean = ranks.mean()
    p = pd.DataFrame(1.0, index=list(metrics), columns=list(metrics))
    for a, b in itertools.combinations(metrics, 2):
        z = abs(mean[a] - mean[b]) / se * np.sqrt(2)
        p.loc[a, b] = p.loc[b, a] = float(stats.studentized_range.sf(z, k, np.inf))
    return p, mean, float(critical)


def cell_table(per_fold, datasets, metrics):
    """
    One row per dataset and metric, averaged over folds, with the mean rank of its gap.

    The rank is taken inside each (dataset, fold) block over the metrics, 1 for the
    smallest standardized gap. It is descriptive, there is no p value.
    """
    ranks = metric_ranks(per_fold, metrics)
    mean_rank = ranks.groupby("dataset").mean().stack().rename("mean_rank")
    out = (
        per_fold.groupby(["dataset", "metric"])[
            ["dl_value", "classical_value", "gap", "z"]
        ]
        .mean()
        .rename(columns={"z": "gap_standardized"})
        .join(mean_rank)
    )
    order = pd.MultiIndex.from_product([datasets, metrics], names=out.index.names)
    return out.reindex(order).reset_index()
