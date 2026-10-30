#!/usr/bin/env python3
"""
Test H1 and H2 on the benchmark rows.

The gap is the DL group (DeepHit and DSM) minus the classical group (Fine-Gray and
cs-RSF) on one metric in one domain, signed so that a positive value means DL is better.
The models are never ranked. Because the metrics have different units, gaps are divided
by the standard deviation of that metric's fold values over all four models.

H1: does the gap change with the metric? A Friedman test over the metrics, the blocks
are the (domain, fold) pairs. The Nemenyi test then compares the metrics in pairs.

H2: is the gap smaller on later data than on a random split of the same size? Each
temporal fold is paired with its matched random fold. The standardized differences are
averaged over the metrics, so each fold pair gives one number, and one exact one sided
Wilcoxon signed-rank test runs on the fold pairs of all domains.

LIMITS
    The folds share training data, so they are not independent. All p values are
    approximate. Read them as supporting evidence.
    A significant H1 result says the gap differs between metrics, not where. The cell
    table shows where. It detects any change in the size of the gap, not only a change
    of winner.
    H2 pools the domains. A smaller advantage can also mean DL falls further behind.
    The claims cover only the cells that were tested.
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
    return (
        ok.groupby(KEYS + ["metric", "cause", "horizon"], dropna=False)
        .value.mean()
        .reset_index()
    )


# ----------------------------------------------------------------------------------------
# Gaps
# ----------------------------------------------------------------------------------------
def model_values(folds, arms, datasets, metrics, horizons):
    """
    One value per dataset, arm, fold, model and metric.

    `folds` is the output of seed_mean. Each model is averaged over the causes and the
    horizons (a metric with no horizon, such as cr_d_calibration, has one value per
    fit).
    """
    use = folds[
        folds.arm.isin(arms)
        & folds.dataset.isin(datasets)
        & folds.metric.isin(metrics)
        & (folds.horizon.isin(horizons) | folds.horizon.isna())
    ]
    return use.groupby(KEYS + ["metric"]).value.mean().reset_index()


def group_gaps(values):
    """
    The group values and the signed gap of every fold. Returns (per_fold, dropped):
    dropped lists the folds left out because a model had no value.
    """
    index = ["dataset", "arm", "metric", "fold"]
    wide = (
        values.set_index(index + ["model"])
        .value.unstack("model")
        .reindex(columns=[*DEEP, *CLASSICAL])
    )
    complete = wide.notna().all(axis=1)
    dropped = wide[~complete].reset_index()[index]
    wide = wide[complete]
    per_fold = wide.reset_index()[index].copy()
    per_fold["dl_value"] = wide[list(DEEP)].mean(axis=1).to_numpy()
    per_fold["classical_value"] = wide[list(CLASSICAL)].mean(axis=1).to_numpy()
    sign = per_fold.metric.map(lambda m: 1.0 if m in HIGHER_IS_BETTER else -1.0)
    per_fold["gap"] = sign * (per_fold.dl_value - per_fold.classical_value)
    return per_fold, dropped


def metric_scale(values):
    """The standard deviation of the fold values of each metric over all models."""
    return (
        values.groupby(["dataset", "metric"]).value.std().rename("scale").reset_index()
    )


# ----------------------------------------------------------------------------------------
# H1
# ----------------------------------------------------------------------------------------
def fold_gaps(
    folds, arm, datasets, metrics, horizons
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The group values, the signed gap and the standardized gap of every fold of every cell.

    Returns (per_fold, dropped): dropped lists the folds left out because a model had
    no value.
    """
    values = model_values(folds, [arm], datasets, metrics, horizons)
    per_fold, dropped = group_gaps(values)
    per_fold = per_fold.merge(metric_scale(values), on=["dataset", "metric"])
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


# ----------------------------------------------------------------------------------------
# H2
# ----------------------------------------------------------------------------------------
def arm_differences(
    folds, arms, datasets, metrics, horizons
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The gap on later data minus the gap on the matched random split, per fold pair.

    `arms` is (later, matched). Fold i of one is paired with fold i of the other, as
    they have the same train and test sizes. The difference is negative when the DL
    advantage is smaller on later data. It is standardized by the scale of its metric
    over both arms. Returns (pairs, dropped): dropped lists the folds left out because
    a model had no value or the fold had no partner.
    """
    later, matched = arms
    values = model_values(folds, arms, datasets, metrics, horizons)
    per_fold, dropped = group_gaps(values)
    wide = per_fold.set_index(["dataset", "metric", "fold", "arm"]).gap.unstack("arm")
    wide = wide.reindex(columns=[later, matched])
    complete = wide.notna().all(axis=1)
    unpaired = wide[~complete].reset_index()[["dataset", "metric", "fold"]]
    dropped = pd.concat([dropped, unpaired], ignore_index=True)
    pairs = wide[complete].reset_index()
    pairs.columns.name = None
    pairs = pairs.rename(columns={later: "gap_later", matched: "gap_matched"})
    pairs["difference"] = pairs.gap_later - pairs.gap_matched
    pairs = pairs.merge(metric_scale(values), on=["dataset", "metric"])
    pairs["z"] = pairs.difference / pairs.scale
    return pairs, dropped


def overall_d(pairs, metrics):
    """The standardized difference averaged over the metrics, one per fold pair."""
    wide = pairs.pivot_table(index=["dataset", "fold"], columns="metric", values="z")
    return wide.reindex(columns=list(metrics)).dropna().mean(axis=1)


def wilcoxon_overall(pairs, metrics):
    """
    Exact one sided Wilcoxon signed-rank test: is the standardized difference, averaged
    over the metrics, below 0? The fold pairs of all domains are pooled.
    """
    d = overall_d(pairs, metrics)
    p = stats.wilcoxon(d, alternative="less", method="exact").pvalue
    return {"p_value": float(p)}


def difference_table(pairs, datasets, metrics):
    """
    One row per metric and dataset: the mean standardized difference and the number of
    fold pairs where the DL advantage is smaller on later data.
    """
    g = pairs.groupby(["metric", "dataset"])
    out = pd.DataFrame(
        {
            "mean_d": g.z.mean(),
            "pairs": g.size(),
            "pairs_dl_lead_smaller": g.z.apply(lambda z: int((z < 0).sum())),
        }
    )
    order = pd.MultiIndex.from_product([metrics, datasets], names=["metric", "dataset"])
    return out.reindex(order).reset_index()
