#!/usr/bin/env python3
"""
Turn the benchmark rows into the tables.

"""

import numpy as np
import pandas as pd
from scipy import stats

from . import report_utils

PRIMARY_HORIZON = 72.0
LOWER_IS_BETTER = {"ibs", "aj_calibration", "cr_d_calibration", "incoherent_cells"}
# Reported but never ranked on: diagnostics, not performance
NOT_PERFORMANCE = {"aj_calibration_spread", "incoherent_cells"}
MODEL_ORDER = ["Fine-Gray", "cs-RSF", "DeepHit", "DSM"]
CLASSICAL = {"Fine-Gray", "cs-RSF"}


def load(paths):
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise SystemExit("no benchmark files found, run the benchmark pipelines first")
    df = pd.concat(frames, ignore_index=True)
    failed = df[df.error.notna() & (df.error != "")]
    if len(failed):
        report_utils.log("Fits that failed:")
        for _, r in (
            failed[["dataset", "arm", "fold", "model", "error"]]
            .drop_duplicates()
            .iterrows()
        ):
            report_utils.log(
                f"  {r.dataset} {r.arm} fold {r.fold} {r.model}: {r.error}"
            )
    return df[df.value.notna()]


def per_fold(df):
    """One value per dataset, arm, fold, model, metric: averaged over causes."""
    d = df.copy()
    # concordance is reported per horizon; keep the primary endpoint
    keep = (d.metric != "concordance") | (d.horizon == PRIMARY_HORIZON)
    d = d[keep]
    return d.groupby(
        ["dataset", "arm", "fold", "model", "metric"], as_index=False
    ).value.mean()


def summary(folds):
    """Mean over folds, the headline table."""
    t = folds.pivot_table(
        index=["dataset", "arm", "model"], columns="metric", values="value"
    ).reset_index()
    t["model"] = pd.Categorical(t.model, MODEL_ORDER, ordered=True)
    return t.sort_values(["dataset", "arm", "model"])


def best_by(folds, metric):
    """Rank models within each dataset and arm for one metric."""
    sub = folds[folds.metric == metric]
    out = []
    for (ds, arm), g in sub.groupby(["dataset", "arm"]):
        m = g.groupby("model").value.mean()
        asc = metric in LOWER_IS_BETTER
        order = m.sort_values(ascending=asc)
        out.append(
            {
                "dataset": ds,
                "arm": arm,
                "metric": metric,
                "best": order.index[0],
                "worst": order.index[-1],
                **{f"rank_{k}": i + 1 for i, k in enumerate(order.index)},
            }
        )
    return pd.DataFrame(out)


def arm_effect(folds):
    """SQ1:"""
    p = folds.pivot_table(
        index=["dataset", "fold", "model", "metric"], columns="arm", values="value"
    ).reset_index()
    if "temporal" not in p or "random" not in p:
        return pd.DataFrame()
    p["difference"] = p.temporal - p.random
    out = (
        p.groupby(["dataset", "model", "metric"])
        .difference.agg(["mean", "std", "count"])
        .reset_index()
    )
    # paired across folds, so a paired t interval on the fold differences
    out["ci_low"] = out["mean"] - 1.96 * out["std"] / np.sqrt(out["count"])
    out["ci_high"] = out["mean"] + 1.96 * out["std"] / np.sqrt(out["count"])
    out["crosses_zero"] = (out.ci_low < 0) & (out.ci_high > 0)
    return out


def rank_agreement(folds):
    """SQ2:"""
    out = []
    for (ds, arm), g in folds.groupby(["dataset", "arm"]):
        means = g.pivot_table(index="model", columns="metric", values="value")
        if "concordance" not in means:
            continue
        disc = means["concordance"].rank(ascending=False)
        for metric in ("ibs", "aj_calibration", "cr_d_calibration"):
            if metric not in means or means[metric].isna().all():
                continue
            other = means[metric].rank(ascending=True)
            pair = pd.concat([disc, other], axis=1).dropna()
            if len(pair) < 3:
                continue
            tau, p = stats.kendalltau(pair.iloc[:, 0], pair.iloc[:, 1])
            out.append(
                {
                    "dataset": ds,
                    "arm": arm,
                    "against": metric,
                    "kendall_tau": tau,
                    "p_value": p,
                    "best_by_concordance": disc.idxmin(),
                    "best_by_metric": other.idxmin(),
                }
            )
    return pd.DataFrame(out)


def friedman(folds, metric):
    """Main question:"""
    sub = folds[folds.metric == metric]
    wide = sub.pivot_table(
        index=["dataset", "arm", "fold"], columns="model", values="value"
    ).dropna()
    models = [m for m in MODEL_ORDER if m in wide.columns]
    if len(models) < 3 or len(wide) < 3:
        return None
    stat, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in models])
    asc = metric in LOWER_IS_BETTER
    ranks = wide[models].rank(axis=1, ascending=asc).mean().sort_values()
    return {
        "metric": metric,
        "blocks": len(wide),
        "statistic": stat,
        "p_value": p,
        "mean_ranks": ranks,
    }


def family_gap(folds, metric):
    """Deep minus classical, on the same folds, which is what the title asks."""
    sub = folds[folds.metric == metric].copy()
    sub["family"] = np.where(sub.model.isin(CLASSICAL), "classical", "deep")
    p = (
        sub.pivot_table(
            index=["dataset", "arm", "fold"], columns="family", values="value"
        )
        .dropna()
        .reset_index()
    )
    if "deep" not in p or "classical" not in p:
        return pd.DataFrame()
    sign = -1 if metric in LOWER_IS_BETTER else 1
    p["deep_advantage"] = sign * (p.deep - p.classical)
    return (
        p.groupby(["dataset", "arm"])
        .deep_advantage.agg(["mean", "std", "count"])
        .reset_index()
        .assign(metric=metric)
    )
