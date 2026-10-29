#!/usr/bin/env python3
"""
Turn the benchmark rows and the bootstrap arrays into the tables.

Datasets and arms are never pooled: every table is per dataset and arm.

POINT ESTIMATES  come from the benchmark CSVs: the mean over folds (and over seeds for
                 the deep models).
INTERVALS        come from the paired bootstrap of the test rows (bootstrap_utils).
                 A fold-level interval uses a t quantile, never 1.96.

The advantage of a deep model over a classical one is signed so that a positive value
always means the deep model is better, whichever way the metric points.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import bootstrap_utils

MODEL_ORDER = ["Fine-Gray", "cs-RSF", "DeepHit", "DSM"]
DEEP = ("DeepHit", "DSM")
CLASSICAL = ("Fine-Gray", "cs-RSF")

# DeepHit and DSM each against Fine-Gray and cs-RSF
PAIRS = [(d, c) for d in DEEP for c in CLASSICAL]

# A domain gets a label only when at least this many of the four pairs agree.
# Fixed before any result was seen.
MIN_PAIRS = 3

# A winner counts as stable when it wins at least this share of the replicates
STABLE_SHARE = 0.70

# Level of every interval
INTERVAL = 0.95

MEASURES = bootstrap_utils.MEASURES
HORIZONS = bootstrap_utils.BOOTSTRAP_HORIZONS
GLOBAL_METRICS = ("aj_calibration", "cr_d_calibration")
KEYS = ["dataset", "arm", "fold", "model"]


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
# Headline
# ----------------------------------------------------------------------------------------
def headline(folds):
    """Mean over folds with a t interval over the fold values."""
    g = folds.groupby(
        ["dataset", "arm", "model", "metric", "cause", "horizon"],
        as_index=False,
        dropna=False,
    ).value.agg(["mean", "std", "count"])
    g = g.reset_index() if "mean" in g.index.names else g
    quantile = stats.t.ppf(0.5 + INTERVAL / 2, (g["count"] - 1).clip(lower=1))
    half = quantile * g["std"] / np.sqrt(g["count"])
    g["ci_low"] = g["mean"] - half
    g["ci_high"] = g["mean"] + half
    g["model"] = pd.Categorical(g.model, MODEL_ORDER, ordered=True)
    return g.sort_values(["dataset", "arm", "metric", "cause", "horizon", "model"])


def headline_preview(folds):
    """One row per dataset, arm and model, averaged over causes and the horizons."""
    wide = folds[folds.horizon.isna() | folds.horizon.isin(HORIZONS)]
    per_fold = wide.groupby(KEYS + ["metric"], as_index=False).value.mean()
    t = per_fold.pivot_table(
        index=["dataset", "arm", "model"], columns="metric", values="value"
    ).reset_index()
    t["model"] = pd.Categorical(t.model, MODEL_ORDER, ordered=True)
    t = t.sort_values(["dataset", "arm", "model"])
    t["model"] = t.model.astype(str)
    order = list(MEASURES) + ["ibs"] + list(GLOBAL_METRICS)
    return t[["dataset", "arm", "model"] + [c for c in order if c in t]]


# ----------------------------------------------------------------------------------------
# Deep versus classical
# ----------------------------------------------------------------------------------------
def sign_vector():
    """+1 where larger is better and -1 where smaller is better, per measure."""
    return np.array(
        [1.0 if m in bootstrap_utils.HIGHER_IS_BETTER else -1.0 for m in MEASURES]
    )


def advantages(point, boot):
    """
    Deep minus classical for the four pairs, signed so positive means deep is better.

    point is (folds, models, measures, causes, horizons) and boot has a replicates
    axis after the folds. Both are averaged over the folds first, replicate by
    replicate. Returns (pairs, measures, causes, horizons) and the same with a
    leading replicates axis.
    """
    names = list(bootstrap_utils.MODELS)
    index = {m: names.index(m) for m in names}
    sign = sign_vector()[:, None, None]
    p = point.mean(axis=0)
    b = boot.mean(axis=0)
    adv_p = np.stack([sign * (p[index[d]] - p[index[c]]) for d, c in PAIRS])
    adv_b = np.stack([sign * (b[:, index[d]] - b[:, index[c]]) for d, c in PAIRS], 1)
    return adv_p, adv_b


def interval(draws):
    """Percentile interval over the first axis."""
    low = 100 * (1 - INTERVAL) / 2
    return np.percentile(draws, [low, 100 - low], axis=0)


def p_value(draws):
    """
    Two-sided bootstrap p-value over the first axis: twice the smaller share of
    replicates on either side of 0, with one added to each count so it is never 0.
    """
    n = draws.shape[0]
    below = ((draws <= 0).sum(axis=0) + 1) / (n + 1)
    above = ((draws >= 0).sum(axis=0) + 1) / (n + 1)
    return np.minimum(1.0, 2 * np.minimum(below, above))


def p_below(draws):
    """One-sided bootstrap p-value that the first-axis draws are below 0."""
    n = draws.shape[0]
    return np.minimum(1.0, ((draws >= 0).sum(axis=0) + 1) / (n + 1))


def label_advantage(low, high):
    if low > 0:
        return "deep better"
    if high < 0:
        return "classical better"
    return "no clear difference"


def cell_frame(point, draws, n_causes, extra):
    """Flatten (pairs, measures, causes, horizons) arrays into one row per cell."""
    low, high = interval(draws)
    p_values = p_value(draws)
    rows = []
    for p, (d, c) in enumerate(PAIRS):
        for m, measure in enumerate(MEASURES):
            for k in range(n_causes):
                for h, tau in enumerate(HORIZONS):
                    rows.append(
                        {
                            **extra,
                            "pair": f"{d} - {c}",
                            "measure": measure,
                            "cause": k + 1,
                            "horizon": tau,
                            "value": point[p, m, k, h],
                            "ci_low": low[p, m, k, h],
                            "ci_high": high[p, m, k, h],
                            "p_value": p_values[p, m, k, h],
                        }
                    )
    return pd.DataFrame(rows)


def deep_vs_classical(dataset, arm, point, boot):
    adv_p, adv_b = advantages(point, boot)
    frame = cell_frame(adv_p, adv_b, adv_p.shape[2], {"dataset": dataset, "arm": arm})
    frame["label"] = [
        label_advantage(lo, hi) for lo, hi in zip(frame.ci_low, frame.ci_high)
    ]
    return frame.rename(columns={"value": "advantage"}), adv_p, adv_b


def domain_labels(pairs):
    """
    One label per dataset, arm, measure, cause and horizon: deep better when at least
    MIN_PAIRS of the four pairs say so, classical better likewise, otherwise mixed.
    """
    keys = ["dataset", "arm", "measure", "cause", "horizon"]
    counts = (
        pairs.assign(
            deep=pairs.label == "deep better",
            classical=pairs.label == "classical better",
        )
        .groupby(keys, as_index=False)[["deep", "classical"]]
        .sum()
    )
    counts["label"] = np.select(
        [counts.deep >= MIN_PAIRS, counts.classical >= MIN_PAIRS],
        ["deep better", "classical better"],
        default="mixed",
    )
    return counts


def reversals(pairs):
    """
    H1: does the winner change with the metric, the period or the domain?

    Two cells are compared when they differ in one factor only. A reversal needs the
    deep advantage to have opposite signs, and its p-value is the larger of the two
    cell p-values (both must be significant), or 1 when the signs agree. Periods
    compare temporal with matched. Domains compare datasets at the same cause number.
    """
    same_cell = ["pair", "cause", "horizon"]
    factors = {
        "metric": (["dataset", "arm"], "measure"),
        "period": (["dataset", "measure"], "arm"),
        "domain": (["arm", "measure"], "dataset"),
    }
    out = []
    for factor, (same, vary) in factors.items():
        cells = pairs[same + same_cell + [vary, "advantage", "p_value"]]
        both = cells.merge(cells, on=same + same_cell, suffixes=("_a", "_b"))
        both = both[both[f"{vary}_a"] < both[f"{vary}_b"]]
        if factor == "period":
            both = both[(both.arm_a == "matched") & (both.arm_b == "temporal")]
        opposite = both.advantage_a * both.advantage_b < 0
        both = both.assign(
            factor=factor,
            p_value=np.where(
                opposite, both[["p_value_a", "p_value_b"]].max(axis=1), 1.0
            ),
        ).rename(columns={f"{vary}_a": "a", f"{vary}_b": "b"})
        out.append(both[["factor", "a", "b"] + same + same_cell + ["p_value"]])
    return pd.concat(out, ignore_index=True)


def reversal_counts(reversal_cells):
    """Per factor and pair of levels: cells compared and cells with p below 0.05."""
    grouped = reversal_cells.groupby(["factor", "a", "b"])
    return grouped.p_value.agg(
        cells="size", reversals=lambda p: int((p < 0.05).sum())
    ).reset_index()


def label_counts(labels, keys):
    """How many cells carry each label."""
    return labels.groupby(keys + ["label"]).size().unstack(fill_value=0).reset_index()


# ----------------------------------------------------------------------------------------
# H2: is the deep advantage smaller on later data?
# ----------------------------------------------------------------------------------------
def arm_difference(dataset, temporal, matched):
    """
    Advantage on the temporal arm minus advantage on the matched arm, per pair.

    The two arms were resampled independently, so replicate b of one is paired with
    replicate b of the other without any dependence between them. A negative value
    means the deep advantage is smaller on later data. p_smaller is the one-sided
    p-value of that alternative.
    """
    (adv_pt, adv_bt), (adv_pm, adv_bm) = temporal, matched
    point = adv_pt - adv_pm
    draws = adv_bt - adv_bm
    frame = cell_frame(point, draws, point.shape[2], {"dataset": dataset})
    frame["p_smaller"] = p_below(draws).ravel()
    frame["label"] = [
        (
            "advantage larger on later data"
            if lo > 0
            else "advantage smaller on later data" if hi < 0 else "no clear difference"
        )
        for lo, hi in zip(frame.ci_low, frame.ci_high)
    ]
    return frame.rename(columns={"value": "difference"})


# ----------------------------------------------------------------------------------------
# Sub question 2: does the best ranker also give the best probabilities?
# ----------------------------------------------------------------------------------------
def ranking_agreement(dataset, arm, point, boot):
    """
    For every fold, cause and horizon: the best model by concordance and the best by
    calibration gap, and how often each wins across the bootstrap replicates.
    """
    names = list(bootstrap_utils.MODELS)
    conc = MEASURES.index("concordance")
    gap = MEASURES.index("calibration_gap")
    rows = []
    for f in range(point.shape[0]):
        for k in range(point.shape[3]):
            for h, tau in enumerate(HORIZONS):
                c_point = point[f, :, conc, k, h]
                g_point = point[f, :, gap, k, h]
                if np.isnan(c_point).any() or np.isnan(g_point).any():
                    continue
                best_c, best_g = int(np.argmax(c_point)), int(np.argmin(g_point))
                c_draws = np.where(
                    np.isnan(boot[f, :, :, conc, k, h]),
                    -np.inf,
                    boot[f, :, :, conc, k, h],
                )
                g_draws = np.where(
                    np.isnan(boot[f, :, :, gap, k, h]),
                    np.inf,
                    boot[f, :, :, gap, k, h],
                )
                win_c = float((np.argmax(c_draws, axis=1) == best_c).mean())
                win_g = float((np.argmin(g_draws, axis=1) == best_g).mean())
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "fold": f + 1,
                        "cause": k + 1,
                        "horizon": tau,
                        "best_by_concordance": names[best_c],
                        "best_by_calibration": names[best_g],
                        "win_share_concordance": win_c,
                        "win_share_calibration": win_g,
                        "differ": best_c != best_g,
                        "stable": min(win_c, win_g) >= STABLE_SHARE,
                    }
                )
    return pd.DataFrame(rows)


def agreement_summary(cases):
    """Share of cases where the two winners differ, overall and per cause."""
    out = []
    for (dataset, arm), g in cases.groupby(["dataset", "arm"]):
        for scope, part in [("all causes", g)] + [
            (f"cause {k}", g[g.cause == k]) for k in sorted(g.cause.unique())
        ]:
            stable = part[part.stable]
            differ = stable[stable.differ]
            deep_ranker = differ.best_by_concordance.isin(DEEP).mean()
            out.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "scope": scope,
                    "cases": len(part),
                    "stable": len(stable),
                    "differ (all)": part.differ.mean(),
                    "differ (stable)": stable.differ.mean() if len(stable) else np.nan,
                    "best ranker deep": deep_ranker if len(differ) else np.nan,
                }
            )
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------
# Secondary
# ----------------------------------------------------------------------------------------
def friedman(folds):
    """
    Friedman test within one dataset and arm, folds as blocks, low power.

    Each block holds one value per model: the metric averaged over causes and, for the
    per-cause measures, over the bootstrap horizons.
    """
    ranked = list(MEASURES) + ["ibs"] + list(GLOBAL_METRICS)
    use = folds[folds.metric.isin(ranked)]
    use = use[use.horizon.isna() | use.horizon.isin(HORIZONS)]
    per_fold = use.groupby(KEYS + ["metric"], as_index=False).value.mean()
    out = []
    for (dataset, arm, metric), g in per_fold.groupby(["dataset", "arm", "metric"]):
        wide = g.pivot_table(index="fold", columns="model", values="value")
        models = [m for m in MODEL_ORDER if m in wide.columns]
        folds_total = len(wide)
        wide = wide[models].dropna()
        if len(models) < 3 or len(wide) < 3:
            continue
        stat, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in models])
        lower_is_better = metric not in bootstrap_utils.HIGHER_IS_BETTER
        ranks = wide.rank(axis=1, ascending=lower_is_better).mean().sort_values()
        out.append(
            {
                "dataset": dataset,
                "arm": arm,
                "metric": metric,
                "folds": folds_total,
                "blocks": len(wide),
                "chi2": f"{stat:.2f}",
                "p": f"{p:.3f}",
                "mean rank, best first": ", ".join(
                    f"{m} {v:.1f}" for m, v in ranks.items()
                ),
            }
        )
    return pd.DataFrame(out)


def cost(df):
    """Seconds to fit and to predict. Diagnostic only."""
    once = df.drop_duplicates(["dataset", "arm", "fold", "model", "seed"])
    return (
        once.groupby(["dataset", "model"], as_index=False)[
            ["fit_seconds", "predict_seconds"]
        ]
        .mean()
        .round(2)
    )
