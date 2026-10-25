#!/usr/bin/env python3
"""
Fold loading, hyperparameter search and the benchmark loop.

The loop knows nothing about either domain. Everything that differs between the
financial and the medical data is passed in by the pipeline: where the folds are,
which columns are identifiers, how many causes there are.

ARMS AND FOLD COUNTS
    The arms are kfold, temporal and matched (see fold_utils). Their fold counts
    differ, so they are read from the folds JSON and never hard coded.

TUNING
    Settings are chosen inside every fold, using that fold's training rows only. The
    validation rows are the latest training group (temporal) or a stratified 20
    percent hold-out (kfold, matched, and temporal fold 1). The criterion is the mean
    integrated Brier score over the causes. The classical models try every setting
    of their grid. The deep models draw a random sample of at most the same budget.

RESUMABLE
    Both tune() and run() write after every cell and skip the cells already written,
    so a crashed or interrupted run continues instead of starting over.

HORIZONS
    Fixed in advance at 12, 24, 36, 48, 60 and 72 months, with 72 as the primary
    endpoint. Choosing the horizon after seeing the results is the failure Sonabend
    et al. call C-hacking, so the grid is settled here and not per analysis.
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import fold_utils, metric_utils, report_utils
from .model_utils import MODELS, SEARCH_SPACE, SEEDED

HORIZONS = np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0])
PRIMARY_HORIZON = 72.0

# Share of a fold's training rows held out to choose the settings
# How the arms are named in titles
ARM_LABELS = {"kfold": "K-Fold", "temporal": "Temporal", "matched": "Matched"}

VALIDATION_FRACTION = 0.2


def features(df, id_col, group_col):
    drop = {id_col, group_col, "time", "event"}
    return [c for c in df.columns if c not in drop]


def read_folds(folds_file):
    """The folds JSON: row positions per arm and fold."""
    return json.loads(Path(folds_file).read_text())


def fold_counts(spec):
    return {arm: len(spec[arm]) for arm in fold_utils.ARMS}


def load_train(folds_dir, arm, k, id_col, group_col):
    """The training rows of one fold: features, time, event, group."""
    tr = pd.read_csv(Path(folds_dir) / arm / f"fold_{k}_train.csv")
    cols = features(tr, id_col, group_col)
    return (
        tr[cols].to_numpy(float),
        tr.time.to_numpy(float),
        tr.event.to_numpy(int),
        tr[group_col].to_numpy(),
    )


def load_fold(folds_dir, arm, k, id_col, group_col):
    """Read one materialized fold pair and return arrays ready to fit."""
    folder = Path(folds_dir) / arm
    te = pd.read_csv(folder / f"fold_{k}_test.csv")
    X_tr, t_tr, e_tr, _ = load_train(folds_dir, arm, k, id_col, group_col)
    cols = features(te, id_col, group_col)
    return (
        X_tr,
        t_tr,
        e_tr,
        te[cols].to_numpy(float),
        te.time.to_numpy(float),
        te.event.to_numpy(int),
        cols,
    )


def sample_configs(space, budget, seed=0):
    """
    The settings to try. A dict is a grid of its lists and a list is the settings
    themselves. With no budget, all of them are returned. With a budget, at most that
    many: all of them when there are no more, otherwise a seeded random sample.
    """
    if isinstance(space, dict):
        keys = list(space)
        grid = [
            dict(zip(keys, c)) for c in itertools.product(*(space[k] for k in keys))
        ]
    else:
        grid = list(space)
    if budget is None or len(grid) <= budget:
        return grid
    rng = np.random.default_rng(seed)
    return [grid[i] for i in sorted(rng.choice(len(grid), budget, replace=False))]


def validation_mask(groups, events, arm, seed=0):
    """
    Validation rows inside one fold's training rows.

    Temporal folds validate on the latest training group, which keeps the order of
    time. With a single training group, and for the other arms, it is a stratified
    random hold-out stratified by event.
    """
    levels = sorted(pd.unique(groups))
    if arm == "temporal" and len(levels) > 1:
        return groups == levels[-1]
    _, val = train_test_split(
        np.arange(len(events)),
        test_size=VALIDATION_FRACTION,
        stratify=events,
        random_state=seed,
    )
    mask = np.zeros(len(events), dtype=bool)
    mask[val] = True
    return mask


# ----------------------------------------------------------------------------------------
# Tuning
# ----------------------------------------------------------------------------------------
def read_params(path):
    """The tuned settings: {arm: {fold: {model: settings}}}, or {} when absent."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    # json turns the layer lists into strings when they came from a tuple
    for arms in raw.values():
        for folds in arms.values():
            for params in folds.values():
                for key, val in list(params.items()):
                    if isinstance(val, str) and val.startswith("["):
                        params[key] = json.loads(val)
    return raw


def tune(folds_file, folds_dir, id_col, group_col, n_causes, budget, out_file):
    """Choose the settings of every model inside every fold of every arm."""
    spec = read_folds(folds_file)
    best = read_params(out_file)
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(len(m) for a in best.values() for m in a.values())
    if done:
        report_utils.log(f"{out} already holds {done} tuned cells, skipping those")
    chosen = []

    for arm, n_folds in fold_counts(spec).items():
        for k in range(1, n_folds + 1):
            X, t, e, groups = load_train(folds_dir, arm, k, id_col, group_col)
            val = validation_mask(groups, e, arm)
            report_utils.h2(f"{ARM_LABELS[arm]}: Fold {k}/{n_folds}")
            report_utils.log(
                f"Tune on {int((~val).sum()):,} rows, validate on {int(val.sum()):,}."
            )
            scorer = metric_utils.FoldScorer(t[~val], e[~val], HORIZONS, n_causes)
            for name, fit in MODELS.items():
                cell = best.setdefault(arm, {}).setdefault(str(k), {})
                if name in cell:
                    continue
                # Grid search for the classical models, random search for the deep ones
                configs = sample_configs(
                    SEARCH_SPACE[name], budget if name in SEEDED else None
                )
                scores = []
                for i, params in enumerate(configs, 1):
                    step = f"[{i}/{len(configs)}]"
                    tag = f"{name:<10} {step:>7}:"
                    started = time.perf_counter()
                    try:
                        cif, _, _ = fit(
                            params,
                            X[~val],
                            t[~val],
                            e[~val],
                            X[val],
                            HORIZONS,
                            n_causes,
                        )
                        score = scorer.mean_ibs(cif, t[val], e[val])
                        note = f"IBS={score:.5f}"
                    except Exception as err:
                        score, note = np.inf, f"failed ({type(err).__name__})"
                    scores.append((score, params))
                    report_utils.log(
                        f"{tag}  {note}  {report_utils.settings(params)}  "
                        f"({time.perf_counter() - started:.1f}s)"
                    )
                score, params = min(scores, key=lambda s: s[0])
                cell[name] = params
                out.write_text(json.dumps(best, indent=1, default=str))
                chosen.append(
                    {
                        "Arm": arm,
                        "Fold": k,
                        "Model": name,
                        "Best IBS": score,
                    }
                )

    if not chosen:
        return best
    report_utils.log(f"Saved {out}")
    frame = pd.DataFrame(chosen)
    report_utils.h2("Results")
    report_utils.table(
        frame.groupby(["Arm", "Model"], sort=False, as_index=False)
        .agg(Folds=("Fold", "size"), Mean_best_IBS=("Best IBS", "mean"))
        .rename(columns={"Mean_best_IBS": "Mean best IBS"})
    )
    return best


# ----------------------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------------------
def fit_seeds(name, seeds):
    """Deep models run once per seed, classical models once."""
    return tuple(seeds) if name in SEEDED else (0,)


def run(
    label, folds_file, folds_dir, id_col, group_col, n_causes, params, seeds, out_dir
):
    """
    Fit every model on every fold of every arm.

    Writes the metrics to {out_dir}/{label}.csv, the calibration bins to
    {out_dir}/calibration_bins.csv and the test predictions of every fit to
    {out_dir}/predictions/{label}/.
    """
    out_dir = Path(out_dir)
    pred_dir = out_dir / "predictions" / label
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / f"{label}.csv"
    bins_file = out_dir / "calibration_bins.csv"
    spec = read_folds(folds_file)

    done = set()
    if metrics_file.exists():
        old = pd.read_csv(metrics_file)
        done = set(
            map(
                tuple,
                old[["arm", "fold", "model", "seed"]].drop_duplicates().to_numpy(),
            )
        )
        report_utils.log(
            f"{metrics_file} already holds {len(done)} finished fits, skipping those"
        )

    stats = []
    for arm, n_folds in fold_counts(spec).items():
        for k in range(1, n_folds + 1):
            pending = [
                (name, seed)
                for name in MODELS
                for seed in fit_seeds(name, seeds)
                if (arm, k, name, seed) not in done
            ]
            if not pending:
                continue
            X_tr, t_tr, e_tr, X_te, t_te, e_te, cols = load_fold(
                folds_dir, arm, k, id_col, group_col
            )
            report_utils.h2(f"{ARM_LABELS[arm]}: Fold {k}/{n_folds}")
            report_utils.log(f"Train on {len(X_tr):,} rows, test on {len(X_te):,}.")
            test_index = np.asarray(spec[arm][k - 1]["test"])
            for name, fit in MODELS.items():
                settings = params.get(arm, {}).get(str(k), {}).get(name)
                if settings is None:
                    report_utils.log(f"No tuned settings for {arm} {k} {name}, stop")
                    raise SystemExit(f"Run the tuning first: {arm} fold {k} {name}")
                model_seeds = fit_seeds(name, seeds)
                for j, seed in enumerate(model_seeds, 1):
                    if (arm, k, name, seed) in done:
                        continue
                    where = f"{label} {arm} fold {k} {name} seed {seed}"
                    started = time.perf_counter()
                    bins = []
                    try:
                        cif, fit_s, pred_s = fit(
                            settings, X_tr, t_tr, e_tr, X_te, HORIZONS, n_causes, seed
                        )
                        np.savez_compressed(
                            pred_dir / f"{arm}_fold{k}_{name}_seed{seed}.npz",
                            cif=cif.astype(np.float32),
                            test_index=test_index,
                        )
                        rows, bins = metric_utils.score(
                            cif, t_tr, e_tr, t_te, e_te, HORIZONS, n_causes, where
                        )
                        failed = ""
                    except Exception as err:
                        failed = f"{type(err).__name__}: {err}"
                        fit_s = pred_s = np.nan
                        rows = [
                            {
                                "metric": "fit",
                                "cause": None,
                                "horizon": None,
                                "value": None,
                                "error": failed,
                            }
                        ]
                        report_utils.log(f"{name} seed {seed}: fit failed, {failed}")

                    context = {
                        "dataset": label,
                        "arm": arm,
                        "fold": k,
                        "model": name,
                        "seed": seed,
                    }
                    if bins:
                        pd.DataFrame([{**context, **b} for b in bins]).to_csv(
                            bins_file,
                            mode="a",
                            header=not bins_file.exists(),
                            index=False,
                        )
                    frame = pd.DataFrame(rows)
                    for i, (key, value) in enumerate(context.items()):
                        frame.insert(i, key, value)
                    frame["n_train"] = len(X_tr)
                    frame["n_test"] = len(X_te)
                    frame["n_features"] = len(cols)
                    # Diagnostic only, not part of any question
                    frame["fit_seconds"] = fit_s
                    frame["predict_seconds"] = pred_s
                    frame.to_csv(
                        metrics_file,
                        mode="a",
                        header=not metrics_file.exists(),
                        index=False,
                    )

                    took = time.perf_counter() - started
                    stats.append(
                        {"Model": name, "Failed": bool(failed), "Seconds": took}
                    )
                    note = failed if failed else f"{len(rows)} scores"
                    step = f"[{j}/{len(model_seeds)}]"
                    report_utils.log(
                        f"{name:<10} {step:>7}: seed={seed}  {note}  ({took:.1f}s)"
                    )
    report_utils.log(f"Wrote {metrics_file}")
    if stats:
        report_utils.h2("This run (diagnostic)")
        grouped = pd.DataFrame(stats).groupby("Model", sort=False)
        report_utils.table(
            grouped.agg(
                Fits=("Failed", "size"),
                Failed=("Failed", "sum"),
                Mean_seconds=("Seconds", "mean"),
            )
            .rename(columns={"Mean_seconds": "Mean seconds"})
            .reset_index(),
            decimals=1,
        )
