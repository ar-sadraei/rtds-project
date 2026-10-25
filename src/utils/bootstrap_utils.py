#!/usr/bin/env python3
"""
Paired bootstrap of the test rows, done once per fold and saved.

Inside every fold the test rows are resampled with replacement. The same resampled
indices are used for all models, the per-cause measures are recomputed for all of
them, and the arrays are saved. The analysis then builds every difference and every
interval from these arrays without refitting anything.

Only the horizons in BOOTSTRAP_HORIZONS (1, 3 and 5 years) are bootstrapped. The
deep models are averaged over their seeds before scoring.

RESUMABLE
    One file per (dataset, arm, fold). A finished fold is skipped on the next run.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmark_utils, fold_utils, metric_utils, report_utils
from .model_utils import MODELS

BOOTSTRAP_HORIZONS = (12.0, 36.0, 60.0)

# The measures that are bootstrapped, in the order of the saved arrays
MEASURES = metric_utils.METRICS

# Metrics where a larger value is better; the others are better when smaller
HIGHER_IS_BETTER = {"scaled_brier", "concordance"}


def horizon_columns(horizons=BOOTSTRAP_HORIZONS):
    """Positions of the bootstrap horizons inside the benchmark horizon grid."""
    grid = list(benchmark_utils.HORIZONS)
    return [grid.index(h) for h in horizons]


def read_predictions(pred_dir, arm, k, seeds):
    """
    The test predictions of one fold, one array (n, causes, horizons) per model.

    Deep models are averaged over their seeds. Returns None for a model whose
    predictions are missing, so the caller can say which one.
    """
    out = {}
    for name in MODELS:
        files = sorted(Path(pred_dir).glob(f"{arm}_fold{k}_{name}_seed*.npz"))
        wanted = len(seeds) if name in benchmark_utils.SEEDED else 1
        if len(files) < wanted:
            out[name] = None
            continue
        out[name] = np.mean(
            [np.load(f)["cif"].astype(float) for f in files[:wanted]], axis=0
        )
    return out


def bootstrap_fold(preds, t_tr, e_tr, t_te, e_te, n_causes, replicates, seed):
    """
    Point estimate and bootstrap draws of every measure for every model.

    Returns (point, boot) with shapes (models, measures, causes, horizons) and
    (replicates, models, measures, causes, horizons).
    """
    cols = horizon_columns()
    times = np.asarray(BOOTSTRAP_HORIZONS)
    scorer = metric_utils.FoldScorer(t_tr, e_tr, times, n_causes)
    names = list(MODELS)
    cifs = [preds[m][:, :, cols] for m in names]
    shape = (len(names), len(MEASURES), n_causes, len(times))

    def measure(rows):
        out = np.full(shape, np.nan)
        for i, cif in enumerate(cifs):
            values, _ = scorer.measures(cif[rows], t_te[rows], e_te[rows])
            for j, metric in enumerate(MEASURES):
                out[i, j] = values[metric]
        return out

    point = measure(np.arange(len(t_te)))
    rng = np.random.default_rng(seed)
    boot = np.stack(
        [measure(rng.integers(0, len(t_te), len(t_te))) for _ in range(replicates)]
    )
    return point, boot


def bootstrap_file(out_dir, dataset, arm, k):
    return Path(out_dir) / f"{dataset}_{arm}_fold{k}.npz"


def run(
    dataset,
    folds_file,
    folds_dir,
    pred_dir,
    id_col,
    group_col,
    n_causes,
    seeds,
    replicates,
    out_dir,
    seed=0,
):
    """Bootstrap every fold of every arm of one dataset that is not done yet."""
    spec = benchmark_utils.read_folds(folds_file)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for a, arm in enumerate(fold_utils.ARMS):
        for k in range(1, len(spec[arm]) + 1):
            target = bootstrap_file(out_dir, dataset, arm, k)
            if target.exists():
                continue
            preds = read_predictions(pred_dir, arm, k, seeds)
            missing = [m for m, v in preds.items() if v is None]
            if missing:
                report_utils.log(
                    f"{dataset} {arm} fold {k}: no predictions for {missing}, "
                    "this fold is left out of every bootstrap table"
                )
                continue
            _, t_tr, e_tr, _ = benchmark_utils.load_train(
                folds_dir, arm, k, id_col, group_col
            )
            test = pd.read_csv(Path(folds_dir) / arm / f"fold_{k}_test.csv")
            started = time.perf_counter()
            point, boot = bootstrap_fold(
                preds,
                t_tr,
                e_tr,
                test.time.to_numpy(float),
                test.event.to_numpy(int),
                n_causes,
                replicates,
                # one independent stream per arm and fold
                seed=[seed, a, k],
            )
            np.savez_compressed(target, point=point, boot=boot.astype(np.float32))
            report_utils.log(
                f"{dataset} {arm} fold {k}: {replicates} replicates, "
                f"{time.perf_counter() - started:.0f}s"
            )


def load(dataset, spec_counts, out_dir):
    """
    The saved arrays of one dataset: {arm: (point, boot)} with the folds stacked on
    axis 0, so shapes are (folds, ...) and (folds, replicates, ...). An arm with a
    fold missing is dropped and said so.
    """
    out = {}
    for arm, n_folds in spec_counts.items():
        files = [
            bootstrap_file(out_dir, dataset, arm, k) for k in range(1, n_folds + 1)
        ]
        have = [f for f in files if f.exists()]
        if len(have) < n_folds:
            report_utils.log(
                f"{dataset} {arm}: {len(have)} of {n_folds} folds bootstrapped, "
                "the arm is left out of the bootstrap tables"
            )
            continue
        loaded = [np.load(f) for f in have]
        out[arm] = (
            np.stack([z["point"] for z in loaded]),
            np.stack([z["boot"].astype(float) for z in loaded]),
        )
    return out
