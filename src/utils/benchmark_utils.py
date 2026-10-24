#!/usr/bin/env python3
"""
Fold loading, hyperparameter search and the benchmark loop.

The loop knows nothing about either domain. Everything that differs between the
financial and the medical data is passed in by the pipeline: where the folds are,
which columns are identifiers, how many causes there are, how many folds.

RESUMABLE
    run() reads the output CSV first and skips any (arm, fold, model) cell already
    in it, so a crashed or interrupted run continues instead of starting over. 128
    model fits is long enough that this matters.

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

from . import metric_utils, report_utils
from .model_utils import MODELS, SEARCH_SPACE

HORIZONS = np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0])
PRIMARY_HORIZON = 72.0


def features(df, id_col, group_col):
    drop = {id_col, group_col, "time", "event"}
    return [c for c in df.columns if c not in drop]


def load_fold(folds_dir, arm, k, id_col, group_col):
    """Read one materialized fold pair and return arrays ready to fit."""
    folder = Path(folds_dir) / arm
    tr = pd.read_csv(folder / f"fold_{k}_train.csv")
    te = pd.read_csv(folder / f"fold_{k}_test.csv")
    cols = features(tr, id_col, group_col)
    return (
        tr[cols].to_numpy(float),
        tr.time.to_numpy(float),
        tr.event.to_numpy(int),
        te[cols].to_numpy(float),
        te.time.to_numpy(float),
        te.event.to_numpy(int),
        cols,
    )


def sample_configs(space, budget, seed=0):
    """Every combination when the grid is small, otherwise a random sample of it."""
    if not space:
        return [{}]
    keys = list(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    if len(grid) <= budget:
        chosen = grid
    else:
        rng = np.random.default_rng(seed)
        chosen = [grid[i] for i in rng.choice(len(grid), budget, replace=False)]
    return [dict(zip(keys, c)) for c in chosen]


def tune(folds_dir, arm, n_folds, id_col, group_col, n_causes, budget, out_file):
    """Pick hyperparameters once per model on an out-of-time validation split."""
    X, t, e, _, _, _, _ = load_fold(folds_dir, arm, n_folds, id_col, group_col)
    groups = pd.read_csv(Path(folds_dir) / arm / f"fold_{n_folds}_train.csv")[group_col]
    levels = sorted(groups.unique())
    is_val = (groups == levels[-1]).to_numpy()
    report_utils.log(
        f"Validation split: train on {levels[0]} to {levels[-2]} "
        f"({int((~is_val).sum()):,} rows), validate on {levels[-1]} "
        f"({int(is_val.sum()):,} rows)"
    )

    best = {}
    summary = []
    for name, fit in MODELS.items():
        configs = sample_configs(SEARCH_SPACE[name], budget)
        if configs == [{}]:
            report_utils.log(f"{name:<10} no hyperparameters, untuned")
            best[name] = {}
            summary.append(
                {"Model": name, "Configs": 0, "Best IBS": np.nan, "Settings": "untuned"}
            )
            continue
        scores = []
        for i, params in enumerate(configs, 1):
            tag = f"{name:<10} {i:>2}/{len(configs)}"
            try:
                cif, _, _ = fit(
                    params,
                    X[~is_val],
                    t[~is_val],
                    e[~is_val],
                    X[is_val],
                    HORIZONS,
                    n_causes,
                )
                rows = metric_utils.score(
                    cif,
                    t[~is_val],
                    e[~is_val],
                    t[is_val],
                    e[is_val],
                    HORIZONS,
                    n_causes,
                )
                ibs = [r["value"] for r in rows if r["metric"] == "ibs" and r["value"]]
                score = float(np.mean(ibs)) if ibs else np.inf
                scores.append((score, params))
                report_utils.log(
                    f"{tag}  IBS {score:.5f}  {report_utils.settings(params)}"
                )
            except Exception as err:
                scores.append((np.inf, params))
                report_utils.log(
                    f"{tag}  failed ({type(err).__name__})  "
                    f"{report_utils.settings(params)}"
                )
        scores.sort(key=lambda s: s[0])
        best[name] = scores[0][1]
        summary.append(
            {
                "Model": name,
                "Configs": len(configs),
                "Best IBS": scores[0][0],
                "Settings": report_utils.settings(scores[0][1], sep="\n"),
            }
        )

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(best, indent=1, default=str))
    report_utils.log(f"Saved {out_file}")
    report_utils.h2("Results")
    report_utils.table(pd.DataFrame(summary))
    return best


def read_params(path):
    p = Path(path)
    if not p.exists():
        report_utils.log(f"No tuning file at {p}, using library defaults")
        return {name: {} for name in MODELS}
    raw = json.loads(p.read_text())
    # json turns the layer lists into strings when they came from a tuple
    for name, params in raw.items():
        for key, val in list(params.items()):
            if isinstance(val, str) and val.startswith("["):
                params[key] = json.loads(val)
    return raw


def run(label, folds_dir, n_folds, id_col, group_col, n_causes, params, out_file):
    """Fit every model on every fold of both arms and append one row per number."""
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(
            map(tuple, old[["arm", "fold", "model"]].drop_duplicates().to_numpy())
        )
        report_utils.log(
            f"{out} already holds {len(done)} finished cells, skipping those"
        )

    stats = []
    for arm in ("temporal", "random"):
        for k in range(1, n_folds + 1):
            X_tr, t_tr, e_tr, X_te, t_te, e_te, cols = load_fold(
                folds_dir, arm, k, id_col, group_col
            )
            for name, fit in MODELS.items():
                if (arm, k, name) in done:
                    continue
                started = time.perf_counter()
                try:
                    cif, fit_s, pred_s = fit(
                        params.get(name, {}), X_tr, t_tr, e_tr, X_te, HORIZONS, n_causes
                    )
                    rows = metric_utils.score(
                        cif, t_tr, e_tr, t_te, e_te, HORIZONS, n_causes
                    )
                    failed = ""
                except Exception as err:
                    rows, fit_s, pred_s = [], np.nan, np.nan
                    failed = f"{type(err).__name__}: {err}"

                frame = pd.DataFrame(rows)
                frame.insert(0, "dataset", label)
                frame.insert(1, "arm", arm)
                frame.insert(2, "fold", k)
                frame.insert(3, "model", name)
                frame["n_train"] = len(X_tr)
                frame["n_test"] = len(X_te)
                frame["n_features"] = len(cols)
                frame["fit_seconds"] = fit_s
                frame["predict_seconds"] = pred_s
                frame["error"] = failed
                frame.to_csv(out, mode="a", header=not out.exists(), index=False)

                took = time.perf_counter() - started
                stats.append({"Model": name, "Failed": bool(failed), "Seconds": took})
                note = failed if failed else f"{len(rows)} numbers"
                report_utils.log(f"{arm:<8} fold {k} {name:<11} {took:>6.1f}s  {note}")
    report_utils.log(f"Wrote {out}")
    if stats:
        report_utils.h2("This run")
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
