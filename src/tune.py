"""
Equal-budget hyperparameter search, with the selection criterion as an experiment.

Protocol
--------
Budget      20 random configurations for DeepHit, 8 for RSF. Cox and Fine-Gray have
            nothing to tune beyond the ridge, so they are reported untuned. The
            budget is fixed in advance and reported -- which is exactly the gap
            Filip's review identifies: benchmarking papers tune inconsistently and
            rarely say what they spent.

Validation  an OUT-OF-TIME split inside the training set, mirroring the outer
            protocol. The test vintages are never touched during the search.
              post-crisis  inner-train 2013-2014, inner-val 2015
              crisis       inner-train 2005,      inner-val 2006

Selection   every configuration is scored on validation AUC@60 AND validation IBS,
            and the best configuration under EACH criterion is refitted on the full
            training set and evaluated on the test vintages.

            This is the point. DeepHit's own paper tunes its loss weight on
            validation discrimination. If selecting on AUC produces a worse
            calibrated model than selecting on IBS -- from the same search, the
            same budget, the same data -- then the review's claim that these models
            are "trained and selected for ranking" is demonstrated rather than
            asserted.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from harness import HORIZONS, evaluate
from metrics import DEFAULT, auc_td, ibs_cr, kaplan_meier_censoring
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from splits import Split, vintage_split

BASE = ["fico", "cltv", "dti", "ltv", "orig_upb", "orig_rate",
        "orig_term", "n_borrowers", "mi_pct"]
CATS = ["state", "property_type", "occupancy", "channel",
        "loan_purpose", "first_time_buyer"]
N_DEEPHIT, N_RSF, HORIZON = 20, 8, 60.0

# config keys per family -- the trials table holds the union of both, with NaN
# where a key does not apply, so the refit step must not read across families.
CFG_KEYS = {
    "deephit": ["alpha", "sigma", "n_bins", "dropout", "lr", "batch_size", "epochs"],
    "rsf": ["n_estimators", "min_samples_leaf", "max_features"],
}
INT_KEYS = {"n_bins", "batch_size", "epochs", "n_estimators", "min_samples_leaf"}


def build_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rich = df[BASE].copy()
    for c in CATS:
        rich = pd.concat(
            [rich, pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)],
            axis=1)
    frame = pd.concat([df[["vintage", "time", "event", "last_age"]], rich], axis=1)
    return frame, list(rich.columns)


def sample_deephit(rng) -> dict:
    return {
        "alpha": float(rng.choice([0.1, 0.2, 0.35, 0.5, 0.75, 1.0])),
        "sigma": float(rng.choice([0.05, 0.1, 0.25, 0.5])),
        "n_bins": int(rng.choice([20, 30, 50])),
        "dropout": float(rng.choice([0.0, 0.1, 0.2, 0.3])),
        "lr": float(rng.choice([3e-4, 1e-3, 3e-3])),
        "batch_size": int(rng.choice([256, 512, 1024])),
        "epochs": 50,
    }


def sample_rsf(rng) -> dict:
    return {
        "n_estimators": int(rng.choice([100, 200, 300])),
        "min_samples_leaf": int(rng.choice([10, 25, 50, 100])),
        "max_features": str(rng.choice(["sqrt", "log2"])),
    }


def score_on_val(model, Xtr, ttr, etr, Xva, tva, eva) -> dict:
    g_t, g_v = kaplan_meier_censoring(ttr, etr)
    t0 = time.perf_counter()
    model.fit(Xtr, ttr, etr)
    fit_s = time.perf_counter() - t0
    cif = model.predict_cif(Xva, HORIZONS, DEFAULT)
    j = int(np.where(HORIZONS == HORIZON)[0][0])
    return {
        "val_auc": auc_td(cif[:, j], tva, eva, DEFAULT, HORIZON, g_t, g_v),
        "val_ibs": ibs_cr(cif, HORIZONS, tva, eva, DEFAULT, g_t, g_v),
        "fit_s": fit_s,
    }


def tune_dataset(tag: str, path: str, train_v, val_v, test_v) -> pd.DataFrame:
    df = pd.read_parquet(path)
    frame, FEATS = build_frame(df)

    inner_tr = np.flatnonzero(frame.vintage.isin(train_v).to_numpy())
    inner_va = np.flatnonzero(frame.vintage.isin(val_v).to_numpy())
    Xtr, Xva = frame.iloc[inner_tr][FEATS], frame.iloc[inner_va][FEATS]
    ttr = frame.iloc[inner_tr].time.to_numpy(float)
    etr = frame.iloc[inner_tr].event.to_numpy(int)
    tva = frame.iloc[inner_va].time.to_numpy(float)
    eva = frame.iloc[inner_va].event.to_numpy(int)
    print(f"[{tag}] inner-train {sorted(train_v)} n={len(inner_tr):,} | "
          f"inner-val {sorted(val_v)} n={len(inner_va):,}", flush=True)

    cache = f"results/tuning_trials_{tag}.csv"
    if __import__("os").path.exists(cache):
        tr = pd.read_csv(cache)
        print(f"[{tag}] reusing {len(tr)} cached trials from {cache}", flush=True)
        return _refit(tag, tr, df, frame, FEATS, train_v, val_v, test_v)

    rng = np.random.default_rng(0)
    trials = []
    for i in range(N_DEEPHIT):
        cfg = sample_deephit(rng)
        s = score_on_val(DeepHitModel(seed=i, **cfg), Xtr, ttr, etr, Xva, tva, eva)
        trials.append({"family": "deephit", "trial": i, **cfg, **s})
        print(f"  deephit {i+1:2d}/{N_DEEPHIT} a={cfg['alpha']:<4} "
              f"val_auc={s['val_auc']:.4f} val_ibs={s['val_ibs']:.5f}", flush=True)
    for i in range(N_RSF):
        cfg = sample_rsf(rng)
        s = score_on_val(CauseSpecificRSF(random_state=i, **cfg), Xtr, ttr, etr, Xva, tva, eva)
        trials.append({"family": "rsf", "trial": i, **cfg, **s})
        print(f"  rsf     {i+1:2d}/{N_RSF} val_auc={s['val_auc']:.4f} "
              f"val_ibs={s['val_ibs']:.5f}", flush=True)

    tr = pd.DataFrame(trials)
    tr["dataset"] = tag
    tr.to_csv(cache, index=False)
    return _refit(tag, tr, df, frame, FEATS, train_v, val_v, test_v)


def _refit(tag, tr, df, frame, FEATS, train_v, val_v, test_v):
    """Refit the winner under each selection criterion on the full training set."""
    # refit the winner under each criterion on the FULL training set
    outer = vintage_split(df, train_vintages=tuple(sorted(set(train_v) | set(val_v))),
                          test_vintages=tuple(test_v))
    rows = []
    for family, Cls in (("deephit", DeepHitModel), ("rsf", CauseSpecificRSF)):
        sub = tr[tr.family == family]
        for crit, best_idx in (("auc", sub.val_auc.idxmax()), ("ibs", sub.val_ibs.idxmin())):
            cfg = {}
            for k in CFG_KEYS[family]:
                v = sub.loc[best_idx, k]
                cfg[k] = int(v) if k in INT_KEYS else (str(v) if isinstance(v, str) else float(v))
            m = Cls(**cfg)
            m.name = f"{family}_sel_{crit}"
            print(f"[{tag}] refit {m.name}: {cfg}", flush=True)
            r = evaluate(m, frame, outer, FEATS)
            r["dataset"], r["selected_on"], r["family"] = tag, crit, family
            r["config"] = str(cfg)
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    out = []
    out.append(tune_dataset("postcrisis", "data/processed/loans.parquet",
                            train_v=(2013, 2014), val_v=(2015,),
                            test_v=(2016, 2017, 2018)))
    out.append(tune_dataset("crisis", "data/processed/loans_crisis.parquet",
                            train_v=(2005,), val_v=(2006,),
                            test_v=(2007, 2008, 2009)))
    res = pd.concat(out, ignore_index=True)
    res.to_csv("results/tuned.csv", index=False)

    pd.set_option("display.width", 240)
    t = (res[(res.cause == "default") & res.metric.isin(["auc_td", "ici"])
             & (res.horizon == HORIZON)]
         .pivot_table(index=["dataset", "model"], columns="metric", values="value"))
    print("\n######## tuned models, test vintages, cause = default ########")
    print(t.round(4).to_string())
    print("\nwrote results/tuned.csv")


if __name__ == "__main__":
    main()
