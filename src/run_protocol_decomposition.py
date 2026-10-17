"""
Two experiments that close the two holes a referee would open first.

HOLE 1: the vintage arm was a sample of one.
    run_random_split.py compared five random seeds against ONE vintage split per
    cohort, then claimed leads "reverse". A single draw carries no uncertainty,
    so the comparison was mean-against-point.

    Fix: rolling origin. Expand the training window one vintage at a time and
    test on the next vintage, giving four vintage splits per cohort, each with
    its own estimate.

        post-crisis   2013-14 -> 2015,  2013-15 -> 2016,
                      2013-16 -> 2017,  2013-17 -> 2018
        crisis        2005    -> 2006,  2005-06 -> 2007,
                      2005-07 -> 2008,  2005-08 -> 2009

HOLE 2: leakage and distribution shift could not be separated.
    A pooled random split removes BOTH the temporal leakage (training on loans
    originated alongside the test loans) AND the vintage shift (training and
    test periods differ). The earlier design could not say which did the work,
    and it compounded the problem by changing the test set too: the pooled
    random arm tests on loans drawn from every vintage, the vintage arm on later
    vintages only. Two things moved at once.

    Fix: hold the test loans fixed and move only the training period.

        test  T   a random half of the later vintages
        arm A     train on the EARLIER vintages, subsampled to |T|
        arm B     train on the other half of the LATER vintages, same size

    Both arms predict the same loans with the same amount of training data. The
    only difference is whether the training loans were originated before the
    test period or alongside it. Whatever separates A from B is the cost of
    having to predict forward, measured on a matched test set.

Output: results/protocol_decomposition.csv
"""

from __future__ import annotations

import ast
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from harness import HORIZONS
from metrics import DEFAULT, PREPAID, auc_td, ici, kaplan_meier_censoring
from models_classical import CauseSpecificCox, FineGray
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from tune import build_frame

HORIZON = 60.0
SEED = 0

# (parquet, rolling-origin folds, earlier vintages, later vintages)
COHORTS = {
    "postcrisis": (
        "data/processed/loans.parquet",
        [((2013, 2014), (2015,)),
         ((2013, 2014, 2015), (2016,)),
         ((2013, 2014, 2015, 2016), (2017,)),
         ((2013, 2014, 2015, 2016, 2017), (2018,))],
        (2013, 2014, 2015), (2016, 2017, 2018),
    ),
    "crisis": (
        "data/processed/loans_crisis.parquet",
        [((2005,), (2006,)),
         ((2005, 2006), (2007,)),
         ((2005, 2006, 2007), (2008,)),
         ((2005, 2006, 2007, 2008), (2009,))],
        (2005, 2006), (2007, 2008, 2009),
    ),
}


def configs_for(tag: str) -> dict[str, dict]:
    t = pd.read_csv("results/tuned.csv")
    t = t[t.dataset == tag]
    return {n: ast.literal_eval(t[t.model == n].config.iloc[0]) for n in t.model.unique()}


def builders(cfg: dict) -> dict:
    return {
        "cause_specific_cox": lambda: CauseSpecificCox(),
        "fine_gray":          lambda: FineGray(),
        "rsf":                lambda: CauseSpecificRSF(**cfg["rsf_sel_auc"]),
        "deephit":            lambda: DeepHitModel(**cfg["deephit_sel_auc"]),
    }


def score(tr, te, FEATS, cfg, rows, **tags) -> None:
    t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
    t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)
    g_t, g_v = kaplan_meier_censoring(t_tr, e_tr)
    j = int(np.where(HORIZONS == HORIZON)[0][0])
    for name, build in builders(cfg).items():
        m = build().fit(tr[FEATS], t_tr, e_tr)
        for cause, cname in ((DEFAULT, "default"), (PREPAID, "prepaid")):
            cif = m.predict_cif(te[FEATS], HORIZONS, cause)
            rows.append({**tags, "model": name, "cause": cname,
                         "auc": auc_td(cif[:, j], t_te, e_te, cause, HORIZON, g_t, g_v),
                         "ici": ici(cif[:, j], t_te, e_te, cause, HORIZON),
                         "n_train": len(tr), "n_test": len(te)})
        print(f"    {name} done", flush=True)


def main() -> None:
    rows: list[dict] = []
    rng = np.random.default_rng(SEED)

    for tag, (path, folds, earlier, later) in COHORTS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        cfg = configs_for(tag)
        vint = df.vintage.to_numpy()

        # ---- experiment 1: rolling origin --------------------------------
        for tr_v, te_v in folds:
            tr_i = np.flatnonzero(np.isin(vint, tr_v))
            te_i = np.flatnonzero(np.isin(vint, te_v))
            print(f"[{tag}] rolling {tr_v[0]}-{tr_v[-1]} -> {te_v[0]} "
                  f"(train {len(tr_i):,}, test {len(te_i):,})", flush=True)
            score(frame.iloc[tr_i], frame.iloc[te_i], FEATS, cfg, rows,
                  cohort=tag, experiment="rolling", train_end=tr_v[-1],
                  test_vintage=te_v[0])

        # ---- experiment 2: same test loans, training period moved --------
        later_i = np.flatnonzero(np.isin(vint, later))
        perm = rng.permutation(later_i)
        half = len(perm) // 2
        test_i, contemp_i = perm[:half], perm[half:]

        earlier_i = np.flatnonzero(np.isin(vint, earlier))
        earlier_i = rng.permutation(earlier_i)[:len(contemp_i)]   # match size

        print(f"[{tag}] matched test {len(test_i):,} loans from {later}; "
              f"train arms A={len(earlier_i):,} (earlier) "
              f"B={len(contemp_i):,} (contemporaneous)", flush=True)
        for arm, idx in (("earlier", earlier_i), ("contemporaneous", contemp_i)):
            print(f"  arm {arm}", flush=True)
            score(frame.iloc[idx], frame.iloc[test_i], FEATS, cfg, rows,
                  cohort=tag, experiment="matched", arm=arm,
                  train_end=-1, test_vintage=-1)

        pd.DataFrame(rows).to_csv("results/protocol_decomposition.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv("results/protocol_decomposition.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n######## rolling origin: spread of the vintage protocol ########")
    r = out[out.experiment == "rolling"]
    piv = r.pivot_table(index=["cohort", "cause", "model"], values="auc",
                        aggfunc=["mean", "std", "min", "max"])
    print(piv.round(4).to_string())

    print("\n######## DeepHit minus each baseline, per rolling fold ########")
    g = r.pivot_table(index=["cohort", "cause", "test_vintage"],
                      columns="model", values="auc")
    g["dh_cox"] = g["deephit"] - g["cause_specific_cox"]
    g["dh_rsf"] = g["deephit"] - g["rsf"]
    print(g[["dh_cox", "dh_rsf"]].round(4).to_string())
    print("\nmean (sd) over folds:")
    print(g.groupby(level=[0, 1])[["dh_cox", "dh_rsf"]]
           .agg(["mean", "std"]).round(4).to_string())

    print("\n######## matched test set: earlier vs contemporaneous training ########")
    m = out[out.experiment == "matched"].pivot_table(
        index=["cohort", "cause", "model"], columns="arm", values="auc")
    m["contemporaneous - earlier"] = m["contemporaneous"] - m["earlier"]
    print(m.round(4).to_string())

    print("\nwrote results/protocol_decomposition.csv")


if __name__ == "__main__":
    main()
