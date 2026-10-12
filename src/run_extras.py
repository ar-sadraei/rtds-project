"""
Two closing experiments.

A. TRAINING VARIABILITY FOR EVERY MODEL
   So far only DeepHit had a seed study, which made the significance tests
   asymmetric: DeepHit's differences were judged against its own training noise
   while the baselines had none. Seeds are the wrong device for Cox and
   Fine--Gray, which are deterministic given the data, so we use the common
   denominator instead -- resample the training set, refit, evaluate on the fixed
   test set. That captures training variability for every model class on the same
   footing.

B. POST-HOC RECALIBRATION
   Alberge et al. (2026) show recalibration improves competing-risks probabilities
   while preserving discrimination. We apply a marginal rescaling in that spirit:
   for each horizon, multiply predicted incidence by the ratio of the
   Aalen--Johansen estimate to the mean prediction on a calibration source, then
   clip to [0, 1]. This corrects calibration-in-the-large, not curve shape, and is
   a simpler device than their two measures -- we state that rather than claim
   their method.

   Fitted two ways, deliberately:
     train   the honest version. The map is estimated on the training vintages,
             which is all a lender has at scoring time.
     oracle  the map is estimated on the TEST vintages. Unattainable in practice;
             it bounds how much of the miscalibration is a fixable calibration-map
             problem rather than genuine vintage shift.

   Discrimination is unchanged by a monotone per-horizon rescaling, so AUC is
   reported once and acts as a check that the rescaling did not reorder anything.
"""

from __future__ import annotations

import ast
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from harness import HORIZONS
from metrics import (DEFAULT, PREPAID, aalen_johansen, auc_td, ici,
                     kaplan_meier_censoring)
from models_classical import CauseSpecificCox, FineGray
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from splits import vintage_split
from tune import build_frame

HOR = 60.0
N_REP = 5
DATASETS = {
    "postcrisis": ("data/processed/loans.parquet", (2013, 2014, 2015), (2016, 2017, 2018)),
    "crisis": ("data/processed/loans_crisis.parquet", (2005, 2006), (2007, 2008, 2009)),
}


def recal_factors(cif_src: np.ndarray, t_src: np.ndarray, e_src: np.ndarray,
                  cause: int) -> np.ndarray:
    """Per-horizon ratio of the Aalen--Johansen estimate to the mean prediction."""
    aj = aalen_johansen(t_src, e_src, cause, HORIZONS)
    mean_pred = np.clip(cif_src.mean(axis=0), 1e-8, None)
    return aj / mean_pred


def main() -> None:
    rows_var, rows_rec = [], []

    for tag, (path, train_v, test_v) in DATASETS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        sp = vintage_split(df, train_vintages=train_v, test_vintages=test_v)
        tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
        t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
        t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)
        g_t, g_v = kaplan_meier_censoring(t_tr, e_tr)
        j = int(np.where(HORIZONS == HOR)[0][0])

        cf = pd.read_csv("results/tuned.csv").query("dataset == @tag")
        cfg = lambda n: ast.literal_eval(cf[cf.model == n].config.iloc[0])
        builders = {
            "cause_specific_cox": lambda: CauseSpecificCox(),
            "fine_gray": lambda: FineGray(),
            "rsf": lambda: CauseSpecificRSF(**cfg("rsf_sel_auc")),
            "deephit": lambda: DeepHitModel(**cfg("deephit_sel_auc")),
        }

        # ---- A. training variability --------------------------------------
        rng = np.random.default_rng(0)
        for name, B in builders.items():
            for rep in range(N_REP):
                idx = rng.integers(0, len(tr), len(tr))
                sub = tr.iloc[idx]
                m = B()
                if hasattr(m, "seed"):
                    m.seed = rep
                m.fit(sub[FEATS], sub.time.to_numpy(float), sub.event.to_numpy(int))
                cif = m.predict_cif(te[FEATS], HORIZONS, DEFAULT)
                rows_var.append({
                    "dataset": tag, "model": name, "rep": rep,
                    "auc": auc_td(cif[:, j], t_te, e_te, DEFAULT, HOR, g_t, g_v),
                    "ici": ici(cif[:, j], t_te, e_te, DEFAULT, HOR),
                })
            print(f"[{tag}] variability done: {name}", flush=True)

        # ---- B. recalibration ----------------------------------------------
        for name, B in builders.items():
            m = B().fit(tr[FEATS], t_tr, e_tr)
            for cause, cname in ((DEFAULT, "default"), (PREPAID, "prepaid")):
                cif_tr = m.predict_cif(tr[FEATS], HORIZONS, cause)
                cif_te = m.predict_cif(te[FEATS], HORIZONS, cause)

                variants = {
                    "raw": cif_te,
                    "recal_train": np.clip(
                        cif_te * recal_factors(cif_tr, t_tr, e_tr, cause), 0, 1),
                    "recal_oracle": np.clip(
                        cif_te * recal_factors(cif_te, t_te, e_te, cause), 0, 1),
                }
                for vname, c in variants.items():
                    rows_rec.append({
                        "dataset": tag, "model": name, "cause": cname,
                        "variant": vname,
                        "auc": auc_td(c[:, j], t_te, e_te, cause, HOR, g_t, g_v),
                        "ici": ici(c[:, j], t_te, e_te, cause, HOR),
                    })
            print(f"[{tag}] recalibration done: {name}", flush=True)

    var = pd.DataFrame(rows_var)
    rec = pd.DataFrame(rows_rec)
    var.to_csv("results/training_variability.csv", index=False)
    rec.to_csv("results/recalibration.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n######## A. training variability (default, AUC@60 / ICI@60) ########")
    print(var.groupby(["dataset", "model"])[["auc", "ici"]]
             .agg(["mean", "std"]).round(4).to_string())
    print("\n######## B. recalibration (ICI@60) ########")
    print(rec.pivot_table(index=["dataset", "cause", "model"],
                          columns="variant", values="ici")
             [["raw", "recal_train", "recal_oracle"]].round(4).to_string())
    print("\n######## B. AUC unchanged by rescaling? ########")
    print(rec.pivot_table(index=["dataset", "cause", "model"],
                          columns="variant", values="auc")
             [["raw", "recal_train"]].round(4).to_string())
    print("\nwrote results/training_variability.csv, results/recalibration.csv")


if __name__ == "__main__":
    main()
