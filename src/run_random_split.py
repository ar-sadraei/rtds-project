"""
What is the evaluation protocol worth?

The benchmarking literature evaluates competing-risks survival models on RANDOM
train/test splits -- Blumenstock et al. used ten of them on US mortgages, with the
same model set and the same event pair we use. We evaluate on vintage-ordered
splits, because a lender fits on loans it has already originated and scores loans
it has not.

This script runs both protocols on the same data, with the same models and the
same hyperparameters, so the only thing that changes is how the test set is
chosen. The difference between the two is an estimate of how much of a reported
advantage is a property of the protocol rather than of the model.

Design decisions that matter:

  same configs      The tuned configurations come from results/tuned.csv, which
                    were selected on an OUT-OF-TIME inner validation vintage.
                    They are not re-tuned here. Re-tuning per protocol would
                    confound "the protocol flatters the model" with "the protocol
                    flatters the search". Holding them fixed means the configs
                    were chosen for out-of-time generalisation, which if anything
                    biases AGAINST finding a random-split advantage -- so the
                    effect we measure is a lower bound.

  matched sizes     The random split uses the same train/test sizes as its
                    cohort's vintage split (post-crisis 150k/150k, crisis
                    100k/150k), so nothing moves because of sample size.

  same weights      Censoring weights are fitted on the training side only, under
                    both protocols, exactly as in the main experiment.

Output: results/protocol_comparison.csv, one row per
(cohort, protocol, seed, model, cause) with AUC and the calibration index at 60
months.
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
from splits import random_split, vintage_split
from tune import build_frame

HORIZON = 60.0
N_SEEDS = 5

DATASETS = {
    "postcrisis": ("data/processed/loans.parquet", (2013, 2014, 2015), (2016, 2017, 2018)),
    "crisis":     ("data/processed/loans_crisis.parquet", (2005, 2006), (2007, 2008, 2009)),
}


def configs_for(tag: str) -> dict[str, dict]:
    t = pd.read_csv("results/tuned.csv")
    t = t[t.dataset == tag]
    return {name: ast.literal_eval(t[t.model == name].config.iloc[0])
            for name in t.model.unique()}


def builders(cfg: dict) -> dict:
    """Same four models as the main experiment, same tuned settings."""
    return {
        "cause_specific_cox": lambda: CauseSpecificCox(),
        "fine_gray":          lambda: FineGray(),
        "rsf":                lambda: CauseSpecificRSF(**cfg["rsf_sel_auc"]),
        "deephit":            lambda: DeepHitModel(**cfg["deephit_sel_auc"]),
    }


def score(tr, te, FEATS, cfg, rows, cohort, protocol, seed) -> None:
    t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
    t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)
    g_t, g_v = kaplan_meier_censoring(t_tr, e_tr)
    j = int(np.where(HORIZONS == HORIZON)[0][0])

    for name, build in builders(cfg).items():
        m = build().fit(tr[FEATS], t_tr, e_tr)
        for cause, cname in ((DEFAULT, "default"), (PREPAID, "prepaid")):
            cif = m.predict_cif(te[FEATS], HORIZONS, cause)
            rows.append({
                "cohort": cohort, "protocol": protocol, "seed": seed,
                "model": name, "cause": cname,
                "auc": auc_td(cif[:, j], t_te, e_te, cause, HORIZON, g_t, g_v),
                "ici": ici(cif[:, j], t_te, e_te, cause, HORIZON),
                "n_train": len(tr), "n_test": len(te),
            })
        print(f"  [{cohort}/{protocol}/seed{seed}] {name} done", flush=True)


def main() -> None:
    rows: list[dict] = []

    for tag, (path, train_v, test_v) in DATASETS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        cfg = configs_for(tag)

        # --- protocol A: vintage-ordered (ours) ----------------------------
        sp = vintage_split(df, train_vintages=train_v, test_vintages=test_v)
        n_tr, n_te = len(sp.train_idx), len(sp.test_idx)
        print(f"[{tag}] vintage split: train {n_tr:,} test {n_te:,}", flush=True)
        score(frame.iloc[sp.train_idx], frame.iloc[sp.test_idx],
              FEATS, cfg, rows, tag, "vintage", -1)

        # --- protocol B: random, with the SAME train/test sizes ------------
        frac = n_te / (n_tr + n_te)
        print(f"[{tag}] random splits at test_frac={frac:.3f}", flush=True)
        for s in range(N_SEEDS):
            rs = random_split(df, seed=s, test_frac=frac)
            score(frame.iloc[rs.train_idx], frame.iloc[rs.test_idx],
                  FEATS, cfg, rows, tag, "random", s)

        pd.DataFrame(rows).to_csv("results/protocol_comparison.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv("results/protocol_comparison.csv", index=False)

    # ---- the comparison the paper actually needs --------------------------
    pd.set_option("display.width", 200)
    print("\n######## AUC at 60 months by protocol ########")
    piv = out.pivot_table(index=["cohort", "cause", "model"],
                          columns="protocol", values="auc", aggfunc="mean")
    piv["random - vintage"] = piv["random"] - piv["vintage"]
    print(piv.round(4).to_string())

    print("\n######## the gap between DeepHit and each baseline, by protocol ########")
    g = out.pivot_table(index=["cohort", "cause", "protocol"],
                        columns="model", values="auc", aggfunc="mean")
    for base in ("cause_specific_cox", "rsf"):
        g[f"deephit - {base}"] = g["deephit"] - g[base]
    print(g[[c for c in g.columns if c.startswith("deephit - ")]].round(4).to_string())

    print("\nwrote results/protocol_comparison.csv")


if __name__ == "__main__":
    main()
