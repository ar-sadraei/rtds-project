"""
Significance tests on the TUNED models, both regimes.

Same protocol as significance.py, re-stated because the comparisons changed:

  H0       the two models do not differ on the metric
  H1       they differ, TWO-SIDED
  alpha    0.05, Bonferroni across three metric families -> reject at 0.0167
  test     paired bootstrap on the test set, B = 500
  seeds    DeepHit is additionally refitted with 5 seeds per dataset, because the
           bootstrap holds the fitted model fixed and therefore understates total
           uncertainty. The seed spread is reported alongside, so a difference
           smaller than it can be dismissed on sight.

Pre-specified comparisons (fixed before looking at test results):
  T1  DeepHit(tuned, sel AUC) vs cause-specific Cox   -- tuned deep vs plain classical
  T2  DeepHit(tuned, sel AUC) vs RSF(tuned, sel AUC)  -- deep vs flexible non-deep
  T3  DeepHit(sel AUC) vs DeepHit(sel IBS)            -- does the selection criterion matter
"""

from __future__ import annotations

import ast
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from harness import HORIZONS
from metrics import DEFAULT, PREPAID, auc_td, ibs_cr, ici, kaplan_meier_censoring
from models_classical import CauseSpecificCox
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from splits import vintage_split
from tune import build_frame

ALPHA_CORRECTED = 0.05 / 3
B = 500
HORIZON = 60.0
N_SEEDS = 5

COMPARISONS = [
    ("T1", "deephit_sel_auc", "cause_specific_cox"),
    ("T2", "deephit_sel_auc", "rsf_sel_auc"),
    ("T3", "deephit_sel_auc", "deephit_sel_ibs"),
]

DATASETS = {
    "postcrisis": ("data/processed/loans.parquet", (2013, 2014, 2015), (2016, 2017, 2018)),
    "crisis": ("data/processed/loans_crisis.parquet", (2005, 2006), (2007, 2008, 2009)),
}


def configs_for(tag: str) -> dict[str, dict]:
    t = pd.read_csv("results/tuned.csv")
    t = t[t.dataset == tag]
    out = {}
    for name in t.model.unique():
        out[name] = ast.literal_eval(t[t.model == name].config.iloc[0])
    return out


def metrics(cif_h, cif_mat, t, e, cause, g_t, g_v) -> dict:
    return {
        "auc_td@60": auc_td(cif_h, t, e, cause, HORIZON, g_t, g_v),
        "ici@60": ici(cif_h, t, e, cause, HORIZON),
        "ibs": ibs_cr(cif_mat, HORIZONS, t, e, cause, g_t, g_v),
    }


def main() -> None:
    rows, seed_rows = [], []

    for tag, (path, train_v, test_v) in DATASETS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        sp = vintage_split(df, train_vintages=train_v, test_vintages=test_v)
        tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
        t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
        t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)
        g_t, g_v = kaplan_meier_censoring(t_tr, e_tr)
        cfg = configs_for(tag)

        builders = {
            "cause_specific_cox": lambda: CauseSpecificCox(),
            "rsf_sel_auc": lambda: CauseSpecificRSF(**cfg["rsf_sel_auc"]),
            "deephit_sel_auc": lambda: DeepHitModel(**cfg["deephit_sel_auc"]),
            "deephit_sel_ibs": lambda: DeepHitModel(**cfg["deephit_sel_ibs"]),
        }

        preds = {}
        for name, B_ in builders.items():
            print(f"[{tag}] fitting {name}", flush=True)
            m = B_().fit(tr[FEATS], t_tr, e_tr)
            for cause in (DEFAULT, PREPAID):
                preds[(name, cause)] = m.predict_cif(te[FEATS], HORIZONS, cause)

        # seed variability for the deep model
        for s in range(N_SEEDS):
            m = DeepHitModel(seed=s, **cfg["deephit_sel_auc"]).fit(tr[FEATS], t_tr, e_tr)
            cif = m.predict_cif(te[FEATS], HORIZONS, DEFAULT)
            j = int(np.where(HORIZONS == HORIZON)[0][0])
            v = metrics(cif[:, j], cif, t_te, e_te, DEFAULT, g_t, g_v)
            seed_rows.append({"dataset": tag, "seed": s, **v})
            print(f"[{tag}] seed {s}: auc={v['auc_td@60']:.4f} ici={v['ici@60']:.4f}",
                  flush=True)

        j = int(np.where(HORIZONS == HORIZON)[0][0])
        rng = np.random.default_rng(0)
        n = len(te)

        for cause, cname in ((DEFAULT, "default"), (PREPAID, "prepaid")):
            names = sorted({m for _, a, b in COMPARISONS for m in (a, b)})
            point = {m: metrics(preds[(m, cause)][:, j], preds[(m, cause)],
                                t_te, e_te, cause, g_t, g_v) for m in names}
            boot = {m: {k: np.empty(B) for k in point[m]} for m in names}
            for b in range(B):
                idx = rng.integers(0, n, n)
                tb, eb = t_te[idx], e_te[idx]
                for m in names:
                    c = preds[(m, cause)][idx]
                    for k, v in metrics(c[:, j], c, tb, eb, cause, g_t, g_v).items():
                        boot[m][k][b] = v
                if (b + 1) % 250 == 0:
                    print(f"  [{tag}/{cname}] bootstrap {b+1}/{B}", flush=True)

            for cid, a, bm in COMPARISONS:
                for metric in ("auc_td@60", "ici@60", "ibs"):
                    d = boot[a][metric] - boot[bm][metric]
                    d = d[~np.isnan(d)]
                    if len(d) < 50:
                        continue
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
                    p = min(1.0, max(p, 1.0 / len(d)))
                    rows.append({
                        "dataset": tag, "comparison": cid, "model_a": a, "model_b": bm,
                        "cause": cname, "metric": metric,
                        "a": round(point[a][metric], 5), "b": round(point[bm][metric], 5),
                        "diff": round(point[a][metric] - point[bm][metric], 5),
                        "ci_lo": round(lo, 5), "ci_hi": round(hi, 5), "p": round(p, 4),
                        "significant": bool(p < ALPHA_CORRECTED),
                    })

    out = pd.DataFrame(rows)
    out.to_csv("results/significance_tuned.csv", index=False)
    sd = pd.DataFrame(seed_rows)
    sd.to_csv("results/deephit_seed_spread.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n######## paired bootstrap on tuned models, Bonferroni alpha=0.0167 ########")
    print(out[out.cause == "default"].to_string(index=False))
    print("\n######## DeepHit seed-to-seed spread (default, sel_auc config) ########")
    print(sd.groupby("dataset")[["auc_td@60", "ici@60"]]
            .agg(["mean", "std", "min", "max"]).round(4).to_string())
    print("\nwrote results/significance_tuned.csv, results/deephit_seed_spread.csv")


if __name__ == "__main__":
    main()
