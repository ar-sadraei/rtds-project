"""
Significance testing for the model comparison.

Protocol, fixed before results are inspected (Lecture 8):

  H0      the two models do not differ on the metric under test
  H1      they differ. TWO-SIDED -- no direction was pre-committed
  alpha   0.05
  test    paired bootstrap on the test set: resample test subjects with
          replacement, recompute both models' metrics on the SAME resample, and
          take the distribution of the paired difference. Paired because some
          subjects are hard for every model, which is the reason Lecture 8 gives
          for pairing.
  p       two-sided, from the bootstrap difference distribution
  correct Bonferroni across the three metric families -> reject at 0.05/3 = 0.0167

Why a bootstrap rather than a t-test across seeds: the protocol of interest is the
vintage split, and there is exactly one of those. Resampling the test set is the
standard way to get a sampling distribution from a single test set, and it keeps
the pairing intact.

Comparisons are PRE-SPECIFIED to limit multiplicity:
  C1  DeepHit(alpha=0.2) vs cause-specific Cox   -- the central claim
  C2  DeepHit(alpha=1.0) vs DeepHit(alpha=0.2)   -- does the ranking term help?
  C3  RSF vs cause-specific Cox                  -- is the flexible model fragile?
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from harness import AalenJohansenBaseline, HORIZONS
from metrics import DEFAULT, PREPAID, auc_td, ibs_cr, ici, kaplan_meier_censoring
from models_classical import CauseSpecificCox, FineGray
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from splits import vintage_split

ALPHA = 0.05
N_FAMILIES = 3                      # discrimination, calibration, accuracy
ALPHA_CORRECTED = ALPHA / N_FAMILIES
B = 500
HORIZON = 60.0

COMPARISONS = [
    ("C1", "deephit_a0.2", "cause_specific_cox"),
    ("C2", "deephit_a1", "deephit_a0.2"),
    ("C3", "rsf", "cause_specific_cox"),
]

FEATURES = ["fico", "cltv", "dti", "ltv", "orig_upb", "orig_rate",
            "orig_term", "n_borrowers", "mi_pct"]


def _metrics(cif_h, cif_mat, t, e, cause, g_t, g_v):
    """The three families on one (possibly resampled) test set."""
    return {
        "auc_td@60": auc_td(cif_h, t, e, cause, HORIZON, g_t, g_v),
        "ici@60": ici(cif_h, t, e, cause, HORIZON),
        "ibs": ibs_cr(cif_mat, HORIZONS, t, e, cause, g_t, g_v),
    }


def main() -> None:
    df = pd.read_parquet("data/processed/loans.parquet")
    sp = vintage_split(df)
    tr, te = df.iloc[sp.train_idx], df.iloc[sp.test_idx]
    t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
    t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)
    g_t, g_v = kaplan_meier_censoring(t_tr, e_tr)

    builders = {
        "aalen_johansen": AalenJohansenBaseline,
        "cause_specific_cox": CauseSpecificCox,
        "fine_gray": FineGray,
        "rsf": CauseSpecificRSF,
        "deephit_a0.2": lambda: DeepHitModel(alpha=0.2, epochs=50),
        "deephit_a1": lambda: DeepHitModel(alpha=1.0, epochs=50),
    }

    needed = {m for _, a, b in COMPARISONS for m in (a, b)}
    preds: dict[tuple[str, int], np.ndarray] = {}
    for name in sorted(needed):
        print(f"fitting {name} ...", flush=True)
        model = builders[name]().fit(tr[FEATURES], t_tr, e_tr)
        for cause in (DEFAULT, PREPAID):
            preds[(name, cause)] = model.predict_cif(te[FEATURES], HORIZONS, cause)

    h_idx = int(np.where(HORIZONS == HORIZON)[0][0])
    rng = np.random.default_rng(0)
    n = len(te)
    rows = []

    for cause, cname in ((DEFAULT, "default"), (PREPAID, "prepaid")):
        # point estimates on the full test set
        point = {m: _metrics(preds[(m, cause)][:, h_idx], preds[(m, cause)],
                             t_te, e_te, cause, g_t, g_v) for m in needed}

        # paired bootstrap
        boot = {m: {k: np.empty(B) for k in point[m]} for m in needed}
        for b in range(B):
            idx = rng.integers(0, n, n)
            tb, eb = t_te[idx], e_te[idx]
            for m in needed:
                cif = preds[(m, cause)][idx]
                vals = _metrics(cif[:, h_idx], cif, tb, eb, cause, g_t, g_v)
                for k, v in vals.items():
                    boot[m][k][b] = v
            if (b + 1) % 100 == 0:
                print(f"  {cname}: bootstrap {b+1}/{B}", flush=True)

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
                    "comparison": cid, "model_a": a, "model_b": bm,
                    "cause": cname, "metric": metric,
                    "a": round(point[a][metric], 5),
                    "b": round(point[bm][metric], 5),
                    "diff": round(point[a][metric] - point[bm][metric], 5),
                    "ci_lo": round(lo, 5), "ci_hi": round(hi, 5),
                    "p": round(p, 4),
                    "significant": bool(p < ALPHA_CORRECTED),
                })

    out = pd.DataFrame(rows)
    out.to_csv("results/significance.csv", index=False)

    pd.set_option("display.width", 240)
    print(f"\nPaired bootstrap, B={B}, vintage split, two-sided, "
          f"Bonferroni alpha = {ALPHA}/{N_FAMILIES} = {ALPHA_CORRECTED:.4f}\n")
    print(out.to_string(index=False))
    print("\nwrote results/significance.csv")


if __name__ == "__main__":
    main()
