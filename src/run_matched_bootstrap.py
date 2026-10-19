"""
Inference for the matched-test experiment.

run_protocol_decomposition.py reported point estimates: DeepHit leads the forest
in every cell when trained on contemporaneous loans and trails in three of four
when trained on earlier ones. Point estimates are not a test, and every other
comparison in the paper carries one.

The design makes a strong test available. Both arms predict the SAME test loans,
so resampling those loans resamples both arms together and the pairing is exact.
Two quantities are bootstrapped:

  bonus         AUC(contemporaneous) - AUC(earlier), per model.
                "What is training alongside the test loans worth?"

  did           [DeepHit - baseline]_contemporaneous
                  - [DeepHit - baseline]_earlier
                A difference in differences. It asks whether the deep model's
                standing relative to a baseline CHANGES with the training
                period, which is the paper's claim. Because both differences
                are taken on the same loans with the same baseline, model-level
                and loan-level difficulty cancel twice.

The split is reproduced exactly: the same seed drives the same permutations in
the same order as the original run, so the test loans here are those loans.

Output: results/matched_bootstrap.csv
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
SEED = 0          # must match run_protocol_decomposition.py
B = 500
BASELINES = ("cause_specific_cox", "rsf")

COHORTS = {
    "postcrisis": ("data/processed/loans.parquet", (2013, 2014, 2015), (2016, 2017, 2018)),
    "crisis":     ("data/processed/loans_crisis.parquet", (2005, 2006), (2007, 2008, 2009)),
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


def pval(d: np.ndarray) -> float:
    """Two-sided percentile p, floored at 1/B as elsewhere in this project."""
    d = d[~np.isnan(d)]
    if len(d) < 50:
        return float("nan")
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(d))))


def main() -> None:
    rng_split = np.random.default_rng(SEED)
    rows: list[dict] = []

    for tag, (path, earlier, later) in COHORTS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        cfg = configs_for(tag)
        vint = df.vintage.to_numpy()

        # --- reproduce the original split, same rng call order ------------
        later_i = np.flatnonzero(np.isin(vint, later))
        perm = rng_split.permutation(later_i)
        half = len(perm) // 2
        test_i, contemp_i = perm[:half], perm[half:]
        earlier_i = rng_split.permutation(
            np.flatnonzero(np.isin(vint, earlier)))[:len(contemp_i)]

        te = frame.iloc[test_i]
        t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)
        j = int(np.where(HORIZONS == HORIZON)[0][0])
        print(f"[{tag}] test {len(te):,} loans; arms {len(earlier_i):,} earlier / "
              f"{len(contemp_i):,} contemporaneous", flush=True)

        # --- fit both arms, keep predictions and each arm's own weights ---
        pred: dict[tuple, np.ndarray] = {}
        gw: dict[str, tuple] = {}
        for arm, idx in (("earlier", earlier_i), ("contemporaneous", contemp_i)):
            tr = frame.iloc[idx]
            t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
            gw[arm] = kaplan_meier_censoring(t_tr, e_tr)
            for name, build in builders(cfg).items():
                m = build().fit(tr[FEATS], t_tr, e_tr)
                for cause in (DEFAULT, PREPAID):
                    pred[(arm, name, cause)] = m.predict_cif(te[FEATS], HORIZONS, cause)[:, j]
                print(f"  [{tag}/{arm}] {name} fitted", flush=True)

        # --- paired bootstrap over the shared test loans ------------------
        rng = np.random.default_rng(1)
        n = len(te)
        names = list(builders(cfg))
        boot = {(a, m, c): np.empty(B)
                for a in ("earlier", "contemporaneous") for m in names
                for c in (DEFAULT, PREPAID)}

        for b in range(B):
            k = rng.integers(0, n, n)
            tb, eb = t_te[k], e_te[k]
            for arm in ("earlier", "contemporaneous"):
                g_t, g_v = gw[arm]
                for name in names:
                    for cause in (DEFAULT, PREPAID):
                        boot[(arm, name, cause)][b] = auc_td(
                            pred[(arm, name, cause)][k], tb, eb, cause,
                            HORIZON, g_t, g_v)
            if (b + 1) % 125 == 0:
                print(f"  [{tag}] bootstrap {b+1}/{B}", flush=True)

        # --- the two quantities -------------------------------------------
        for cause, cname in ((DEFAULT, "default"), (PREPAID, "prepaid")):
            point = {(a, m): auc_td(pred[(a, m, cause)], t_te, e_te, cause,
                                    HORIZON, *gw[a])
                     for a in ("earlier", "contemporaneous") for m in names}

            for name in names:
                d = boot[("contemporaneous", name, cause)] - boot[("earlier", name, cause)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({"cohort": tag, "cause": cname, "quantity": "bonus",
                             "model": name, "baseline": "",
                             "estimate": round(point[("contemporaneous", name)]
                                               - point[("earlier", name)], 5),
                             "ci_lo": round(lo, 5), "ci_hi": round(hi, 5),
                             "p": pval(d)})

            for base in BASELINES:
                dc = boot[("contemporaneous", "deephit", cause)] - boot[("contemporaneous", base, cause)]
                de = boot[("earlier", "deephit", cause)] - boot[("earlier", base, cause)]
                d = dc - de
                lo, hi = np.percentile(d, [2.5, 97.5])
                est = ((point[("contemporaneous", "deephit")] - point[("contemporaneous", base)])
                       - (point[("earlier", "deephit")] - point[("earlier", base)]))
                rows.append({"cohort": tag, "cause": cname, "quantity": "did",
                             "model": "deephit", "baseline": base,
                             "estimate": round(est, 5),
                             "ci_lo": round(lo, 5), "ci_hi": round(hi, 5),
                             "p": pval(d)})

        pd.DataFrame(rows).to_csv("results/matched_bootstrap.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv("results/matched_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n######## contemporaneity bonus: AUC(contemp) - AUC(earlier) ########")
    print(out[out.quantity == "bonus"][
        ["cohort", "cause", "model", "estimate", "ci_lo", "ci_hi", "p"]
    ].to_string(index=False))

    print("\n######## difference in differences: does DeepHit's standing change? ########")
    print(out[out.quantity == "did"][
        ["cohort", "cause", "baseline", "estimate", "ci_lo", "ci_hi", "p"]
    ].to_string(index=False))

    sig = out[(out.quantity == "did") & (out.p < 0.05 / 3)]
    print(f"\nDiD significant at the Bonferroni level 0.05/3: {len(sig)} of "
          f"{len(out[out.quantity=='did'])}")
    print("\nwrote results/matched_bootstrap.csv")


if __name__ == "__main__":
    main()
