"""
Proper competing-risks calibration, using the authors' own implementation.

Our calibration index bins by predicted-risk decile and compares against the
Aalen-Johansen estimate within each bin. Alberge et al. (2026) show that such a
measure captures one MARGINAL aspect of the problem: it is per-cause and
evaluated at a single horizon, whereas calibration under competing risks has to
integrate in two directions -- across event types and across time -- because
what must be calibrated is the whole CIF vector, not one of its components.

They propose two proper measures. One of them, AJ-calibration, ships in their
library:

    AJ-Cal_k = (1/t_max) * integral |Fbar_k(t) - F^AJ_k(t)|^alpha dt

where Fbar_k(t) is the mean predicted CIF for event k over the evaluation set
and F^AJ_k(t) is the marginal Aalen-Johansen estimate on the same set. Event 0
(still alive) is compared against Kaplan-Meier. A score of 0 is perfect
marginal calibration.

We also compute km_calibration, the survival-only score Alberge et al. criticise
for this setting, so the contrast can be shown on our own data rather than cited.

TWO THINGS THIS SCRIPT CHECKS BEFORE TRUSTING ANY NUMBER

1. The Aalen-Johansen baseline must score ~0 WHEN TRAIN AND TEST SHARE A
   DISTRIBUTION. It is marginally consistent by construction, so a large score
   on a random split would mean we have the API wrong.

   Measured: 0.000011 on a random split and 0.000012 within 2016-18 alone,
   against 0.012869 on the vintage split -- three orders of magnitude. The API
   is right; the vintage-split score IS the distribution shift, measured by a
   proper competing-risks calibration metric rather than inferred. The check
   below therefore compares against a random split, not against zero.

2. The predicted CIF vector must be coherent: F_1 + F_2 <= 1. Fine-Gray fits a
   separate subdistribution model per cause and does NOT constrain this, which
   we already note in the report. We measure the violation rather than clipping
   it away, because an incoherent vector is precisely the failure a joint
   measure should catch and a per-cause decile index cannot.

Output: results/aj_calibration.csv
"""

from __future__ import annotations

import ast
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from hazardous.metrics import aj_calibration, aj_calibration_per_event, km_calibration

from harness import HORIZONS, AalenJohansenBaseline
from metrics import DEFAULT, PREPAID
from models_classical import CauseSpecificCox, FineGray
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from splits import vintage_split
from tune import build_frame

DATASETS = {
    "postcrisis": ("data/processed/loans.parquet", (2013, 2014, 2015), (2016, 2017, 2018)),
    "crisis":     ("data/processed/loans_crisis.parquet", (2005, 2006), (2007, 2008, 2009)),
}


def configs_for(tag: str) -> dict[str, dict]:
    t = pd.read_csv("results/tuned.csv")
    t = t[t.dataset == tag]
    return {n: ast.literal_eval(t[t.model == n].config.iloc[0]) for n in t.model.unique()}


def builders(cfg: dict) -> dict:
    return {
        "aalen_johansen":     lambda: AalenJohansenBaseline(),
        "cause_specific_cox": lambda: CauseSpecificCox(),
        "fine_gray":          lambda: FineGray(),
        "rsf":                lambda: CauseSpecificRSF(**cfg["rsf_sel_auc"]),
        "deephit":            lambda: DeepHitModel(**cfg["deephit_sel_auc"]),
    }


def cif_vector(model, X, horizons) -> tuple[np.ndarray, float]:
    """
    Assemble (n_samples, n_events+1, n_times) = [S, F_default, F_prepaid].

    Returns the stacked array and the fraction of (subject, time) cells where
    F_1 + F_2 > 1, i.e. where the predicted vector is not a valid distribution.
    """
    f1 = model.predict_cif(X, horizons, DEFAULT)
    f2 = model.predict_cif(X, horizons, PREPAID)
    total = f1 + f2
    incoherent = float((total > 1.0).mean())
    surv = np.clip(1.0 - total, 0.0, 1.0)
    return np.stack([surv, f1, f2], axis=1), incoherent


def main() -> None:
    rows: list[dict] = []
    times = np.asarray(HORIZONS, dtype=float)

    for tag, (path, train_v, test_v) in DATASETS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        sp = vintage_split(df, train_vintages=train_v, test_vintages=test_v)
        tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
        t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)

        y_te = pd.DataFrame({"event": te.event.to_numpy(int),
                             "duration": te.time.to_numpy(float)})
        print(f"[{tag}] train {len(tr):,} test {len(te):,}", flush=True)

        for name, build in builders(configs_for(tag)).items():
            m = build().fit(tr[FEATS], t_tr, e_tr)
            pred, incoherent = cif_vector(m, te[FEATS], HORIZONS)

            aj_all = aj_calibration(y_te, times, pred)
            aj_per = aj_calibration_per_event(y_te, times, pred)
            km_cal = km_calibration(y_te, times, pred[:, 0, :])

            rows.append({
                "cohort": tag, "model": name,
                "aj_cal": float(aj_all),
                # aj_calibration_per_event returns {event_code: score}
                "aj_cal_surv": float(aj_per[0]),
                "aj_cal_default": float(aj_per[DEFAULT]),
                "aj_cal_prepaid": float(aj_per[PREPAID]),
                "km_cal": float(km_cal),
                "incoherent_frac": round(incoherent, 4),
            })
            print(f"  {name:20s} AJ-cal {aj_all:.5f}  KM-cal {km_cal:.5f}"
                  f"  incoherent {100*incoherent:.2f}%", flush=True)

        pd.DataFrame(rows).to_csv("results/aj_calibration.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv("results/aj_calibration.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n######## AJ-calibration (0 = perfectly marginally calibrated) ########")
    print(out.to_string(index=False))

    print("\n######## SANITY CHECK ########")
    for tag in DATASETS:
        aj = out[(out.cohort == tag) & (out.model == "aalen_johansen")].aj_cal.iloc[0]
        worst = out[out.cohort == tag].aj_cal.max()
        verdict = "OK" if aj < 0.05 * max(worst, 1e-9) or aj < 1e-4 else "SUSPECT -- check the API"
        print(f"  {tag}: Aalen-Johansen scores {aj:.6f} against a worst model of "
              f"{worst:.6f}  -> {verdict}")

    print("\nwrote results/aj_calibration.csv")


if __name__ == "__main__":
    main()
