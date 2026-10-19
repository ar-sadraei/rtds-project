"""
CR D-calibration: the distributional competing-risks calibration measure.

AJ-calibration (run_aj_calibration.py) is MARGINAL -- it asks whether the
population-average CIF is right. Our decile index is CONDITIONAL within risk
strata at one horizon. Neither is the distributional measure Alberge et al.
(2026) prove is proper, which asks whether individuals predicted to have
probability rho of an event actually experience it at frequency rho, across all
events and the whole time axis.

That measure is CR D-calibration. It is not in any released hazardous: the
implementation lives in the authors' open pull request

    soda-inria/hazardous#100, branch `dcal`, commit 26e64323

installed here from that pinned commit. It is unreleased code and the report
must say so.

API follows the authors' own example, examples/plot_05_calibration_dcr.py:
predictions are supplied on a time grid as (n_samples, n_events+1, n_times) with
component 0 the survival function, and the metric interpolates to each
individual's observed time. The last grid point stands in for infinity, so the
grid has to span the follow-up rather than stopping at 60 months like the
horizon grid used elsewhere in this project.

VALIDATION, as in run_aj_calibration.py: the Aalen-Johansen baseline is
marginally consistent by construction and should score near zero and pass the KS
test WHEN TRAIN AND TEST SHARE A DISTRIBUTION. We therefore score it on both a
vintage split and a random split. If it fails on the random split the API is
being misused; if it fails only on the vintage split, that is the distribution
shift, which is this project's subject.

Output: results/cr_dcalibration.csv
"""

from __future__ import annotations

import ast
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

warnings.filterwarnings("ignore")

from hazardous.metrics import d_cr_calibration, d_cr_calibration_ks_test
from hazardous.utils import make_time_grid

from harness import AalenJohansenBaseline
from metrics import DEFAULT, PREPAID
from models_classical import CauseSpecificCox, FineGray
from models_deephit import DeepHitModel
from models_rsf import CauseSpecificRSF
from splits import random_split, vintage_split
from tune import build_frame

N_GRID = 50          # time-grid resolution; last point stands in for infinity
N_TEST = 50_000      # subsample of the test set, for memory on a 3-D grid array

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


def cif_grid(model, X, times) -> np.ndarray:
    """(n_samples, 3, n_times) = [survival, F_default, F_prepaid]."""
    f1 = model.predict_cif(X, times, DEFAULT)
    f2 = model.predict_cif(X, times, PREPAID)
    surv = np.clip(1.0 - f1 - f2, 0.0, 1.0)
    return np.stack([surv, f1, f2], axis=1)


def evaluate(model, X_te, y_te, times, label) -> dict:
    pred = cif_grid(model, X_te, times)
    score = d_cr_calibration(y_test=y_te, y_test_pred=pred, times=times,
                             exact=False, alpha=2, reduction="mean")
    ks = d_cr_calibration_ks_test(y_te, event_of_interest=None, n_buckets=100,
                                  epsilon=1e-3, y_test_pred=pred, times=times)
    out = {"cr_dcal": float(score)}
    for ev, r in ks.items():
        out[f"ks_stat_e{ev}"] = round(float(r["statistic"]), 4)
        out[f"ks_p_e{ev}"] = float(r["pvalue"])
    passed = sum(1 for r in ks.values() if r["pvalue"] > 0.05)
    out["ks_passed"] = f"{passed}/{len(ks)}"
    print(f"  {label:20s} CR-Dcal {score:.5f}   KS passed {passed}/{len(ks)}", flush=True)
    return out


def main() -> None:
    rows: list[dict] = []
    rng = np.random.default_rng(0)

    for tag, (path, train_v, test_v) in DATASETS.items():
        df = pd.read_parquet(path)
        frame, FEATS = build_frame(df)
        cfg = configs_for(tag)

        sp = vintage_split(df, train_vintages=train_v, test_vintages=test_v)
        tr = frame.iloc[sp.train_idx]
        te_idx = sp.test_idx
        if len(te_idx) > N_TEST:
            te_idx = rng.choice(te_idx, N_TEST, replace=False)
        te = frame.iloc[te_idx]

        t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
        y_te = pd.DataFrame({"event": te.event.to_numpy(int),
                             "duration": te.time.to_numpy(float)})
        times = make_time_grid(event=e_tr, duration=t_tr, n_time_grid_steps=N_GRID)
        print(f"[{tag}] train {len(tr):,}  test {len(te):,}  "
              f"grid {len(times)} points, {times[0]:.0f}-{times[-1]:.0f} months", flush=True)

        for name, build in builders(cfg).items():
            m = build().fit(tr[FEATS], t_tr, e_tr)
            rows.append({"cohort": tag, "split": "vintage", "model": name,
                         **evaluate(m, te[FEATS], y_te, times, name)})

        # --- validation: the null model on a split with no vintage shift ---
        rs = random_split(df, seed=0, test_frac=0.5)
        rtr = frame.iloc[rs.train_idx]
        r_idx = rs.test_idx
        if len(r_idx) > N_TEST:
            r_idx = rng.choice(r_idx, N_TEST, replace=False)
        rte = frame.iloc[r_idx]
        rt, re_ = rtr.time.to_numpy(float), rtr.event.to_numpy(int)
        ry = pd.DataFrame({"event": rte.event.to_numpy(int),
                           "duration": rte.time.to_numpy(float)})
        rtimes = make_time_grid(event=re_, duration=rt, n_time_grid_steps=N_GRID)
        m = AalenJohansenBaseline().fit(rtr[FEATS], rt, re_)
        rows.append({"cohort": tag, "split": "random", "model": "aalen_johansen",
                     **evaluate(m, rte[FEATS], ry, rtimes, "aalen_johansen (random)")})

        pd.DataFrame(rows).to_csv("results/cr_dcalibration.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv("results/cr_dcalibration.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n######## CR D-calibration (0 = perfectly calibrated; KS p>0.05 = passes) ########")
    cols = ["cohort", "split", "model", "cr_dcal", "ks_passed"]
    print(out[cols].to_string(index=False))

    print("\n######## VALIDATION ########")
    for tag in DATASETS:
        v = out[(out.cohort == tag) & (out.split == "vintage") & (out.model == "aalen_johansen")]
        r = out[(out.cohort == tag) & (out.split == "random")]
        print(f"  {tag}: Aalen-Johansen scores {r.cr_dcal.iloc[0]:.6f} on a random split "
              f"(KS {r.ks_passed.iloc[0]}) and {v.cr_dcal.iloc[0]:.6f} on the vintage split "
              f"(KS {v.ks_passed.iloc[0]})")
    print("  A low random-split score means the API is used correctly; any excess on the")
    print("  vintage split is the distribution shift, not a bug.")
    print("\nwrote results/cr_dcalibration.csv")


if __name__ == "__main__":
    main()
