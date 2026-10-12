"""Audit: are the IPCW control weights in auc_td correct?

Blanche et al. (2013) weight a control that had a COMPETING event at T<=t by
1/G(T-), not by 1/G(t): such a subject is known to be a control only because
they were uncensored up to T. metrics.auc_td gives every control 1/G(t).
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import DEFAULT, PREPAID, CENSORED, _step_eval, auc_td, kaplan_meier_censoring
from models_classical import CauseSpecificCox
from models_rsf import CauseSpecificRSF
from splits import vintage_split
from tune import build_frame
from harness import HORIZONS

def auc_td_blanche(risk, time, event, cause, horizon, g_t, g_v):
    """Same as metrics.auc_td but with per-control IPCW weights."""
    is_case = (time <= horizon) & (event == cause)
    ctrl_free = time > horizon
    ctrl_comp = (time <= horizon) & (event != cause) & (event != CENSORED)
    is_ctrl = ctrl_free | ctrl_comp
    if is_case.sum() == 0 or is_ctrl.sum() == 0:
        return float("nan")

    g_at_h = max(_step_eval(g_t, g_v, np.array([horizon]))[0], 1e-8)
    w_case = 1.0 / np.clip(_step_eval(g_t, g_v, time[is_case] - 1e-9), 1e-8, None)

    w_all = np.zeros_like(time, dtype=float)
    w_all[ctrl_free] = 1.0 / g_at_h
    w_all[ctrl_comp] = 1.0 / np.clip(
        _step_eval(g_t, g_v, time[ctrl_comp] - 1e-9), 1e-8, None)
    wk = w_all[is_ctrl]

    rc, wc = risk[is_case], w_case
    rk = risk[is_ctrl]
    order = np.argsort(rk, kind="mergesort")
    rk_s, wk_s = rk[order], wk[order]
    cum = np.concatenate(([0.0], np.cumsum(wk_s)))
    lo = np.searchsorted(rk_s, rc, side="left")
    hi = np.searchsorted(rk_s, rc, side="right")
    num = float(np.sum(wc * (cum[lo] + 0.5 * (cum[hi] - cum[lo]))))
    den = float(np.sum(wc) * np.sum(wk))
    return num / den if den > 0 else float("nan")

DATASETS = {
    "postcrisis": ("data/processed/loans.parquet", (2013,2014,2015), (2016,2017,2018)),
    "crisis":     ("data/processed/loans_crisis.parquet", (2005,2006), (2007,2008,2009)),
}
j = int(np.where(HORIZONS == 60.0)[0][0])
rows = []
for tag,(path,tr_v,te_v) in DATASETS.items():
    df = pd.read_parquet(path); frame, FEATS = build_frame(df)
    sp = vintage_split(df, train_vintages=tr_v, test_vintages=te_v)
    tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
    t_tr,e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
    t_te,e_te = te.time.to_numpy(float), te.event.to_numpy(int)
    g_t,g_v = kaplan_meier_censoring(t_tr,e_tr)
    print(f"[{tag}] G(60)={_step_eval(g_t,g_v,np.array([60.0]))[0]:.4f} "
          f"censored={np.mean(e_te==0):.3f}", flush=True)
    for name, m in (("Cox", CauseSpecificCox()), ("RSF", CauseSpecificRSF())):
        m.fit(tr[FEATS], t_tr, e_tr)
        for cause,cn in ((DEFAULT,"default"),(PREPAID,"prepaid")):
            cif = m.predict_cif(te[FEATS], HORIZONS, cause)[:, j]
            a_old = auc_td(cif, t_te, e_te, cause, 60.0, g_t, g_v)
            a_new = auc_td_blanche(cif, t_te, e_te, cause, 60.0, g_t, g_v)
            rows.append(dict(cohort=tag, model=name, cause=cn,
                             as_coded=round(a_old,4), blanche=round(a_new,4),
                             diff=round(a_new-a_old,4)))
            print("  ", rows[-1], flush=True)
pd.DataFrame(rows).to_csv(
  "/private/tmp/claude-501/-Users-rudevo-Library-Mobile-Documents-com-apple-CloudDocs-UNI-Semester-3-RTDS-Literature-Review/828be3ce-db2b-4ab1-ae30-63d9b0cc54b8/scratchpad/auc_audit.csv", index=False)
