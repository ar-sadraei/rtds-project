"""Fit once, dump the decile curves, so the figure can be redrawn without refitting."""
import ast, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from metrics import DEFAULT, PREPAID, calibration_table, ici
from models_classical import CauseSpecificCox
from models_rsf import CauseSpecificRSF
from models_deephit import DeepHitModel
from splits import vintage_split
from tune import build_frame
from harness import HORIZONS

HOR = 60.0
df = pd.read_parquet("data/processed/loans_crisis.parquet")
frame, FEATS = build_frame(df)
sp = vintage_split(df, train_vintages=(2005, 2006), test_vintages=(2007, 2008, 2009))
tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)

cfgs = pd.read_csv("results/tuned.csv").query("dataset=='crisis'")
cfg = lambda n: ast.literal_eval(cfgs[cfgs.model == n].config.iloc[0])

models = [("Cause-specific Cox", CauseSpecificCox()),
          ("RSF (tuned)",        CauseSpecificRSF(**cfg("rsf_sel_auc"))),
          ("DeepHit (tuned)",    DeepHitModel(**cfg("deephit_sel_auc")))]

j = int(np.where(HORIZONS == HOR)[0][0])
rows = []
for name, m in models:
    print("fitting", name, flush=True)
    m.fit(tr[FEATS], t_tr, e_tr)
    for cause, label in ((DEFAULT, "default"), (PREPAID, "prepaid")):
        cif = m.predict_cif(te[FEATS], HORIZONS, cause)[:, j]
        tab = calibration_table(cif, t_te, e_te, cause, HOR, n_bins=10)
        v = ici(cif, t_te, e_te, cause, HOR)
        for k, (pr, ob) in enumerate(zip(tab.predicted, tab.observed), 1):
            rows.append({"model": name, "cause": label, "decile": k,
                         "predicted": float(pr), "observed": float(ob), "ici": float(v)})
pd.DataFrame(rows).to_csv("results/calibration_curves.csv", index=False)
print("wrote results/calibration_curves.csv")
