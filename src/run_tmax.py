"""
Is the alpha effect confounded by DeepHit's discretisation horizon?

DeepHit's softmax over (cause, time-bin) sums to one, so the model asserts that
every subject experiences an event by the last bin t_max. For a cohort that is
20.6% censored, that is false, and a calibration paper has argued it distorts the
distribution for long survivors regardless of the loss weight.

If t_max is the real driver, the alpha effect should shrink or move as t_max
grows. If alpha acts independently, the gap between alpha=0.2 and alpha=1.0
should stay roughly constant across horizons.

Grid: t_max in {72, 120, 160} months x alpha in {0.2, 1.0} x 2 seeds.
Evaluated at 60 months, which every horizon covers.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from harness import HORIZONS
from metrics import DEFAULT, auc_td, ici, kaplan_meier_censoring
from models_deephit import DeepHitModel
from splits import vintage_split
from tune import build_frame

HOR = 60.0
df = pd.read_parquet("data/processed/loans.parquet")
frame, FEATS = build_frame(df)
sp = vintage_split(df)
tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
t_tr,e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
t_te,e_te = te.time.to_numpy(float), te.event.to_numpy(int)
g_t,g_v = kaplan_meier_censoring(t_tr, e_tr)
j = int(np.where(HORIZONS==HOR)[0][0])

rows=[]
for tmax in (72.0, 120.0, 160.0):
    for alpha in (0.2, 1.0):
        for seed in (0,1):
            m = DeepHitModel(alpha=alpha, n_bins=30, epochs=50, seed=seed, t_max=tmax)
            m.fit(tr[FEATS], t_tr, e_tr)
            cif = m.predict_cif(te[FEATS], HORIZONS, DEFAULT)
            rows.append({"t_max":tmax, "alpha":alpha, "seed":seed,
                         "auc": auc_td(cif[:,j], t_te, e_te, DEFAULT, HOR, g_t, g_v),
                         "ici": ici(cif[:,j], t_te, e_te, DEFAULT, HOR)})
            print(f"  t_max={tmax:.0f} alpha={alpha} seed={seed} "
                  f"auc={rows[-1]['auc']:.4f} ici={rows[-1]['ici']:.4f}", flush=True)

r = pd.DataFrame(rows); r.to_csv("results/tmax_ablation.csv", index=False)
pd.set_option("display.width",200)
g = r.groupby(["t_max","alpha"])[["auc","ici"]].mean().round(4)
print("\n######## mean over 2 seeds ########"); print(g.to_string())
print("\n######## alpha effect (alpha=1.0 minus alpha=0.2) at each t_max ########")
for tmax in (72.0,120.0,160.0):
    a = g.loc[(tmax,1.0)]; b = g.loc[(tmax,0.2)]
    print(f"  t_max={tmax:.0f}:  dAUC {a.auc-b.auc:+.4f}   dICI {a.ici-b.ici:+.4f}")
print("\nIf the alpha effect is roughly constant across t_max, t_max is not the driver.")
