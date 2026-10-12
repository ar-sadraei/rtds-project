"""
Controlled ablation of DeepHit's loss weight alpha.

Table V in the first draft pooled 40 tuning configurations by alpha. That is a
MARGINAL, not an ablation: alpha varied together with depth, dropout, learning
rate, batch size and bin count, so a configuration effect could masquerade as an
alpha effect. Here everything except alpha is held fixed.

Fixed: n_bins=30, epochs=50, lr=1e-3, batch=512, dropout=0.1, sigma=0.1.
Varied: alpha in {0.1, 0.2, 0.35, 0.5, 0.75, 1.0}, 5 seeds, both cohorts.
Evaluated at 60 months on the held-out test vintages, both causes.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from harness import HORIZONS
from metrics import DEFAULT, PREPAID, auc_td, ici, kaplan_meier_censoring
from models_deephit import DeepHitModel
from splits import vintage_split
from tune import build_frame

HOR, SEEDS = 60.0, 5
ALPHAS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
FIXED = dict(n_bins=30, epochs=50, lr=1e-3, batch_size=512, dropout=0.1, sigma=0.1)
COHORTS = {"postcrisis": ("data/processed/loans.parquet", (2013,2014,2015), (2016,2017,2018)),
           "crisis":     ("data/processed/loans_crisis.parquet", (2005,2006), (2007,2008,2009))}

rows=[]
for tag,(path,trv,tev) in COHORTS.items():
    df = pd.read_parquet(path); frame, FEATS = build_frame(df)
    sp = vintage_split(df, train_vintages=trv, test_vintages=tev)
    tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
    t_tr,e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
    t_te,e_te = te.time.to_numpy(float), te.event.to_numpy(int)
    g_t,g_v = kaplan_meier_censoring(t_tr, e_tr)
    j = int(np.where(HORIZONS==HOR)[0][0])
    for a in ALPHAS:
        for seed in range(SEEDS):
            m = DeepHitModel(alpha=a, seed=seed, **FIXED).fit(tr[FEATS], t_tr, e_tr)
            for cause,cn in ((DEFAULT,"default"),(PREPAID,"prepaid")):
                cif = m.predict_cif(te[FEATS], HORIZONS, cause)
                rows.append({"cohort":tag,"alpha":a,"seed":seed,"cause":cn,
                    "auc":auc_td(cif[:,j],t_te,e_te,cause,HOR,g_t,g_v),
                    "ici":ici(cif[:,j],t_te,e_te,cause,HOR)})
        d=[r for r in rows if r["cohort"]==tag and r["alpha"]==a and r["cause"]=="default"]
        print(f"  {tag} a={a}: auc {np.mean([x['auc'] for x in d]):.4f} "
              f"+-{np.std([x['auc'] for x in d]):.4f}  "
              f"ici {np.mean([x['ici'] for x in d]):.4f}", flush=True)

r=pd.DataFrame(rows); r.to_csv("results/alpha_ablation.csv", index=False)
pd.set_option("display.width",200)
for cause in ["default","prepaid"]:
    print(f"\n######## {cause} ########")
    g=r[r.cause==cause].groupby(["cohort","alpha"])[["auc","ici"]].agg(["mean","std"]).round(4)
    print(g.to_string())
