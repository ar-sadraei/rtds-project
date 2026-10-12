"""Calibration curves: predicted vs observed cumulative incidence at 60 months.

Observed is the Aalen-Johansen estimate within each decile of predicted risk, so
competing events and censoring are both handled. A perfectly calibrated model lies
on the diagonal; above it the model under-predicts, below it over-predicts.

Two panels: default and prepayment, crisis cohort. The second panel is the point --
the competing event is where calibration falls apart, and it is the event the credit
literature most often censors.
"""
import ast, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"src")
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
sp = vintage_split(df, train_vintages=(2005,2006), test_vintages=(2007,2008,2009))
tr, te = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
t_tr,e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
t_te,e_te = te.time.to_numpy(float), te.event.to_numpy(int)

cfgs = pd.read_csv("results/tuned.csv").query("dataset=='crisis'")
cfg = lambda n: ast.literal_eval(cfgs[cfgs.model==n].config.iloc[0])

models = [("Cause-specific Cox", CauseSpecificCox(),            "o", "-",  "#1b4965"),
          ("RSF (tuned)",        CauseSpecificRSF(**cfg("rsf_sel_auc")), "s", "--", "#e07a5f"),
          ("DeepHit (tuned)",    DeepHitModel(**cfg("deephit_sel_auc")), "^", ":",  "#7b2d8e")]

fitted=[]
for name,m,mk,ls,col in models:
    print("fitting",name,flush=True); m.fit(tr[FEATS], t_tr, e_tr); fitted.append((name,m,mk,ls,col))

plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":7,
                     "xtick.labelsize":7,"ytick.labelsize":7,"figure.dpi":200})
fig, axes = plt.subplots(1,2, figsize=(7.0,3.0))
for ax,(cause,label) in zip(axes, [(DEFAULT,"Default"),(PREPAID,"Prepayment")]):
    hi=0
    for name,m,mk,ls,col in fitted:
        cif = m.predict_cif(te[FEATS], HORIZONS, cause)[:, int(np.where(HORIZONS==HOR)[0][0])]
        tab = calibration_table(cif, t_te, e_te, cause, HOR, n_bins=10)
        v = ici(cif, t_te, e_te, cause, HOR)
        ax.plot(tab.predicted, tab.observed, marker=mk, linestyle=ls, color=col,
                markersize=3.5, linewidth=1.1, label=f"{name} (ICI {v:.3f})")
        hi = max(hi, tab.predicted.max(), tab.observed.max())
    lim = hi*1.08
    ax.plot([0,lim],[0,lim], color="0.55", linewidth=0.8, dashes=(4,3), zorder=0)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.set_aspect("equal")
    ax.set_xlabel("Predicted cumulative incidence")
    ax.set_title(f"{label} at 60 months", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Observed (Aalen\u2013Johansen)")
fig.tight_layout(pad=0.4)
fig.savefig("report/fig_calibration.pdf", bbox_inches="tight")
print("wrote report/fig_calibration.pdf")
