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

# The poster scales this figure to a ~240mm column on an 841mm sheet, about
# 1.35x, next to 39pt body text. At the report's 8pt the axis labels land near
# 11pt effective and cannot be read from a metre away.
#
# Scaling the canvas with the type does nothing: what reaches the eye is the
# ratio of type size to figure WIDTH, since the figure is then stretched to the
# column. So the poster build enlarges the type and leaves the canvas alone.
POSTER = "--poster" in sys.argv
S = 2.2 if POSTER else 1.0          # type, lines and markers
FIGSIZE = (7.0, 4.0) if POSTER else (7.0, 3.0)   # room for the larger type
# Bigger type only fits if the strings get shorter, so the poster drops the
# words a reader standing in front of it does not need.
SHORTNAME = {"Cause-specific Cox":"Cox", "RSF (tuned)":"RSF",
             "DeepHit (tuned)":"DeepHit"}
OUTFILE = ("report/fig_calibration_poster.pdf" if POSTER
           else "report/fig_calibration.pdf")

plt.rcParams.update({"font.size":8*S,"axes.labelsize":8*S,"legend.fontsize":7*S,
                     "xtick.labelsize":7*S,"ytick.labelsize":7*S,"figure.dpi":200,
                     "axes.linewidth":0.8*S,
                     "xtick.major.width":0.8*S,"ytick.major.width":0.8*S,
                     "xtick.major.size":3.5*S,"ytick.major.size":3.5*S})
fig, axes = plt.subplots(1,2, figsize=FIGSIZE)
for ax,(cause,label) in zip(axes, [(DEFAULT,"Default"),(PREPAID,"Prepayment")]):
    hi=0
    for name,m,mk,ls,col in fitted:
        cif = m.predict_cif(te[FEATS], HORIZONS, cause)[:, int(np.where(HORIZONS==HOR)[0][0])]
        tab = calibration_table(cif, t_te, e_te, cause, HOR, n_bins=10)
        v = ici(cif, t_te, e_te, cause, HOR)
        ax.plot(tab.predicted, tab.observed, marker=mk, linestyle=ls, color=col,
                markersize=3.5*S, linewidth=1.1*S, label=(f"{SHORTNAME[name]} ({v:.3f})" if POSTER
                       else f"{name} (ICI {v:.3f})"))
        hi = max(hi, tab.predicted.max(), tab.observed.max())
    lim = hi*1.08
    ax.plot([0,lim],[0,lim], color="0.55", linewidth=0.8*S, dashes=(4,3), zorder=0)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.set_aspect("equal")
    ax.set_xlabel("Predicted CIF" if POSTER else "Predicted cumulative incidence")
    ax.set_title(label if POSTER else f"{label} at 60 months", fontsize=8.5*S)
    # At upper left the legend sits on top of the curves and the diagonal runs
    # through its text. Every curve lies above the diagonal, so the lower-right
    # wedge is the one empty region of the panel: put it there, unframed.
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Observed" if POSTER else "Observed (Aalen\u2013Johansen)")
fig.tight_layout(pad=0.4)
fig.savefig(OUTFILE, bbox_inches="tight")
print("wrote", OUTFILE)
