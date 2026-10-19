r"""Calibration curves: predicted vs observed cumulative incidence at 60 months.

Plots results/calibration_curves.csv, which dump_calibration_curves.py writes.
Fitting and drawing are separate so the figure can be resized without refitting
three models on 150,000 loans.

Observed is the Aalen-Johansen estimate within each decile of predicted risk, so
competing events and censoring are both handled. A perfectly calibrated model
lies on the diagonal; above it the model under-predicts.

Two panels: default and prepayment, crisis cohort. The second panel is the point
-- the competing event is where calibration falls apart, and it is the event the
credit literature most often censors.

SIZING. The figure is drawn at the width it is rendered at (one IEEE column,
3.45in) and included with no scaling, so the point sizes set here are the point
sizes on the page. Scaling a figure in LaTeX scales the type with it, which is
why this script owns the size rather than \includegraphics.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTFILE = "report/fig_calibration.pdf"
STYLE = {"Cause-specific Cox": ("Cox", "o", "-", "#1b4965"),
         "RSF (tuned)":        ("RSF", "s", "--", "#e07a5f"),
         "DeepHit (tuned)":    ("DeepHit", "^", ":", "#7b2d8e")}

cur = pd.read_csv("results/calibration_curves.csv")

plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "legend.fontsize": 5,
                     "xtick.labelsize": 5, "ytick.labelsize": 5, "figure.dpi": 300,
                     "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                     "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))

for ax, (cause, label) in zip(axes, [("default", "Default"), ("prepaid", "Prepayment")]):
    hi = 0.0
    for name, (short, mk, ls, col) in STYLE.items():
        g = cur[(cur.model == name) & (cur.cause == cause)].sort_values("decile")
        ax.plot(g.predicted, g.observed, marker=mk, linestyle=ls, color=col,
                markersize=1.8, linewidth=0.8,
                label=f"{short} {g.ici.iloc[0]:.3f}")
        hi = max(hi, g.predicted.max(), g.observed.max())
    lim = hi * 1.08
    ax.plot([0, lim], [0, lim], color="0.55", linewidth=0.5, dashes=(3, 2), zorder=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Predicted CIF")
    ax.set_title(f"{label}, 60 months", fontsize=6)
    # Every curve lies above the diagonal, so the lower-right wedge is the one
    # empty region of the panel. The legend goes there, unframed, and carries
    # each model's calibration index as its label.
    ax.legend(loc="lower right", frameon=False, handlelength=1.4,
              borderpad=0.1, labelspacing=0.25, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Observed (Aalen–Johansen)")
fig.tight_layout(pad=0.2, w_pad=0.6)
fig.savefig(OUTFILE, bbox_inches="tight")
print("wrote", OUTFILE)
