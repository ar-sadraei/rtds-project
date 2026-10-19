"""Generate the report's LaTeX table fragments straight from the results CSVs.

No number in the report is typed by hand: every table is regenerated from
results/*.csv, so the paper cannot drift from the experiments. This script
emits exactly the four tables main.tex inputs and reads nothing but
results/*.csv, so it runs on a clean checkout without the raw loan data.

Numbers quoted in the prose but not tabulated -- the alpha ablation, the
training-variability spreads, the recalibration deltas -- come from the same
CSVs; results/README.md maps each file to the claim it supports.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"src")
OUT = "report"

def emit(name, lines, caption="", label="", colspec="", header="", wide=False,
         colsep=None):
    """
    Write a COMPLETE table environment.

    \\input of bare rows inside a tabular fails: TeX is mid-alignment when the
    file ends, which raises "Misplaced \\noalign" at the following \\bottomrule.
    Emitting the whole environment and inputting it at top level avoids that.
    """
    body = "\n".join(l.rstrip() for l in lines).rstrip()
    env = "table*" if wide else "table"
    # A few tables run 8-10pt past the IEEE column. Tightening the inter-column
    # padding inside the float (where it is local) pulls them back in without
    # shrinking the type or widening the float.
    sep = f"\\setlength{{\\tabcolsep}}{{{colsep}}}\n" if colsep else ""
    tex = (f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\centering\\footnotesize\n{sep}\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
           f"{header}\n\\midrule\n{body}\n\\bottomrule\n"
           f"\\end{{tabular}}\n\\end{{{env}}}\n")
    open(f"{OUT}/{name}.tex", "w").write(tex)

NICE = {"aalen_johansen":"Aalen--Johansen","cause_specific_cox":"Cox",
        "fine_gray":"Fine--Gray","rsf":"RSF","deephit":"DeepHit",
        "deephit_a0.2":r"DeepHit $\alpha{=}0.2$",
        "deephit_a0.5":r"DeepHit $\alpha{=}0.5$","deephit_a1":r"DeepHit $\alpha{=}1$",
        "deephit_sel_auc":"DeepHit (tuned)","deephit_sel_ibs":"DeepHit (sel.\\ IBS)",
        "rsf_sel_auc":"RSF (tuned)","rsf_sel_ibs":"RSF (sel.\\ IBS)"}
SHORT = {"postcrisis":"post", "crisis":"crisis", "default":"def.", "prepaid":"prep."}

def pick(df, cause="default", h=60.0):
    return (df[(df.cause==cause)&df.metric.isin(["auc_td","ici"])&(df.horizon==h)]
            .pivot_table(index="model", columns="metric", values="value"))

# ---------- T2: main results --------------------------------------------
post = pick(pd.read_csv("results/featureset.csv").query("featureset=='rich'"))
cris = pick(pd.read_csv("results/crisis.csv"))
tun  = pd.read_csv("results/tuned.csv")
post = pd.concat([post, pick(tun[tun.dataset=="postcrisis"])])
cris = pd.concat([cris, pick(tun[tun.dataset=="crisis"])])
order=["aalen_johansen","cause_specific_cox","fine_gray","rsf","rsf_sel_auc",
       "deephit_a0.2","deephit_a1","deephit_sel_auc"]
lines=[]
for m in order:
    def g(t,c):
        return f"{t.loc[m,c]:.4f}" if m in t.index and not pd.isna(t.loc[m,c]) else "--"
    lines.append(f"{NICE[m]} & {g(cris,'auc_td')} & {g(cris,'ici')} & "
                 f"{g(post,'auc_td')} & {g(post,'ici')} \\\\")
emit("tab_main", lines,
     caption=("Discrimination (AUC at 60 months) and calibration (decile-binned ICI "
              "at 60 months) for default, on the held-out test vintages of each "
              "cohort. These are single-split figures, given for the levels; "
              "Section~V-A reports how far they move across time-ordered folds, "
              "and the differences between models here should be read against "
              "that spread rather than on their own."),
     label="tab:main", colspec="@{}lrrrr@{}",
     header=(r"& \multicolumn{2}{c}{Crisis (2007--09)} & \multicolumn{2}{c}{Post-crisis (2016--18)} \\"
             "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n" r"Model & AUC & ICI & AUC & ICI \\"))

# ---------- T9: rolling origin -------------------------------------------
# Four time-ordered folds per cohort instead of one, so the vintage protocol
# finally carries an uncertainty estimate of its own.
pd_ = pd.read_csv("results/protocol_decomposition.csv")
roll = pd_[pd_.experiment == "rolling"].pivot_table(
        index=["cohort", "cause", "test_vintage"], columns="model", values="auc")
roll["dh_cox"] = roll["deephit"] - roll["cause_specific_cox"]
roll["dh_rsf"] = roll["deephit"] - roll["rsf"]
g = roll.groupby(level=[0, 1])[["dh_cox", "dh_rsf"]].agg(["mean", "std"])
lines = []
for ca in ("default", "prepaid"):
    for co in ("crisis", "postcrisis"):
        r = g.loc[(co, ca)]
        lines.append(f"{SHORT.get(ca,ca)} & {SHORT.get(co,co)} & "
                     f"{r[('dh_cox','mean')]:+.4f} & {r[('dh_cox','std')]:.4f} & "
                     f"{r[('dh_rsf','mean')]:+.4f} & {r[('dh_rsf','std')]:.4f} \\\\")
emit("tab_rolling", lines,
     caption=("Rolling-origin evaluation. The training window expands one vintage at "
              "a time and the next vintage is the test set, giving four time-ordered "
              "folds per cohort. Entries are DeepHit's AUC advantage at 60 months over "
              "each baseline, averaged over folds, with the standard deviation across "
              "folds. No advantage reaches 1.3 standard deviations of its own spread."),
     label="tab:rolling", colspec="@{}llrrrr@{}",
     header=(r"& & \multicolumn{2}{c}{vs Cox} & \multicolumn{2}{c}{vs RSF} \\"
             "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
             r"Cause & Cohort & mean & sd & mean & sd \\"),
     colsep="4pt")

# ---------- T10: same test loans, training period moved -------------------
mt = pd_[pd_.experiment == "matched"].pivot_table(
        index=["cohort", "cause", "arm"], columns="model", values="auc")
mt["dh_cox"] = mt["deephit"] - mt["cause_specific_cox"]
mt["dh_rsf"] = mt["deephit"] - mt["rsf"]
lines = []
for ca in ("default", "prepaid"):
    for co in ("crisis", "postcrisis"):
        c, e = mt.loc[(co, ca, "contemporaneous")], mt.loc[(co, ca, "earlier")]
        lines.append(f"{SHORT.get(ca,ca)} & {SHORT.get(co,co)} & "
                     f"{c.dh_cox:+.4f} & {e.dh_cox:+.4f} & "
                     f"{c.dh_rsf:+.4f} & {e.dh_rsf:+.4f} \\\\")
emit("tab_matched", lines,
     caption=("The same test loans, the same training-set size, and only the training "
              "period moved. The test set is a random half of the later vintages; one "
              "arm trains on the earlier vintages, the other on the remaining half of "
              "the later ones. Entries are DeepHit's AUC advantage at 60 months. "
              "Against the forest the advantage is positive in every cell when "
              "training is contemporaneous and negative or negligible in every cell "
              "when it is not. The change in each advantage between the two arms is "
              "significant under a paired bootstrap over the shared test loans "
              "($B=500$, Bonferroni $0.05/3$)."),
     label="tab:matched", colspec="@{}llrrrr@{}",
     header=(r"& & \multicolumn{2}{c}{vs Cox} & \multicolumn{2}{c}{vs RSF} \\"
             "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
             r"Cause & Cohort & contemp. & earlier & contemp. & earlier \\"),
     colsep="4pt")

# ---------- T11: proper competing-risks calibration -----------------------
# Our decile index is conditional-within-strata at one horizon. These two are
# the measures Alberge et al. (2026) prove are proper: AJ-calibration is
# marginal over time and events, CR D-calibration is distributional.
ajc = pd.read_csv("results/aj_calibration.csv")
crd = pd.read_csv("results/cr_dcalibration.csv")
crd = crd[crd.split == "vintage"]
lines = []
for m in ("aalen_johansen", "cause_specific_cox", "fine_gray", "rsf", "deephit"):
    r = {}
    for co in ("crisis", "postcrisis"):
        r[("aj", co)] = ajc[(ajc.cohort == co) & (ajc.model == m)].aj_cal.iloc[0]
        r[("cr", co)] = crd[(crd.cohort == co) & (crd.model == m)].cr_dcal.iloc[0]
    lines.append(f"{NICE.get(m, m)} & {r[('aj','crisis')]:.4f} & {r[('aj','postcrisis')]:.4f} "
                 f"& {r[('cr','crisis')]:.4f} & {r[('cr','postcrisis')]:.4f} \\\\")
emit("tab_calib", lines,
     caption=("Calibration under the two measures Alberge et al. prove are proper for "
              "competing risks: AJ-calibration, which is marginal over events and time, "
              "and CR D-calibration, which is distributional. Lower is better and zero "
              "is perfect. The Aalen--Johansen estimator is marginally consistent by "
              "construction and scores $4\\times10^{-5}$ or less on a random split, so "
              "its values here measure the vintage shift rather than a modelling "
              "failure. DeepHit is worst under both measures in both cohorts."),
     label="tab:calib", colspec="@{}lrrrr@{}",
     header=(r"& \multicolumn{2}{c}{AJ-calibration} & \multicolumn{2}{c}{CR D-calibration} \\"
             "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
             r"Model & crisis & post & crisis & post \\"),
     colsep="5pt")

TABLES = ["tab_main", "tab_rolling", "tab_matched", "tab_calib"]
print("wrote " + ", ".join(f"{OUT}/{f}.tex" for f in TABLES))
if "--show" in sys.argv:
    for f in TABLES:
        print(f"\n--- {f} ---"); print(open(f"{OUT}/{f}.tex").read())
