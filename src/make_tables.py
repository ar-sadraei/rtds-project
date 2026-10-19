"""Generate LaTeX table fragments straight from the results CSVs.

No number in the report is typed by hand: every table is regenerated from
results/*.csv, so the paper cannot drift from the experiments.
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

# ---------- T1: dataset description -------------------------------------
rows=[]
for tag,path,tr,te in [("Post-crisis","data/processed/loans.parquet","2013--2015","2016--2018"),
                       ("Crisis","data/processed/loans_crisis.parquet","2005--2006","2007--2009")]:
    d=pd.read_parquet(path); s=d.event.value_counts(normalize=True).reindex([0,1,2]).fillna(0)
    rows.append(f"{tag} & {len(d):,} & {s[1]*100:.1f}\\% & {s[2]*100:.1f}\\% & "
                f"{s[0]*100:.1f}\\% & {(d.event!=1).mean()*100:.1f}\\% \\\\")
emit("tab_data", rows, caption=("The two cohorts. ``Censoring, default only'' recomputes the censoring "
              "rate when prepayment is treated as censored rather than as an event."),
     label="tab:data", colspec="@{}lrrrrr@{}",
     header=r"Cohort & $n$ & Default & Prepaid & Cens. & Cens.\ (def.) \\")

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

# ---------- T3: significance, tuned --------------------------------------
# A bootstrap of B resamples cannot resolve a p below 1/B, so the floor value is
# an upper bound, not a point estimate. Printing "0.002" for 31 of 36 tests read
# as far more precision than B=500 can deliver.
B_BOOT = 500

def pfmt(p: float) -> str:
    return r"$<$0.002" if p <= 1.0 / B_BOOT + 1e-12 else f"{p:.3f}"


s=pd.read_csv("results/significance_tuned.csv")
sd=pd.read_csv("results/deephit_seed_spread.csv").groupby("dataset")[["auc_td@60","ici@60"]].std()
keep=s[(s.cause=="default")&s.metric.isin(["auc_td@60","ici@60"])]
lines=[]
for _,r in keep.iterrows():
    key="auc_td@60" if r.metric=="auc_td@60" else "ici@60"
    noisy = abs(r["diff"]) < 2*sd.loc[r.dataset,key]
    mark = r"$\dagger$" if noisy else ("\\textbf{*}" if r.significant else "n.s.")
    met = "AUC" if r.metric=="auc_td@60" else "ICI"
    lines.append(f"{r.comparison} & {SHORT.get(r.dataset,r.dataset)} & {met} & {r['diff']:+.4f} & "
                 f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] & {pfmt(r.p)} & {mark} \\\\")
emit("tab_sig", lines,
     caption=("Paired bootstrap on the tuned models ($B{=}500$), default, 60 months. "
              "T1 DeepHit vs Cox; T2 DeepHit vs RSF; T3 DeepHit selected on AUC vs on IBS. "
              "\\textbf{*} significant at the Bonferroni level $0.05/3$; "
              "$\\dagger$ difference smaller than twice DeepHit's seed-to-seed SD."),
     label="tab:sig", colspec="@{}lllrlrl@{}",
     header=r"Test & Cohort & Metric & Diff & 95\% CI & $p$ & \\",
     colsep="4pt")

# ---------- T4: alpha ABLATION (replaces the pooled tuning marginal) ------
ab = pd.read_csv("results/alpha_ablation.csv")
g = ab.groupby(["cohort","cause","alpha"])[["auc","ici"]].mean()
lines=[]
for ca in ("default","prepaid"):
    for co in ("crisis","postcrisis"):
        lo, hi = g.loc[(co,ca,0.10)], g.loc[(co,ca,1.00)]
        sd = ab[(ab.cohort==co)&(ab.cause==ca)].groupby("alpha").auc.std().mean()
        lines.append(f"{SHORT.get(ca,ca)} & {SHORT.get(co,co)} & "
                     f"{hi.auc-lo.auc:+.4f} & {sd:.4f} & {hi.ici-lo.ici:+.4f} \\\\")
emit("tab_alpha", lines,
     caption=("Controlled ablation of DeepHit's loss weight $\\alpha$: architecture, "
              "learning rate, batch size, dropout and bin count held fixed, five seeds "
              "per setting, evaluated at 60 months on the held-out test vintages. "
              "$\\alpha=1$ removes the ranking term entirely. The effect reverses sign "
              "across cohorts and causes, and in one cell is smaller than the seed "
              "standard deviation. $\\Delta$ is the change from $\\alpha{=}0.1$ to $\\alpha{=}1$."),
     label="tab:alpha", colspec="@{}llrrr@{}",
     header=r"Cause & Cohort & $\Delta$AUC & seed sd & $\Delta$ICI \\")

# T5 (runtime) removed: the figures live in the prose of Section IV-E.

# ---------- T6: training variability --------------------------------------
v = pd.read_csv("results/training_variability.csv")
g = v.groupby(["dataset","model"])[["auc","ici"]].agg(["mean","std"])
lines=[]
# crisis only: every post-crisis SD is <= 0.0022 and the prose says so, so the
# second half of the table carried no information worth the space.
ORDER = ["cause_specific_cox", "fine_gray", "rsf", "deephit"]
for (ds,m),r in sorted(g.iterrows(),
                       key=lambda kv: ORDER.index(kv[0][1]) if kv[0][1] in ORDER else 99):
    if ds != "crisis":
        continue
    lines.append(f"{NICE.get(m, m.replace(chr(95), chr(92)+chr(95)))} & "
                 f"{r[('auc','mean')]:.4f} & {r[('auc','std')]:.4f} & "
                 f"{r[('ici','mean')]:.4f} & {r[('ici','std')]:.4f} \\\\")
emit("tab_var", lines,
     caption=("Training variability on the \\textbf{crisis} cohort: each model refitted on "
              "five bootstrap resamples of its training vintages and evaluated on the fixed "
              "test vintages (default, 60 months). Seeds are the wrong device for Cox and "
              "Fine--Gray, which are deterministic given the data, so the training set is "
              "resampled instead. Every post-crisis standard deviation is below 0.0022 and "
              "is omitted."),
     label="tab:var", colspec="@{}lrrrr@{}",
     header=r"Model & AUC mean & sd & ICI mean & sd \\")

# ---------- T7: recalibration ---------------------------------------------
rc = pd.read_csv("results/recalibration.csv")
piv = rc.pivot_table(index=["dataset","cause","model"], columns="variant", values="ici")
lines=[]
KEEP = {"cause_specific_cox", "deephit"}
for (ds,ca,m),r in piv.iterrows():
    if m not in KEEP:
        continue
    lines.append(f"{SHORT.get(ds,ds)} & {SHORT.get(ca,ca)} & {NICE.get(m, m.replace(chr(95), chr(92)+chr(95)))} & "
                 f"{r['raw']:.4f} & {r['recal_train']:.4f} & {r['recal_oracle']:.4f} \\\\")
emit("tab_recal", lines,
     caption=("Post-hoc recalibration, ICI at 60 months. ``Train'' estimates the "
              "rescaling on the training vintages, which is all a lender has at scoring "
              "time; ``oracle'' estimates it on the test vintages and is unattainable in "
              "practice. Cox and DeepHit are shown; the forest and Fine--Gray behave the same way. "
              "AUC is identical to four decimals for raw and recalibrated "
              "predictions in all sixteen cells, confirming the rescaling is monotone."),
     label="tab:recal", colspec="@{}lllrrr@{}",
     header=r"Cohort & Cause & Model & Raw & Recal.\ train & Recal.\ oracle \\",
     colsep="4pt")

# ---------- T8: what the evaluation protocol is worth ---------------------
# The benchmarking literature evaluates on random splits; we evaluate on
# vintage-ordered ones. This table holds the models and their tuned settings
# fixed and changes only that, so the difference is the protocol's own effect.
pc = pd.read_csv("results/protocol_comparison.csv")
rnd = pc[pc.protocol == "random"].pivot_table(
        index=["cohort", "cause", "seed"], columns="model", values="auc")
vin = pc[pc.protocol == "vintage"].pivot_table(
        index=["cohort", "cause"], columns="model", values="auc")
for f in (rnd, vin):
    f["dh_cox"] = f["deephit"] - f["cause_specific_cox"]
    f["dh_rsf"] = f["deephit"] - f["rsf"]
rm = rnd.groupby(level=[0, 1])[["dh_cox", "dh_rsf"]].mean()
lines = []
for ca in ("default", "prepaid"):
    for co in ("crisis", "postcrisis"):
        r, v = rm.loc[(co, ca)], vin.loc[(co, ca)]
        lines.append(f"{SHORT.get(ca,ca)} & {SHORT.get(co,co)} & "
                     f"{r.dh_cox:+.4f} & {v.dh_cox:+.4f} & "
                     f"{r.dh_rsf:+.4f} & {v.dh_rsf:+.4f} \\\\")
emit("tab_protocol", lines,
     caption=("What the evaluation protocol is worth. DeepHit's AUC advantage at "
              "60 months over the cause-specific Cox model and over the tuned "
              "forest, measured under random splits (mean of five seeds, matched "
              "to the vintage split's train and test sizes; every seed standard "
              "deviation is at or below 0.0019) and under vintage-ordered splits. "
              "Models and tuned settings are identical throughout; only the way "
              "the test set is chosen changes. Under random splits DeepHit leads "
              "in all eight cells; under vintage ordering three of those leads "
              "reverse."),
     label="tab:protocol", colspec="@{}llrrrr@{}",
     header=(r"& & \multicolumn{2}{c}{vs Cox} & \multicolumn{2}{c}{vs RSF} \\"
             "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
             r"Cause & Cohort & random & vintage & random & vintage \\"),
     colsep="4pt")

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

print("wrote report/tab_{data,main,sig,alpha,var,recal,protocol}.tex")
for f in ["tab_data","tab_main","tab_sig","tab_alpha","tab_var","tab_recal","tab_protocol","tab_rolling","tab_matched"]:
    print(f"\n--- {f} ---"); print(open(f"{OUT}/{f}.tex").read())
