"""Does DeepHit's advantage appear when the feature set is richer and more non-linear?

BASE = 9 numeric origination features (what every model has used so far)
RICH = BASE + one-hot geography and product type (~70 features). Interactions among
       state, product and credit quality are exactly where a neural net should beat
       a linear Cox model, if it beats it anywhere.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from splits import vintage_split
from harness import evaluate, HORIZONS
from models_classical import CauseSpecificCox
from models_rsf import CauseSpecificRSF
from models_deephit import DeepHitModel

df = pd.read_parquet("data/processed/loans.parquet")
BASE = ["fico","cltv","dti","ltv","orig_upb","orig_rate","orig_term","n_borrowers","mi_pct"]
CATS = ["state","property_type","occupancy","channel","loan_purpose","first_time_buyer"]

rich = df[BASE].copy()
for c in CATS:
    d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
    rich = pd.concat([rich, d], axis=1)
df_rich = pd.concat([df[["vintage","time","event","last_age"]], rich], axis=1)
RICH = list(rich.columns)
print(f"BASE: {len(BASE)} features | RICH: {len(RICH)} features\n")

sp = vintage_split(df)
builders = [("cause_specific_cox", CauseSpecificCox),
            ("rsf", CauseSpecificRSF),
            ("deephit_a0.2", lambda: DeepHitModel(alpha=0.2, epochs=50)),
            ("deephit_a1", lambda: DeepHitModel(alpha=1.0, epochs=50))]

parts = []
for tag, feats, frame in (("base", BASE, df), ("rich", RICH, df_rich)):
    for name, B in builders:
        m = B()
        print(f"  {tag:4s} {m.name}", flush=True)
        r = evaluate(m, frame, sp, feats)
        r["featureset"] = tag
        parts.append(r)
res = pd.concat(parts, ignore_index=True)
res.to_csv("results/featureset.csv", index=False)

pd.set_option("display.width", 220)
for cause in ["default","prepaid"]:
    print(f"\n######## {cause} ########")
    t = (res[(res.cause==cause) & res.metric.isin(["auc_td","ici"]) & (res.horizon==60.0)]
           .pivot_table(index="model", columns=["metric","featureset"], values="value"))
    t.columns = [f"{m}_{f}" for m,f in t.columns]
    t["auc_gain"] = (t.auc_td_rich - t.auc_td_base).round(4)
    print(t.round(4).to_string())
print("\nwrote results/featureset.csv")
