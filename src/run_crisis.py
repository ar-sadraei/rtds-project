"""Does DeepHit's advantage appear when the data spans a crisis?

Train 2005-2006, test 2007-2009. Default rate 10.5% vs 4.5% post-crisis, and a
genuine structural break between training and test vintages -- the setting where
a flexible model has the most to gain over a proportional-hazards baseline.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from splits import Split, vintage_split
from harness import AalenJohansenBaseline, evaluate
from models_classical import CauseSpecificCox, FineGray
from models_rsf import CauseSpecificRSF
from models_deephit import DeepHitModel

df = pd.read_parquet("data/processed/loans_crisis.parquet")
BASE = ["fico","cltv","dti","ltv","orig_upb","orig_rate","orig_term","n_borrowers","mi_pct"]
CATS = ["state","property_type","occupancy","channel","loan_purpose","first_time_buyer"]
rich = df[BASE].copy()
for c in CATS:
    rich = pd.concat([rich, pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)], axis=1)
frame = pd.concat([df[["vintage","time","event","last_age"]], rich], axis=1)
RICH = list(rich.columns)

sp = vintage_split(df, train_vintages=(2005, 2006), test_vintages=(2007, 2008, 2009))
print(sp, f"| {len(RICH)} features\n")

builders = [("aalen_johansen", AalenJohansenBaseline),
            ("cause_specific_cox", CauseSpecificCox),
            ("fine_gray", FineGray),
            ("rsf", CauseSpecificRSF),
            ("deephit_a0.2", lambda: DeepHitModel(alpha=0.2, epochs=50)),
            ("deephit_a1", lambda: DeepHitModel(alpha=1.0, epochs=50))]

parts = []
for name, B in builders:
    m = B(); print(f"  {m.name}", flush=True)
    parts.append(evaluate(m, frame, sp, RICH))
res = pd.concat(parts, ignore_index=True)
res.to_csv("results/crisis.csv", index=False)

pd.set_option("display.width", 220)
for cause in ["default","prepaid"]:
    print(f"\n######## {cause}  (crisis: train 2005-06, test 2007-09) ########")
    t = (res[(res.cause==cause) & res.metric.isin(["auc_td","ici"]) & (res.horizon==60.0)]
           .pivot_table(index="model", columns="metric", values="value"))
    print(t.round(4).to_string())
print("\nwrote results/crisis.csv")
