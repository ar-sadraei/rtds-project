import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from splits import vintage_split, random_split, describe
from harness import AalenJohansenBaseline, evaluate, summarise, HORIZONS
from models_classical import CauseSpecificCox, FineGray
from models_rsf import CauseSpecificRSF

df = pd.read_parquet("data/processed/loans.parquet")
FEATURES = ["fico","cltv","dti","ltv","orig_upb","orig_rate","orig_term","n_borrowers","mi_pct"]

splits = [vintage_split(df)] + [random_split(df, seed=s) for s in range(3)]
models = [AalenJohansenBaseline, CauseSpecificCox, FineGray, CauseSpecificRSF]

parts = []
for sp in splits:
    for M in models:
        m = M()
        print(f"  {m.name:20s} {sp.name} ...", flush=True)
        parts.append(evaluate(m, df, sp, FEATURES))
res = pd.concat(parts, ignore_index=True)
res.to_csv("results/classical.csv", index=False)

pd.set_option("display.width", 200)
for metric in ["auc_td","ibs","ici"]:
    print(f"\n===== {metric} (default) =====")
    print(summarise(res, metric, "default").to_string(index=False))
print("\n===== runtime (seconds) =====")
rt = res[res.metric.isin(["fit_seconds","predict_seconds"])]
print(rt.groupby(["model","metric"]).value.mean().round(3).to_string())
