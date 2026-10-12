import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from splits import vintage_split, random_split
from harness import AalenJohansenBaseline, evaluate, summarise
from models_classical import CauseSpecificCox, FineGray
from models_rsf import CauseSpecificRSF
from models_deephit import DeepHitModel

df = pd.read_parquet("data/processed/loans.parquet")
F = ["fico","cltv","dti","ltv","orig_upb","orig_rate","orig_term","n_borrowers","mi_pct"]
splits = [vintage_split(df)] + [random_split(df, seed=s) for s in range(3)]

def builders():
    yield AalenJohansenBaseline
    yield CauseSpecificCox
    yield FineGray
    yield CauseSpecificRSF
    for a in (0.2, 0.5, 1.0):          # 0.2 = pycox default (ranking-dominated); 1.0 = pure likelihood
        yield (lambda a=a: DeepHitModel(alpha=a, epochs=50))

parts = []
for sp in splits:
    for B in builders():
        m = B()
        print(f"  {m.name:18s} {sp.name}", flush=True)
        parts.append(evaluate(m, df, sp, F))
res = pd.concat(parts, ignore_index=True)
res.to_csv("results/all.csv", index=False)

pd.set_option("display.width", 220)
for metric in ["auc_td","ibs","ici"]:
    print(f"\n===== {metric} (default) =====")
    print(summarise(res, metric, "default").to_string(index=False))
print("\n===== runtime (s) =====")
print(res[res.metric=="fit_seconds"].groupby("model").value.mean().round(2).to_string())
print("\nwrote results/all.csv")
