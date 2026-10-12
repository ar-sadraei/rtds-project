"""Sanity checks. The harness must pass these before any real model is trusted."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from metrics import aalen_johansen, kaplan_meier_censoring, auc_td, brier_cr, ici, CENSORED, DEFAULT, PREPAID
from splits import make_splits, describe, vintage_split
from harness import AalenJohansenBaseline, evaluate, HORIZONS, summarise

df = pd.read_parquet("data/processed/loans.parquet")
FEATURES = ["fico","cltv","dti","ltv","orig_upb","orig_rate","orig_term","n_borrowers","mi_pct"]

print("=== 1. splits ===")
sp = vintage_split(df)
print(sp); print(describe(df, sp).to_string(index=False))

print("\n=== 2. Aalen-Johansen marginal CIF, full data ===")
for cause,label in [(DEFAULT,"default"),(PREPAID,"prepaid")]:
    aj = aalen_johansen(df.time.to_numpy(float), df.event.to_numpy(int), cause, HORIZONS)
    print(f"  {label:8s}", " ".join(f"{h:.0f}mo={v:.4f}" for h,v in zip(HORIZONS,aj)))
aj_d = aalen_johansen(df.time.to_numpy(float), df.event.to_numpy(int), DEFAULT, np.array([200.0]))[0]
aj_p = aalen_johansen(df.time.to_numpy(float), df.event.to_numpy(int), PREPAID, np.array([200.0]))[0]
print(f"  CIF_default(inf)+CIF_prepaid(inf) = {aj_d+aj_p:.4f}  (must be <= 1)")
assert aj_d + aj_p <= 1.0 + 1e-9, "CIFs sum above 1 -- estimator is wrong"

print("\n=== 3. null model on the vintage split ===")
res = evaluate(AalenJohansenBaseline(), df, sp, FEATURES)
piv = res[res.metric.isin(["auc_td","brier","ici"])].pivot_table(
        index=["cause","horizon"], columns="metric", values="value").round(4)
print(piv.to_string())

auc_vals = res[(res.metric=="auc_td")].value.dropna()
ici_v = res[(res.metric=="ici")].value.dropna()
print(f"\n  AUC range {auc_vals.min():.4f}-{auc_vals.max():.4f}  (expect exactly 0.5: constant prediction)")
assert abs(auc_vals.mean()-0.5) < 0.01, "null model discriminates -- AUC is wrong"

# The null model is calibrated only when train and test share a distribution.
# On a RANDOM split it must be ~perfect; on the VINTAGE split any gap is the
# distribution shift itself, which is what the split exists to expose.
from splits import random_split
rs = random_split(df, seed=0)
res_rand = evaluate(AalenJohansenBaseline(), df, rs, FEATURES)
ici_rand = res_rand[res_rand.metric=="ici"].value.dropna()
print(f"  ICI, random  split: max {ici_rand.max():.4f}  (expect ~0: same distribution)")
print(f"  ICI, vintage split: max {ici_v.max():.4f}  (this IS the vintage shift)")
assert ici_rand.max() < 0.02, "null model miscalibrated in-distribution -- calibration metric is wrong"
assert ici_v.max() > ici_rand.max(), "vintage split shows no shift -- suspicious"

print("\n=== 4. a perfect-information oracle must beat the null on AUC ===")
class Oracle:
    name="oracle"
    def fit(self,X,t,e): self._t,self._e=t,e; return self
    def predict_cif(self,X,horizons,cause):
        # cheats: uses the test outcome. Only to prove AUC responds to signal.
        y=((self._t_te<=horizons[:,None]) & (self._e_te==cause)).T.astype(float)
        return np.clip(y*0.98+0.01,0,1)
orc=Oracle(); orc._t_te=df.iloc[sp.test_idx].time.to_numpy(float); orc._e_te=df.iloc[sp.test_idx].event.to_numpy(int)
r2=evaluate(orc, df, sp, FEATURES)
print(f"  oracle mean AUC = {r2[r2.metric=='auc_td'].value.mean():.4f}  (expect ~1.0)")
assert r2[r2.metric=="auc_td"].value.mean() > 0.95, "AUC does not respond to signal"

print("\n=== 5. runtime captured ===")
print(res[res.metric.isin(["fit_seconds","predict_seconds"])][["cause","metric","value"]].to_string(index=False))
print("  hardware:", res.hardware.iloc[0])

print("\nALL CHECKS PASSED")
