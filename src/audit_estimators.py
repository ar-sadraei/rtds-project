"""Does the weighted-Cox estimator recover known coefficients?"""
import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from models_classical import CauseSpecificCox, FineGray

rng = np.random.default_rng(7)
n = 40000
X = pd.DataFrame(rng.normal(size=(n,3)), columns=list("abc"))
BETA1 = np.array([0.8,-0.5,0.3])     # cause 1
BETA2 = np.array([-0.4,0.2,0.6])     # cause 2
lp1, lp2 = np.exp(X.values@BETA1), np.exp(X.values@BETA2)
t1 = rng.exponential(1/(0.01*lp1)); t2 = rng.exponential(1/(0.02*lp2))
tc = rng.exponential(1/0.004)
t  = np.minimum(np.minimum(t1,t2),tc)
e  = np.where((t1<=t2)&(t1<=tc),1,np.where((t2<t1)&(t2<=tc),2,0))
print("event mix:", {int(k): int(v) for k,v in zip(*np.unique(e,return_counts=True))})

m = CauseSpecificCox().fit(X,t,e)
for c,truth in ((1,BETA1),(2,BETA2)):
    est = m.fits_[c][0] / m.std_.scale_       # undo standardisation
    print(f"  cause {c}: true {np.round(truth,3)}  est {np.round(est,3)}  "
          f"max|err| {np.max(np.abs(est-truth)):.4f}")

# CIFs from the two causes must sum to <= 1 at every horizon
H = np.array([5.,20.,60.,200.])
c1 = m.predict_cif(X.iloc[:2000], H, 1); c2 = m.predict_cif(X.iloc[:2000], H, 2)
print("  max CIF1+CIF2 over subjects/horizons:", round(float((c1+c2).max()),4), "(must be <= 1)")

fg = FineGray().fit(X,t,e)
cif_fg = fg.predict_cif(X.iloc[:2000], H, 1)
print("  Fine-Gray CIF1 at t=200 vs cause-specific:",
      round(float(cif_fg[:,3].mean()),4), "vs", round(float(c1[:,3].mean()),4))
