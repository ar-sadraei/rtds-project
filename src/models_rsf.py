"""
Random survival forest for competing risks.

Approach: one forest per cause, with the competing event treated as censored, then
the two cause-specific cumulative hazards combined into a cumulative incidence by
the usual product-limit construction

    S(t|x)     = exp( -sum_m H_m(t|x) )
    CIF_k(t|x) = sum_{u <= t} S(u-|x) * dH_k(u|x)

This is a **cause-specific random survival forest**, not Ishwaran's competing-risks
RSF. Ishwaran's version grows a single forest with a composite splitting rule that
considers all causes at once; here each forest splits on one cause only. The
distinction matters and belongs in the report: Frydman & Matuszyk (2022) used the
Ishwaran formulation via randomForestSRC in R, so this is a close relative rather
than a replication of their estimator.

Unlike Cox, the hazard does not factorise into baseline x exp(xb) -- every subject
has their own curve -- so prediction is chunked to bound memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from metrics import CENSORED
from models_classical import _Standardiser

# monthly grid the CIF integral is accumulated on
_GRID = np.arange(1.0, 121.0)


def _step_to_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation of a cumulative hazard; 0 before the first knot."""
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.zeros((values.shape[0], len(grid)))
    ok = idx >= 0
    out[:, ok] = values[:, idx[ok]]
    return out


class CauseSpecificRSF:
    name = "rsf"

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 50,
                 max_features: str = "sqrt", random_state: int = 0,
                 n_jobs: int = -1, chunk: int = 20000) -> None:
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                       max_features=max_features, random_state=random_state,
                       n_jobs=n_jobs, low_memory=False)
        self.chunk = chunk

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray):
        self.std_ = _Standardiser().fit(X)
        Z = self.std_.transform(X)
        self.causes_ = tuple(int(c) for c in np.unique(event) if c != CENSORED)
        self.forests_ = {}
        for c in self.causes_:
            y = Surv.from_arrays(event=(event == c), time=time)
            self.forests_[c] = RandomSurvivalForest(**self.kw).fit(Z, y)
        return self

    def _cumhaz(self, Z: np.ndarray, cause: int) -> np.ndarray:
        """Cumulative hazard for one cause on the monthly grid, shape (n, len(_GRID))."""
        f = self.forests_[cause]
        out = np.empty((len(Z), len(_GRID)))
        for s in range(0, len(Z), self.chunk):
            block = Z[s:s + self.chunk]
            H = f.predict_cumulative_hazard_function(block, return_array=True)
            out[s:s + self.chunk] = _step_to_grid(f.unique_times_, H, _GRID)
        return out

    def predict_cif(self, X: pd.DataFrame, horizons: np.ndarray, cause: int) -> np.ndarray:
        Z = self.std_.transform(X)
        H = {c: self._cumhaz(Z, c) for c in self.causes_}

        total = sum(H.values())
        total_prev = np.hstack([np.zeros((len(Z), 1)), total[:, :-1]])
        S_prev = np.exp(-total_prev)

        dHk = np.diff(np.hstack([np.zeros((len(Z), 1)), H[cause]]), axis=1)
        cif = np.cumsum(S_prev * dHk, axis=1)

        j = np.searchsorted(_GRID, np.asarray(horizons, float), side="right") - 1
        out = np.zeros((len(Z), len(horizons)))
        ok = j >= 0
        out[:, ok] = cif[:, j[ok]]
        return np.clip(out, 0.0, 1.0)
