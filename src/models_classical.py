"""
Classical competing-risks baselines: cause-specific Cox and Fine-Gray.

Both are written directly against the harness contract (fit / predict_cif) and
both are implemented here rather than pulled from a package, for two reasons:
no mainstream Python library ships Fine-Gray, and a protocol paper should have an
auditable estimator rather than a black box.

They share one engine. A Cox partial likelihood with risk-set weights

    L(b) = prod_i  exp(x_i b) / sum_{j in R(t_i)} w_j(t_i) exp(x_j b)

becomes

  cause-specific   R(t) = {j : T_j >= t},  w = 1, competing events censored
  Fine-Gray        R(t) = {j : T_j >= t} union {j : T_j < t, competing event},
                   with w_j(t) = G(t)/G(T_j) for those retained after their
                   competing event. The weighting is Fine & Gray's (1999); the
                   equivalent weighted-Cox formulation, and its extension to left
                   truncation, are due to Geskus (2011).

The Fine-Gray risk-set sum stays O(n) per iteration because it separates:

    sum_j w_j(t) e^{x_j b} = sum_{T_j >= t} e^{x_j b}
                             + G(t) * sum_{T_j < t, comp} e^{x_j b} / G(T_j)

a reverse cumulative sum plus a forward one. No data expansion, no counting-process
reshaping, so it runs on all 300,000 loans rather than a subsample.

Ties: Breslow. Baseline hazards: Breslow estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import CENSORED, _step_eval, kaplan_meier_censoring


# --------------------------------------------------------------------------
# shared preprocessing
# --------------------------------------------------------------------------
class _Standardiser:
    """Median-impute then z-score. Fitted on train only."""

    def fit(self, X: pd.DataFrame):
        A = X.to_numpy(dtype=float)
        self.median_ = np.nanmedian(A, axis=0)
        A = np.where(np.isnan(A), self.median_, A)
        self.mean_ = A.mean(axis=0)
        self.scale_ = np.where(A.std(axis=0) > 1e-12, A.std(axis=0), 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        A = X.to_numpy(dtype=float)
        A = np.where(np.isnan(A), self.median_, A)
        return (A - self.mean_) / self.scale_


# --------------------------------------------------------------------------
# weighted Cox engine
# --------------------------------------------------------------------------
def _risk_sums(Z: np.ndarray, eta: np.ndarray, order: np.ndarray,
               t_sorted: np.ndarray, event_times: np.ndarray,
               fg: dict | None):
    """
    Risk-set sums S0 (scalar) and S1 (vector) at each event time.

    Both terms of the Fine-Gray sum are cumulative sums, so this stays O(n p).
    """
    w = np.exp(np.clip(eta, -30.0, 30.0))
    ws, wZs = w[order], (w[:, None] * Z)[order]

    rev0 = np.concatenate([np.cumsum(ws[::-1])[::-1], [0.0]])
    rev1 = np.concatenate([np.cumsum(wZs[::-1], axis=0)[::-1], np.zeros((1, Z.shape[1]))])
    k = np.searchsorted(t_sorted, event_times, side="left")
    S0, S1 = rev0[k], rev1[k]

    if fg is not None:
        wi = w / fg["g_at_Ti"]
        c0 = np.where(fg["is_comp"], wi, 0.0)[order]
        c1 = np.where(fg["is_comp"][:, None], wi[:, None] * Z, 0.0)[order]
        fwd0 = np.concatenate([[0.0], np.cumsum(c0)])
        fwd1 = np.concatenate([np.zeros((1, Z.shape[1])), np.cumsum(c1, axis=0)])
        S0 = S0 + fg["g_at_event"] * fwd0[k]
        S1 = S1 + fg["g_at_event"][:, None] * fwd1[k]
    return S0, S1


def _fit_weighted_cox(Z: np.ndarray, time: np.ndarray, is_event: np.ndarray,
                      fg: dict | None = None, ridge: float = 1e-4):
    """
    Maximise the Breslow partial likelihood by L-BFGS with an analytic gradient.

    Covariates arrive standardised, so the linear predictor is centred and exp()
    does not overflow; a small ridge keeps the solution finite when a covariate is
    near-separating. No rescaling of the linear predictor is applied, so the
    baseline hazard below is on the same scale as predictions.
    """
    from scipy.optimize import minimize

    n, p = Z.shape
    order = np.argsort(time, kind="mergesort")
    t_sorted = time[order]

    event_times = np.unique(time[is_event])
    d = np.array([np.sum((time == t) & is_event) for t in event_times], float)
    Zsum = np.array([Z[(time == t) & is_event].sum(axis=0) for t in event_times])
    Zsum_tot = Zsum.sum(axis=0)

    if fg is not None:
        fg = dict(fg, order=order)

    def negll_grad(beta):
        S0, S1 = _risk_sums(Z, Z @ beta, order, t_sorted, event_times, fg)
        S0 = np.clip(S0, 1e-300, None)
        ll = float(Zsum_tot @ beta - np.sum(d * np.log(S0))) - ridge * float(beta @ beta)
        g = Zsum_tot - (d[:, None] * S1 / S0[:, None]).sum(axis=0) - 2 * ridge * beta
        return -ll, -g

    res = minimize(negll_grad, np.zeros(p), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-10})
    beta = res.x

    S0, _ = _risk_sums(Z, Z @ beta, order, t_sorted, event_times, fg)
    H0 = np.cumsum(d / np.clip(S0, 1e-300, None))
    return beta, event_times, H0, res.success


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class CauseSpecificCox:
    """
    One Cox model per cause, combined into a CIF.

        CIF_k(t|x) = sum_{u <= t} S(u-|x) * dH_k0(u) * exp(x b_k)
        S(t|x)     = exp( -sum_m H_m0(t) exp(x b_m) )

    Competing events are censored in each cause-specific fit -- legitimate for the
    hazard, and only misleading if 1 - S(t) is then read as an absolute probability.
    Going through the formula above is the correct route to that probability.
    """

    name = "cause_specific_cox"

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray):
        self.std_ = _Standardiser().fit(X)
        Z = self.std_.transform(X)
        self.causes_ = tuple(c for c in np.unique(event) if c != CENSORED)
        self.fits_ = {}
        for c in self.causes_:
            self.fits_[int(c)] = _fit_weighted_cox(Z, time, event == c)
        return self

    def predict_cif(self, X: pd.DataFrame, horizons: np.ndarray, cause: int) -> np.ndarray:
        Z = self.std_.transform(X)
        grid = np.unique(np.concatenate([f[1] for f in self.fits_.values()]))
        grid = grid[grid <= max(horizons)]
        if len(grid) == 0:
            return np.zeros((len(Z), len(horizons)))

        # cumulative hazard per cause on the shared grid
        Hs, lp = {}, {}
        for c, (beta, et, H0, _ok) in self.fits_.items():
            Hs[c] = _step_eval(et, H0, grid, left=0.0)
            lp[c] = np.exp(np.clip(Z @ beta, -30.0, 30.0))

        Hk = Hs[cause]
        dHk = np.diff(np.concatenate([[0.0], Hk]))

        # S(u-) uses the cumulative hazard just before u
        tot = sum(Hs[c][:, None] * lp[c][None, :] for c in self.fits_)
        tot_prev = np.vstack([np.zeros((1, len(Z))), tot[:-1]])
        S_prev = np.exp(-tot_prev)                      # (n_grid, n_subj)

        inc = S_prev * (dHk[:, None] * lp[cause][None, :])
        cif_grid = np.cumsum(inc, axis=0)               # (n_grid, n_subj)

        j = np.searchsorted(grid, horizons, side="right") - 1
        out = np.zeros((len(Z), len(horizons)))
        ok = j >= 0
        out[:, ok] = cif_grid[j[ok]].T
        return np.clip(out, 0.0, 1.0)


class FineGray:
    """
    Direct model for the subdistribution hazard of one cause.

        CIF_k(t|x) = 1 - exp( -H_k0(t) * exp(x b) )

    Subjects who experience a competing event stay in the risk set, weighted by
    G(t)/G(T_i). A separate model is fitted per cause, so the two CIFs are not
    constrained to sum below 1 -- a known property of Fine-Gray, worth a sentence
    in the report.
    """

    name = "fine_gray"

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray):
        self.std_ = _Standardiser().fit(X)
        self._X, self._time, self._event = X, time, event
        self._cache: dict[int, tuple] = {}
        self.g_t_, self.g_v_ = kaplan_meier_censoring(time, event)
        return self

    def _fit_cause(self, cause: int):
        if cause in self._cache:
            return self._cache[cause]
        Z = self.std_.transform(self._X)
        time, event = self._time, self._event

        is_event = event == cause
        is_comp = (event != cause) & (event != CENSORED)
        g_at_Ti = np.clip(_step_eval(self.g_t_, self.g_v_, time), 1e-10, None)

        order = np.argsort(time, kind="mergesort")
        event_times = np.unique(time[is_event])
        fg = {
            "is_comp": is_comp,
            "g_at_Ti": g_at_Ti,
            "order": order,
            "g_at_event": _step_eval(self.g_t_, self.g_v_, event_times),
        }
        self._cache[cause] = _fit_weighted_cox(Z, time, is_event, fg=fg)
        return self._cache[cause]

    def predict_cif(self, X: pd.DataFrame, horizons: np.ndarray, cause: int) -> np.ndarray:
        beta, et, H0, _ok = self._fit_cause(cause)
        Z = self.std_.transform(X)
        H = _step_eval(et, H0, np.asarray(horizons, float), left=0.0)
        lp = np.exp(np.clip(Z @ beta, -30.0, 30.0))
        return np.clip(1.0 - np.exp(-H[None, :] * lp[:, None]), 0.0, 1.0)
