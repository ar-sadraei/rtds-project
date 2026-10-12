"""
Evaluation metrics for competing-risks survival predictions.

Every model in this study must expose the same object: a predicted cumulative
incidence function for cause k, evaluated on a grid of horizons --

    cif[i, j] = P(event of cause k by horizon[j] | covariates of subject i)

Everything below is computed from that matrix, so no model gets a metric the
others cannot be given.

Metric families, matching the three axes of the review:

  discrimination   time-dependent AUC at a horizon (IPCW, Blanche et al.)
                   -- answers "does it rank?"
  calibration      IPCW Brier score for the CIF (Graf; Schoop et al. for the
                   competing-risks form), integrated Brier score, a calibration
                   table against the Aalen-Johansen estimate, and the integrated
                   calibration index
                   -- answers "are the probabilities right?"
  cost             measured in harness.py, not here

Censoring is handled by inverse probability of censoring weights throughout, with
G(t) the Kaplan-Meier estimate of the censoring distribution fitted on the
*training* set and applied to the test set. On this data G(60) = 0.9986, so those
weights are all close to one: it is the competing event, not censoring, that
shapes the results.

NOT IMPLEMENTED: a per-cause D-calibration. Alberge et al. (2026) introduce one;
Ali has read that paper and should add it here. Deliberately left out rather than
approximated, so nothing in the results is a measure we invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CENSORED, DEFAULT, PREPAID = 0, 1, 2


# --------------------------------------------------------------------------
# non-parametric estimators
# --------------------------------------------------------------------------
def kaplan_meier_censoring(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    KM estimate of the censoring survival function G(t) = P(C > t).

    Censoring is the 'event' here, so the indicator is event == 0.
    Returns (times, G) as a right-continuous step function.
    """
    order = np.argsort(time, kind="mergesort")
    t, e = time[order], event[order]
    uniq = np.unique(t)

    at_risk = np.searchsorted(t, uniq, side="left")
    n_at_risk = len(t) - at_risk
    n_cens = np.array([np.sum((t == u) & (e == CENSORED)) for u in uniq])

    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(n_at_risk > 0, 1.0 - n_cens / n_at_risk, 1.0)
    return uniq, np.cumprod(factors)


def _step_eval(grid_t: np.ndarray, grid_v: np.ndarray, q: np.ndarray,
               left: float = 1.0) -> np.ndarray:
    """
    Evaluate a right-continuous step function at points q.

    `left` is the value before the first knot: 1.0 for a survival function such as
    G(t), 0.0 for a cumulative hazard. Getting this wrong inflates every predicted
    probability, so callers must state it.
    """
    idx = np.searchsorted(grid_t, q, side="right") - 1
    out = np.full(np.shape(np.asarray(q, dtype=float)), float(left))
    ok = idx >= 0
    out[ok] = grid_v[idx[ok]]
    return out


def aalen_johansen(time: np.ndarray, event: np.ndarray, cause: int,
                   horizons: np.ndarray) -> np.ndarray:
    """
    Marginal cumulative incidence of `cause`, evaluated at `horizons`.

    Vectorised: event counts per unique time come from bincount, so this is fast
    enough to sit inside a bootstrap loop.

    This is the non-parametric reference. Used as the null model in the harness:
    it cannot discriminate (AUC = 0.5) but it is calibrated by construction, which
    makes it the sanity check for the calibration metrics.
    """
    n = len(time)
    if n == 0:
        return np.zeros(len(np.atleast_1d(horizons)))

    uniq, inv = np.unique(time, return_inverse=True)
    m = len(uniq)

    n_at_time = np.bincount(inv, minlength=m).astype(float)
    d_k = np.bincount(inv, weights=(event == cause).astype(float), minlength=m)
    d_any = np.bincount(inv, weights=(event != CENSORED).astype(float), minlength=m)

    # subjects still at risk just before each unique time
    at_risk = n - np.concatenate(([0.0], np.cumsum(n_at_time)[:-1]))

    with np.errstate(divide="ignore", invalid="ignore"):
        haz_any = np.where(at_risk > 0, d_any / at_risk, 0.0)
        inc_k = np.where(at_risk > 0, d_k / at_risk, 0.0)

    surv_prev = np.concatenate(([1.0], np.cumprod(1.0 - haz_any)[:-1]))
    cif_curve = np.cumsum(surv_prev * inc_k)

    return _step_eval(uniq, cif_curve, np.atleast_1d(np.asarray(horizons, float)),
                      left=0.0)


# --------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------
def auc_td(risk: np.ndarray, time: np.ndarray, event: np.ndarray, cause: int,
           horizon: float, g_t: np.ndarray, g_v: np.ndarray) -> float:
    """
    Cumulative/dynamic time-dependent AUC for `cause` at `horizon`, IPCW-weighted.

    cases    subjects with an event of `cause` at or before the horizon
    controls subjects still event-free after the horizon, plus subjects who had a
             competing event by the horizon -- they can never have cause k, so they
             belong in the control set when the prediction target is the CIF

    Weights follow Blanche et al. (2013). A case is weighted by 1/G(T_i-). The two
    kinds of control are weighted differently, and the distinction matters: a
    subject still event-free at the horizon is known to be a control because they
    survived uncensored to the horizon, so the weight is 1/G(horizon), while a
    subject who had the competing event at T_i <= horizon is known to be a control
    only because they were uncensored up to T_i, so the weight is 1/G(T_i-).

    Giving every control 1/G(horizon) -- which this function did until the audit of
    12 Oct 2026 -- over-weights the competing-event controls by G(T_i-)/G(horizon).
    On this data G(60) = 0.9986, so the correction moved no reported figure by as
    much as 1e-4; see AUDIT.md. brier_cr has always made the same distinction.
    """
    is_case = (time <= horizon) & (event == cause)
    ctrl_free = time > horizon
    ctrl_comp = (time <= horizon) & (event != cause) & (event != CENSORED)
    is_ctrl = ctrl_free | ctrl_comp

    if is_case.sum() == 0 or is_ctrl.sum() == 0:
        return float("nan")

    g_at_h = max(_step_eval(g_t, g_v, np.array([horizon]))[0], 1e-8)
    w_case = 1.0 / np.clip(_step_eval(g_t, g_v, time[is_case] - 1e-9), 1e-8, None)

    w_all = np.zeros_like(np.asarray(time, dtype=float))
    w_all[ctrl_free] = 1.0 / g_at_h
    w_all[ctrl_comp] = 1.0 / np.clip(
        _step_eval(g_t, g_v, time[ctrl_comp] - 1e-9), 1e-8, None)

    rc, wc = risk[is_case], w_case
    rk, wk = risk[is_ctrl], w_all[is_ctrl]

    # Weighted Mann-Whitney U in O(n log n): for each case, accumulate the control
    # weight strictly below its risk, plus half the weight tied with it.
    order = np.argsort(rk, kind="mergesort")
    rk_s, wk_s = rk[order], wk[order]
    cum = np.concatenate(([0.0], np.cumsum(wk_s)))

    lo = np.searchsorted(rk_s, rc, side="left")
    hi = np.searchsorted(rk_s, rc, side="right")
    below = cum[lo]
    tied = cum[hi] - cum[lo]

    num = float(np.sum(wc * (below + 0.5 * tied)))
    den = float(np.sum(wc) * np.sum(wk))
    return num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------
# calibration / accuracy
# --------------------------------------------------------------------------
def brier_cr(cif: np.ndarray, time: np.ndarray, event: np.ndarray, cause: int,
             horizon: float, g_t: np.ndarray, g_v: np.ndarray) -> float:
    """
    IPCW Brier score for the CIF of `cause` at `horizon` (Graf; Schoop et al.).

    Lower is better. Penalises a wrong probability, not only a wrong ordering,
    which is exactly the property the credit studies in the review almost never test.

        subject had any event at T <= t   -> weight 1/G(T-),  target 1{cause}
        subject event-free at t           -> weight 1/G(t),   target 0
        subject censored at T <= t        -> weight 0 (uninformative)
    """
    had_event = (time <= horizon) & (event != CENSORED)
    still_at_risk = time > horizon

    w = np.zeros_like(time, dtype=float)
    w[had_event] = 1.0 / np.clip(_step_eval(g_t, g_v, time[had_event] - 1e-9), 1e-8, None)
    w[still_at_risk] = 1.0 / max(_step_eval(g_t, g_v, np.array([horizon]))[0], 1e-8)

    y = ((time <= horizon) & (event == cause)).astype(float)
    denom = w.sum()
    return float(np.sum(w * (y - cif) ** 2) / denom) if denom > 0 else float("nan")


def ibs_cr(cif_matrix: np.ndarray, horizons: np.ndarray, time: np.ndarray,
           event: np.ndarray, cause: int, g_t: np.ndarray, g_v: np.ndarray) -> float:
    """Integrated Brier score: trapezoidal integral of brier_cr over the horizon grid."""
    bs = np.array([
        brier_cr(cif_matrix[:, j], time, event, cause, h, g_t, g_v)
        for j, h in enumerate(horizons)
    ])
    ok = ~np.isnan(bs)
    if ok.sum() < 2:
        return float("nan")
    h, b = np.asarray(horizons, float)[ok], bs[ok]
    return float(np.trapezoid(b, h) / (h[-1] - h[0]))


def calibration_table(cif: np.ndarray, time: np.ndarray, event: np.ndarray,
                      cause: int, horizon: float, n_bins: int = 10) -> pd.DataFrame:
    """
    Predicted vs observed CIF by bin of predicted risk.

    Observed is the Aalen-Johansen estimate within the bin, so competing events and
    censoring are both handled. This is the Gerds-style calibration curve, and it is
    the same comparison Thackham & Ma make between predicted and observed defaults.
    """
    edges = np.quantile(cif, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(cif, edges[1:-1])

    rows = []
    for k in range(n_bins):
        m = b == k
        if m.sum() < 20:
            continue
        obs = float(aalen_johansen(time[m], event[m], cause, np.array([horizon]))[0])
        rows.append({"bin": k, "n": int(m.sum()),
                     "predicted": float(cif[m].mean()), "observed": obs})
    return pd.DataFrame(rows)


def ici(cif: np.ndarray, time: np.ndarray, event: np.ndarray, cause: int,
        horizon: float, n_bins: int = 10) -> float:
    """
    Integrated calibration index: n-weighted mean |observed - predicted| across bins.
    One number for 'how far off are the probabilities'. Zero is perfect.
    """
    tab = calibration_table(cif, time, event, cause, horizon, n_bins)
    if tab.empty:
        return float("nan")
    return float(np.average((tab.observed - tab.predicted).abs(), weights=tab.n))
