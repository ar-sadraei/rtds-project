#!/usr/bin/env python3
"""
The evaluation metrics behind one signature.

    score(cif, t_train, e_train, t_test, e_test, times, n_causes, where)
        -> (rows, bins)

Each row is one number: {metric, cause, horizon, value, error}. Storing results this
way rather than one column per metric keeps the per-cause and per-horizon structure
out of column names, so the analysis is a groupby. A metric that cannot be computed
gets a row with an empty value and the reason in error, so failures show in the CSV.

Per cause and horizon: Brier score, scaled Brier score, concordance and calibration
gap. Per cause: the integrated Brier score. Across causes: the Aalen-Johansen
calibration and the CR D-calibration from hazardous (extra global measures), and the
share of predictions whose incidences sum above one.

The Aalen-Johansen estimator used for the baseline and for the calibration gap is
written here and handles tied event times exactly. Our times are whole months, so
almost every event time is tied.
"""

import warnings

import numpy as np
import pandas as pd
from numba import njit

from hazardous._ipcw import KaplanMeierIPCW
from hazardous.metrics import aj_calibration, d_cr_calibration
from hazardous.metrics._brier_score import IncidenceScoreComputer

from . import report_utils

# Draws used to average out the jitter inside hazardous aj_calibration
AJ_REPEATS = 5

# Number of quantile bins of the predicted incidence for the calibration gap
CALIBRATION_BINS = 5

# Two predictions closer than this count as tied in the concordance
TIED_TOLERANCE = 1e-8

# The failures a metric is expected to raise. Anything else is a bug and propagates.
EXPECTED_ERRORS = (
    ValueError,
    ZeroDivisionError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


def as_frame(t, e):
    return pd.DataFrame({"event": np.asarray(e, int), "duration": np.asarray(t, float)})


def pack(cif):
    """(n, K, T) incidences -> (n, 1 + K, T) with survival first, as hazardous wants."""
    cif = np.clip(np.asarray(cif, float), 0.0, 1.0)
    surv = np.clip(1.0 - cif.sum(axis=1), 0.0, 1.0)
    return np.concatenate([surv[:, None, :], cif], axis=1)


# ----------------------------------------------------------------------------------------
# Aalen-Johansen without jitter
# ----------------------------------------------------------------------------------------
def aalen_johansen(t, e, n_causes, times):
    """
    Cumulative incidence of each cause, shape (n_causes, len(times)).

    Events at the same time are processed together. A person censored at time u is
    still at risk at u, so censoring never removes someone from an event at u.
    """
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    times = np.asarray(times, float)
    if len(t) == 0:
        return np.zeros((n_causes, len(times)))
    unique, inverse = np.unique(t, return_inverse=True)
    at_risk = len(t) - np.searchsorted(np.sort(t), unique, side="left")
    counts = np.zeros((n_causes + 1, len(unique)))
    np.add.at(counts, (e, inverse), 1.0)
    any_event = counts[1:].sum(axis=0)
    surv_before = np.concatenate([[1.0], np.cumprod(1.0 - any_event / at_risk)[:-1]])
    cif = np.cumsum(surv_before * counts[1:] / at_risk, axis=1)
    pos = np.searchsorted(unique, times, side="right") - 1
    out = cif[:, np.clip(pos, 0, None)]
    out[:, pos < 0] = 0.0
    return out


# ----------------------------------------------------------------------------------------
# Concordance in O(n log n)
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def _add(tree, i, value):
    i += 1
    while i < len(tree):
        tree[i] += value
        i += i & (-i)


@njit(cache=True)
def _prefix(tree, i):
    total = 0.0
    i += 1
    while i > 0:
        total += tree[i]
        i -= i & (-i)
    return total


@njit(cache=True)
def _concordance_sums(event, rank, weight, order, bounds, k, n_ranks):
    """Numerator and denominator of the IPCW concordance for cause k."""
    num = 0.0
    den = 0.0
    n_groups = len(bounds) - 1

    # Type A: a case i against everyone still event free after it. Walk the duration
    # groups from the last to the first so the tree holds exactly the later rows.
    tree = np.zeros(n_ranks + 1)
    total = 0.0
    for g in range(n_groups - 1, -1, -1):
        lo, hi = bounds[g], bounds[g + 1]
        for p in range(lo, hi):
            j = order[p]
            if event[j] == 0:
                _add(tree, rank[j], 1.0)
                total += 1.0
        for p in range(lo, hi):
            i = order[p]
            if event[i] == k:
                below = _prefix(tree, rank[i] - 1)
                tied = _prefix(tree, rank[i]) - below
                w = weight[i] * weight[i]
                num += w * (below + 0.5 * tied)
                den += w * total
        for p in range(lo, hi):
            j = order[p]
            if event[j] != 0:
                _add(tree, rank[j], 1.0)
                total += 1.0

    # Type B: a case i against rows that had a competing event no later than i
    tree = np.zeros(n_ranks + 1)
    total = 0.0
    for g in range(n_groups):
        lo, hi = bounds[g], bounds[g + 1]
        for p in range(lo, hi):
            j = order[p]
            if event[j] != 0 and event[j] != k:
                _add(tree, rank[j], weight[j])
                total += weight[j]
        for p in range(lo, hi):
            i = order[p]
            if event[i] == k:
                below = _prefix(tree, rank[i] - 1)
                tied = _prefix(tree, rank[i]) - below
                num += weight[i] * (below + 0.5 * tied)
                den += weight[i] * total
    return num, den


def concordance(event, duration, pred, weight, k, tau):
    """
    Time-dependent concordance of cause k at tau, with inverse censoring weights.

    The same quantity as hazardous concordance_index_incidence, in O(n log n) so the
    bootstrap can afford it. Returns nan when cause k has no event up to tau.
    """
    event = np.where(duration > tau, 0, event).astype(np.int64)
    if not (event == k).any():
        return np.nan
    # Predictions closer than the tolerance are one value
    _, rank = np.unique(np.round(pred / TIED_TOLERANCE), return_inverse=True)
    order = np.argsort(duration, kind="stable")
    sorted_duration = duration[order]
    bounds = np.concatenate(
        [np.flatnonzero(np.diff(sorted_duration)) + 1, [len(order)]]
    )
    bounds = np.concatenate([[0], bounds]).astype(np.int64)
    num, den = _concordance_sums(
        event,
        rank.astype(np.int64),
        weight.astype(np.float64),
        order.astype(np.int64),
        bounds,
        int(k),
        int(rank.max()) + 1,
    )
    return num / den if den > 0 else np.nan


# ----------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------
def calibration_bins(pred, t, e, k, tau, n_causes):
    """
    Quantile bins of the predicted incidence of cause k at tau.

    Returns one dict per bin with the mean prediction, the observed cumulative
    incidence (Aalen-Johansen on the rows of the bin) and the number of rows.
    """
    if len(np.unique(pred)) < 2:
        groups = np.zeros(len(pred), dtype=int)
    else:
        groups = pd.qcut(pred, CALIBRATION_BINS, labels=False, duplicates="drop")
    out = []
    for b in np.unique(groups):
        rows = groups == b
        observed = aalen_johansen(t[rows], e[rows], n_causes, [tau])[k - 1, 0]
        out.append(
            {
                "bin": int(b) + 1,
                "mean_pred": float(pred[rows].mean()),
                "observed": float(observed),
                "n": int(rows.sum()),
            }
        )
    return out


def calibration_gap(bins):
    """Weighted mean absolute difference between predicted and observed incidence."""
    n = sum(b["n"] for b in bins)
    return sum(b["n"] * abs(b["mean_pred"] - b["observed"]) for b in bins) / n


# ----------------------------------------------------------------------------------------
# One fold
# ----------------------------------------------------------------------------------------
class FoldScorer:
    """
    Everything that depends on the training rows of one fold, fitted once: the
    censoring distribution and the no-covariate Aalen-Johansen baseline. The
    per-cause measures can then be recomputed on any set of test rows, which is what
    the bootstrap does.
    """

    def __init__(self, t_train, e_train, times, n_causes):
        self.times = np.asarray(times, float)
        self.n_causes = n_causes
        y_train = as_frame(t_train, e_train)
        self.baseline = aalen_johansen(t_train, e_train, n_causes, self.times)
        self.ipcw = KaplanMeierIPCW().fit(
            {"event": y_train["event"] > 0, "duration": y_train["duration"]}
        )
        self.computers = {
            k: IncidenceScoreComputer(y_train, event_of_interest=k)
            for k in range(1, n_causes + 1)
        }

    def brier(self, cif, t, e, k):
        """Brier score of cause k at every horizon, shape (n_times,)."""
        y_test = as_frame(t, e)
        return self.computers[k].brier_score_incidence(
            y_test, cif[:, k - 1, :], self.times
        )

    def scaled_brier(self, brier, cif, t, e, k):
        """1 minus the Brier score over the one of the no-covariate baseline."""
        base = np.tile(self.baseline[k - 1], (len(t), 1))
        reference = self.computers[k].brier_score_incidence(
            as_frame(t, e), base, self.times
        )
        return 1.0 - brier / reference

    def concordance(self, cif, t, e, k, j):
        weight = self.ipcw.compute_ipcw_at(np.asarray(t, float))
        return concordance(
            np.asarray(e, int),
            np.asarray(t, float),
            cif[:, k - 1, j],
            weight,
            k,
            self.times[j],
        )

    def measures(self, cif, t, e):
        """
        Brier, scaled Brier, concordance and calibration gap, each (n_causes, n_times),
        and the calibration bins behind the gap.
        """
        t = np.asarray(t, float)
        e = np.asarray(e, int)
        shape = (self.n_causes, len(self.times))
        out = {m: np.full(shape, np.nan) for m in METRICS}
        bins = []
        for k in range(1, self.n_causes + 1):
            brier = self.brier(cif, t, e, k)
            out["brier"][k - 1] = brier
            with np.errstate(divide="ignore", invalid="ignore"):
                out["scaled_brier"][k - 1] = self.scaled_brier(brier, cif, t, e, k)
            for j, tau in enumerate(self.times):
                out["concordance"][k - 1, j] = self.concordance(cif, t, e, k, j)
                table = calibration_bins(cif[:, k - 1, j], t, e, k, tau, self.n_causes)
                out["calibration_gap"][k - 1, j] = calibration_gap(table)
                bins += [{"cause": k, "horizon": tau, **row} for row in table]
        return out, bins

    def mean_ibs(self, cif, t, e):
        """Integrated Brier score averaged over the causes, the tuning criterion."""
        span = self.times[-1] - self.times[0]
        scores = [
            np.trapezoid(self.brier(cif, t, e, k), self.times) / span
            for k in range(1, self.n_causes + 1)
        ]
        return float(np.mean(scores))


METRICS = ("brier", "scaled_brier", "concordance", "calibration_gap")


# ----------------------------------------------------------------------------------------
# All the numbers of one fit
# ----------------------------------------------------------------------------------------
def score(cif, t_train, e_train, t_test, e_test, times, n_causes, where=""):
    """Every metric of one fit as rows, plus the calibration bins."""
    times = np.asarray(times, float)
    rows = []

    def add(metric, cause, horizon, value, error=""):
        if error == "" and (value is None or not np.isfinite(value)):
            error = "not defined (no events of this cause up to the horizon?)"
        if error:
            report_utils.log(f"{where}: {metric} cause {cause} {horizon}: {error}")
            value = None
        rows.append(
            {
                "metric": metric,
                "cause": cause,
                "horizon": horizon,
                "value": None if value is None else float(value),
                "error": error,
            }
        )

    scorer = FoldScorer(t_train, e_train, times, n_causes)
    try:
        values, bins = scorer.measures(cif, t_test, e_test)
    except EXPECTED_ERRORS as err:
        add("per_cause_measures", None, None, None, f"{type(err).__name__}: {err}")
        values, bins = {}, []
    for metric, grid in values.items():
        for k in range(1, n_causes + 1):
            for j, tau in enumerate(times):
                add(metric, k, tau, grid[k - 1, j])
    for k in range(1, n_causes + 1):
        if "brier" in values:
            span = times[-1] - times[0]
            add("ibs", k, None, np.trapezoid(values["brier"][k - 1], times) / span)

    # Across causes: the two hazardous calibration measures. Our times are whole
    # months, so almost every event time is tied; the Aalen-Johansen estimator inside
    # aj_calibration breaks ties by random jitter, which makes a single call
    # irreproducible. Repeating it and reporting the mean fixes that, and the spread
    # is kept as the noise floor: two models whose AJ-calibration differs by less than
    # it are not distinguishable, however the ranking happens to fall.
    y_te = as_frame(t_test, e_test)
    packed = pack(cif)
    try:
        # The tie warning is expected and handled by the repeats, so silence it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tied event times")
            draws = [
                float(aj_calibration(y_te, times, packed)) for _ in range(AJ_REPEATS)
            ]
        add("aj_calibration", None, None, float(np.mean(draws)))
        add("aj_calibration_spread", None, None, float(max(draws) - min(draws)))
    except EXPECTED_ERRORS as err:
        add("aj_calibration", None, None, None, f"{type(err).__name__}: {err}")
    try:
        add(
            "cr_d_calibration",
            None,
            None,
            float(d_cr_calibration(y_te, y_test_pred=packed, times=times)),
        )
    except EXPECTED_ERRORS as err:
        add("cr_d_calibration", None, None, None, f"{type(err).__name__}: {err}")

    # Coherence: Fine-Gray and cs-RSF fit one model per cause, so their incidences
    # are not constrained to sum below one. Report how often that is violated.
    add("incoherent_cells", None, None, float((cif.sum(axis=1) > 1.0).mean()))
    return rows, bins
