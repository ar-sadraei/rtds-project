#!/usr/bin/env python3
"""
The four evaluation metrics behind one signature.

    score(cif, t_train, e_train, t_test, e_test, times, n_causes) -> list of rows

Each row is one number: {metric, cause, horizon, value}. Storing results this way
rather than one column per metric keeps the per-cause and per-horizon structure out
of column names, so the analysis is a groupby.
"""

import numpy as np
import pandas as pd

# Draws used to average out the Aalen-Johansen tie-breaking jitter
AJ_REPEATS = 5
from hazardous.metrics import (
    aj_calibration,
    concordance_index_incidence,
    d_cr_calibration,
    integrated_brier_score_incidence,
)


def as_frame(t, e):
    return pd.DataFrame({"event": np.asarray(e, int), "duration": np.asarray(t, float)})


def pack(cif):
    """(n, K, T) incidences -> (n, 1 + K, T) with survival first, as hazardous wants."""
    cif = np.clip(np.asarray(cif, float), 0.0, 1.0)
    surv = np.clip(1.0 - cif.sum(axis=1), 0.0, 1.0)
    return np.concatenate([surv[:, None, :], cif], axis=1)


def score(cif, t_train, e_train, t_test, e_test, times, n_causes):
    y_tr, y_te = as_frame(t_train, e_train), as_frame(t_test, e_test)
    packed = pack(cif)
    times = np.asarray(times, float)
    rows = []

    def add(metric, cause, horizon, value):
        rows.append(
            {
                "metric": metric,
                "cause": cause,
                "horizon": horizon,
                "value": (
                    None if value is None or not np.isfinite(value) else float(value)
                ),
            }
        )

    # Per cause: discrimination and the proper scoring rule, at each horizon
    for k in range(1, n_causes + 1):
        pred_k = cif[:, k - 1, :]
        for j, tau in enumerate(times):
            try:
                c = concordance_index_incidence(
                    y_te,
                    pred_k[:, [j]],
                    y_train=y_tr,
                    time_grid=np.array([tau]),
                    taus=[tau],
                    event_of_interest=k,
                )
                c = float(np.mean(c)) if hasattr(c, "__len__") else float(c)
            except Exception:
                c = None
            add("concordance", k, tau, c)
        try:
            ibs = integrated_brier_score_incidence(
                y_tr, y_te, pred_k, times, event_of_interest=k
            )
            add("ibs", k, None, float(ibs))
        except Exception:
            add("ibs", k, None, None)

    # Across causes: the two calibration measures, integrated over the horizons.
    # Our times are whole months, so almost every event time is tied: fold 7 of the
    # medical data has 73 distinct times among 5,576 people. The Aalen-Johansen
    # estimator cannot take ties and breaks them by random jitter, which makes a
    # single call irreproducible. Repeating it and reporting the mean fixes that, and
    # the spread is kept as the noise floor: two models whose AJ-calibration differs
    # by less than it are not distinguishable, however the ranking happens to fall.
    try:
        draws = [float(aj_calibration(y_te, times, packed)) for _ in range(AJ_REPEATS)]
        add("aj_calibration", None, None, float(np.mean(draws)))
        add("aj_calibration_spread", None, None, float(max(draws) - min(draws)))
    except Exception:
        add("aj_calibration", None, None, None)
        add("aj_calibration_spread", None, None, None)
    try:
        add(
            "cr_d_calibration",
            None,
            None,
            float(d_cr_calibration(y_te, y_test_pred=packed, times=times)),
        )
    except Exception:
        add("cr_d_calibration", None, None, None)

    # Coherence: Fine-Gray and cs-RSF fit one model per cause, so their incidences
    # are not constrained to sum below one. Report how often that is violated.
    add("incoherent_cells", None, None, float((cif.sum(axis=1) > 1.0).mean()))
    return rows
