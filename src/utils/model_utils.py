#!/usr/bin/env python3
"""
The four benchmarked models behind one interface.

Every model is wrapped in a function with the same signature and the same return
shape, so the benchmark loop does not know which model it is running:

    fit_predict(params, X_train, t_train, e_train, X_test, times, n_causes)
        -> cif, fit_seconds, predict_seconds

    cif has shape (n_test, n_causes, n_times) and cif[i, k - 1, j] is the
    predicted cumulative incidence of cause k for subject i at times[j].

The four libraries return three incompatible shapes natively: one array per cause,
one array with causes on axis 1, and one with causes first and time discretized
into bins. This module is where that is absorbed, once.
"""

import sys
import time
from pathlib import Path

import numpy as np

DSM_DIR = (
    Path(__file__).resolve().parents[2] / "vendor" / "nfg" / "DeepSurvivalMachines"
)
if str(DSM_DIR) not in sys.path:
    sys.path.insert(0, str(DSM_DIR))


# ----------------------------------------------------------------------------------------
# cs-RSF: one random survival forest per cause, other causes recoded as censored
# ----------------------------------------------------------------------------------------
def fit_cs_rsf(params, X_tr, t_tr, e_tr, X_te, times, n_causes):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    start = time.perf_counter()
    hazards = []
    for k in range(1, n_causes + 1):
        y = Surv.from_arrays(event=(e_tr == k), time=t_tr)
        forest = RandomSurvivalForest(random_state=0, n_jobs=-1, **params).fit(X_tr, y)
        hazards.append(forest)
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    cum = [
        np.vstack([f(times) for f in m.predict_cumulative_hazard_function(X_te)])
        for m in hazards
    ]
    # The CIF must come from the survival of ALL causes together. Using 1 - S_k
    # from each forest alone gives incoherent probabilities that sum above one.
    total = sum(cum)
    zeros = np.zeros((len(X_te), 1))
    surv = np.exp(-np.concatenate([zeros, total[:, :-1]], axis=1))
    cif = np.stack(
        [
            np.cumsum(
                surv * np.diff(np.concatenate([zeros, h], axis=1), axis=1), axis=1
            )
            for h in cum
        ],
        axis=1,
    )
    return cif, fit_s, time.perf_counter() - start


# ----------------------------------------------------------------------------------------
# Fine-Gray: one subdistribution hazard model per cause
# ----------------------------------------------------------------------------------------
def fit_fine_gray(params, X_tr, t_tr, e_tr, X_te, times, n_causes):
    from comprisk import FineGrayRegression, Surv

    y = Surv.from_arrays(event=e_tr.astype(int), time=t_tr)
    start = time.perf_counter()
    models = [
        FineGrayRegression(cause=k, **params).fit(X_tr, y)
        for k in range(1, n_causes + 1)
    ]
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    cif = np.stack(
        [
            np.asarray(m.predict_cumulative_incidence(X_te, times=times), float)
            for m in models
        ],
        axis=1,
    )
    return cif, fit_s, time.perf_counter() - start


# ----------------------------------------------------------------------------------------
# DeepHit: one softmax over all (cause, time bin) pairs
# ----------------------------------------------------------------------------------------
def fit_deephit(params, X_tr, t_tr, e_tr, X_te, times, n_causes):
    import torch
    import torchtuples as tt
    from pycox.models import DeepHit
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime

    class CRLabTransform(LabTransDiscreteTime):
        """Discretize the durations but keep the cause label; pycox's is single-event."""

        def transform(self, durations, events):
            dur, is_event = super().transform(durations, events > 0)
            ev = np.asarray(events).copy()
            ev[is_event == 0] = 0
            return dur, ev.astype("int64")

    n_bins = params.get("n_bins", 20)
    nodes = params.get("nodes", [32, 32])
    dropout = params.get("dropout", 0.1)
    lr = params.get("lr", 0.01)
    alpha = params.get("alpha", 0.2)
    sigma = params.get("sigma", 0.1)
    epochs = params.get("epochs", 10)
    batch = params.get("batch_size", 256)

    torch.manual_seed(0)
    labtrans = CRLabTransform(n_bins)
    y_tr = labtrans.fit_transform(t_tr, e_tr)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = tt.practical.MLPVanilla(
                X_tr.shape[1],
                nodes,
                n_causes * n_bins,
                batch_norm=True,
                dropout=dropout,
            )

        def forward(self, x):
            return self.body(x).view(x.size(0), n_causes, n_bins)

    model = DeepHit(
        Net(), tt.optim.Adam(lr), alpha=alpha, sigma=sigma, duration_index=labtrans.cuts
    )
    start = time.perf_counter()
    model.fit(
        X_tr.astype("float32"), y_tr, batch_size=batch, epochs=epochs, verbose=False
    )
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    # pycox returns (causes, bins, n); interpolate each curve onto our horizons
    raw = np.asarray(model.predict_cif(X_te.astype("float32")), float)
    cif = np.stack(
        [
            np.stack([np.interp(times, labtrans.cuts, row) for row in raw[k].T])
            for k in range(n_causes)
        ],
        axis=1,
    )
    return cif, fit_s, time.perf_counter() - start


# ----------------------------------------------------------------------------------------
# Deep Survival Machines: mixture of parametric distributions
# ----------------------------------------------------------------------------------------
DSM_EPS = 1e-3


def fit_dsm(params, X_tr, t_tr, e_tr, X_te, times, n_causes):
    import contextlib
    import io

    from dsm import DeepSurvivalMachines

    model = DeepSurvivalMachines(
        k=params.get("k", 3),
        layers=params.get("layers", [32]),
        distribution=params.get("distribution", "Weibull"),
    )
    start = time.perf_counter()
    # DSM prints a 10,000-step pretraining bar to stderr on every fit
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
        io.StringIO()
    ):
        model.fit(
            X_tr.astype("float64"),
            t_tr.astype("float64") + DSM_EPS,
            e_tr.astype("float64"),
            iters=params.get("iters", 10),
            learning_rate=params.get("lr", 1e-3),
            batch_size=params.get("batch_size", 256),
            random_state=0,
        )
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
        io.StringIO()
    ):
        cif = np.stack(
            [
                np.asarray(
                    model.predict_risk(
                        X_te.astype("float64"), list(times + DSM_EPS), risk=k
                    ),
                    float,
                )
                for k in range(1, n_causes + 1)
            ],
            axis=1,
        )
    return cif, fit_s, time.perf_counter() - start


MODELS = {
    "cs-RSF": fit_cs_rsf,
    "Fine-Gray": fit_fine_gray,
    "DeepHit": fit_deephit,
    "DSM": fit_dsm,
}

# Searched by 1_tune_*.py; Fine-Gray has nothing to tune and is reported untuned
SEARCH_SPACE = {
    "cs-RSF": {
        "n_estimators": [100, 300],
        "min_samples_leaf": [5, 15, 50],
        "max_depth": [10, 15],
    },
    "Fine-Gray": {},
    "DeepHit": {
        "nodes": [[32], [32, 32], [64, 64]],
        "dropout": [0.0, 0.1, 0.3],
        "lr": [1e-3, 1e-2],
        "alpha": [0.2, 0.5],
        "n_bins": [20],
        "epochs": [20],
    },
    "DSM": {
        "k": [3, 4, 6],
        "layers": [[32], [64], [32, 32]],
        "lr": [1e-3, 1e-2],
        "iters": [10, 20],
    },
}

DEFAULTS = {
    "cs-RSF": {"n_estimators": 100, "min_samples_leaf": 15, "max_depth": 15},
    "Fine-Gray": {},
    "DeepHit": {},
    "DSM": {},
}
