#!/usr/bin/env python3
"""
The four benchmarked models behind one interface.

Every model is wrapped in a function with the same signature and the same return
shape, so the benchmark loop does not know which model it is running:

    fit_predict(params, X_train, t_train, e_train, X_test, times, n_causes, seed)
        -> cif, fit_seconds, predict_seconds

    The seed changes the initialization of the deep models and nothing else. The
    validation rows they hold out for early stopping are chosen with a fixed seed, so
    every seed sees the same split.

    cif has shape (n_test, n_causes, n_times) and cif[i, k - 1, j] is the
    predicted cumulative incidence of cause k for subject i at times[j].

The four libraries return three incompatible shapes natively: one array per cause,
one array with causes on axis 1, and one with causes first and time discretized
into bins. This module is where that is absorbed, once.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DSM_DIR = Path(__file__).resolve().parents[2] / "vendor" / "DeepSurvivalMachines"
if str(DSM_DIR) not in sys.path:
    sys.path.insert(0, str(DSM_DIR))


# ----------------------------------------------------------------------------------------
# cs-RSF: one random survival forest per cause, other causes recoded as censored
# ----------------------------------------------------------------------------------------
def fit_cs_rsf(params, X_tr, t_tr, e_tr, X_te, times, n_causes, seed=0):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    start = time.perf_counter()
    hazards = []
    for k in range(1, n_causes + 1):
        y = Surv.from_arrays(event=(e_tr == k), time=t_tr)
        forest = RandomSurvivalForest(random_state=seed, n_jobs=-1, **params).fit(
            X_tr, y
        )
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
def fit_fine_gray(params, X_tr, t_tr, e_tr, X_te, times, n_causes, seed=0):
    from comprisk import FineGrayRegression, PenalizedFineGrayRegression, Surv

    # penalty "none" is the plain model; the others shrink the coefficients and
    # choose the penalty strength by BIC along the lambda path
    params = dict(params)
    penalty = params.pop("penalty", "none")

    def make(k):
        if penalty == "none":
            return FineGrayRegression(cause=k, **params)
        return PenalizedFineGrayRegression(cause=k, penalty=penalty, **params)

    y = Surv.from_arrays(event=e_tr.astype(int), time=t_tr)
    start = time.perf_counter()
    models = [make(k).fit(X_tr, y) for k in range(1, n_causes + 1)]
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
# Early stopping for DeepHit: share of the training rows held out, maximum epochs and
# patience in epochs
EARLY_STOP_FRACTION = 0.15
DEEPHIT_MAX_EPOCHS = 100
DEEPHIT_PATIENCE = 10

# Models that are fitted once per seed; the others run once
SEEDED = ("DeepHit", "DSM")


def fit_deephit(params, X_tr, t_tr, e_tr, X_te, times, n_causes, seed=0):
    import torch
    import torchtuples as tt
    from pycox.models import DeepHit
    from sklearn.model_selection import train_test_split
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
    batch = params.get("batch_size", 256)

    # Hold out rows for early stopping, with a fixed seed so every seed sees the same
    fit_idx, val_idx = train_test_split(
        np.arange(len(X_tr)),
        test_size=EARLY_STOP_FRACTION,
        stratify=e_tr,
        random_state=0,
    )
    torch.manual_seed(seed)
    labtrans = CRLabTransform(n_bins)
    y_fit = labtrans.fit_transform(t_tr[fit_idx], e_tr[fit_idx])
    y_val = labtrans.transform(t_tr[val_idx], e_tr[val_idx])

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
    # The best weights are checkpointed to a temporary file, not the working folder
    handle, checkpoint = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    stopper = tt.callbacks.EarlyStopping(
        patience=DEEPHIT_PATIENCE, file_path=checkpoint, rm_file=False
    )
    start = time.perf_counter()
    try:
        model.fit(
            X_tr[fit_idx].astype("float32"),
            y_fit,
            batch_size=batch,
            epochs=DEEPHIT_MAX_EPOCHS,
            callbacks=[stopper],
            verbose=False,
            val_data=(X_tr[val_idx].astype("float32"), y_val),
        )
    finally:
        os.remove(checkpoint)
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

# DSM holds out 15 percent of the rows itself and stops after three epochs without
# an improvement, so this only has to be higher than it ever gets
DSM_MAX_ITERS = 1000


def fit_dsm(params, X_tr, t_tr, e_tr, X_te, times, n_causes, seed=0):
    import contextlib
    import io

    import torch
    from dsm import DeepSurvivalMachines

    model = DeepSurvivalMachines(
        k=params.get("k", 3),
        layers=params.get("layers", [32]),
        distribution=params.get("distribution", "Weibull"),
    )
    # DSM draws its initial weights from torch's global generator
    torch.manual_seed(seed)
    start = time.perf_counter()
    # DSM prints a 10,000-step pretraining bar to stderr on every fit
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
        io.StringIO()
    ):
        model.fit(
            X_tr.astype("float64"),
            t_tr.astype("float64") + DSM_EPS,
            e_tr.astype("float64"),
            iters=DSM_MAX_ITERS,
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

# Searched by 1_tune_*.py. A dict is a grid of its lists, a list is a set of settings.
# The deep models have no epochs setting: early stopping decides.
SEARCH_SPACE = {
    "cs-RSF": {
        "n_estimators": [100, 300],
        "min_samples_leaf": [5, 15, 50],
        "max_depth": [10, 15],
    },
    "Fine-Gray": [
        {"penalty": "none"},
        {"penalty": "ridge"},
        {"penalty": "lasso"},
        {"penalty": "elasticnet", "l1_ratio": 0.5},
    ],
    "DeepHit": {
        "nodes": [[32], [32, 32], [64, 64]],
        "dropout": [0.0, 0.1, 0.3],
        "lr": [1e-3, 1e-2],
        "alpha": [0.2, 0.5],
        "n_bins": [20],
    },
    "DSM": {
        "k": [3, 4, 6],
        "layers": [[32], [64], [32, 32]],
        "lr": [1e-3, 1e-2],
    },
}
