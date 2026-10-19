"""
Evaluation harness.

Every model implements the same two methods, so no model can be given a metric the
others cannot be given:

    fit(X_train, time_train, event_train) -> self
    predict_cif(X, horizons, cause) -> ndarray (n_subjects, n_horizons)

`evaluate()` runs one model on one split and returns tidy rows:

    model, split, protocol, seed, cause, metric, horizon, value

Every number in the report comes from that table, so each claim is
traceable to a single row.

Runtime is instrumented here, not bolted on afterwards. Filip's gap is that the
benchmarking literature omits computational cost; the harness therefore records
fit and predict seconds, plus the hardware, for every run.
"""

from __future__ import annotations

import platform
import time as _time
from typing import Protocol

import numpy as np
import pandas as pd

from metrics import (CENSORED, DEFAULT, PREPAID, aalen_johansen, auc_td, brier_cr,
                     ibs_cr, ici, kaplan_meier_censoring)
from splits import Split

# Horizons in months. 12-60 keeps every vintage comparable: the 2018 cohort has a
# median of 29 months of follow-up, so anything beyond ~60 is thinly observed there.
HORIZONS = np.array([12, 24, 36, 48, 60], dtype=float)


class SurvivalModel(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> "SurvivalModel": ...

    def predict_cif(self, X: pd.DataFrame, horizons: np.ndarray, cause: int) -> np.ndarray: ...


class AalenJohansenBaseline:
    """
    Marginal CIF, identical for every subject. The null model.

    It cannot discriminate, so its AUC must come out at ~0.5; it is calibrated by
    construction, so its ICI must be near zero. Any model that fails to beat it on
    AUC, or fails to match it on calibration, is telling you something -- and in
    Alberge et al. (2026) the deep models did fail to match it on calibration while
    Aalen-Johansen passed every run. Use it to validate the harness before trusting
    any result.
    """

    name = "aalen_johansen"

    def __init__(self) -> None:
        self._curves: dict[int, np.ndarray] = {}
        self._horizons: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray):
        self._time, self._event = time, event
        return self

    def predict_cif(self, X: pd.DataFrame, horizons: np.ndarray, cause: int) -> np.ndarray:
        key = (cause, tuple(horizons))
        if key not in self._curves:
            self._curves[key] = aalen_johansen(self._time, self._event, cause, horizons)
        return np.tile(self._curves[key], (len(X), 1))


def evaluate(
    model: SurvivalModel,
    df: pd.DataFrame,
    split: Split,
    features: list[str],
    causes: tuple[int, ...] = (DEFAULT, PREPAID),
    horizons: np.ndarray = HORIZONS,
) -> pd.DataFrame:
    """Fit on the training side, evaluate on the test side, return tidy metric rows."""
    tr, te = df.iloc[split.train_idx], df.iloc[split.test_idx]
    X_tr, X_te = tr[features], te[features]
    t_tr, e_tr = tr.time.to_numpy(float), tr.event.to_numpy(int)
    t_te, e_te = te.time.to_numpy(float), te.event.to_numpy(int)

    # censoring distribution estimated on TRAIN, applied to TEST -- no leakage
    g_t, g_v = kaplan_meier_censoring(t_tr, e_tr)

    t0 = _time.perf_counter()
    model.fit(X_tr, t_tr, e_tr)
    fit_s = _time.perf_counter() - t0

    rows: list[dict] = []
    for cause in causes:
        t0 = _time.perf_counter()
        cif = model.predict_cif(X_te, horizons, cause)
        predict_s = _time.perf_counter() - t0

        cif = np.clip(np.asarray(cif, dtype=float), 0.0, 1.0)
        if cif.shape != (len(X_te), len(horizons)):
            raise ValueError(
                f"{model.name}: predict_cif returned {cif.shape}, "
                f"expected {(len(X_te), len(horizons))}"
            )

        def add(metric: str, value: float, horizon: float | None = None) -> None:
            rows.append({
                "model": model.name, "split": split.name, "protocol": split.protocol,
                "seed": split.seed, "cause": {DEFAULT: "default", PREPAID: "prepaid"}[cause],
                "metric": metric, "horizon": horizon, "value": value,
            })

        for j, h in enumerate(horizons):
            add("auc_td", auc_td(cif[:, j], t_te, e_te, cause, h, g_t, g_v), h)
            add("brier", brier_cr(cif[:, j], t_te, e_te, cause, h, g_t, g_v), h)
            add("ici", ici(cif[:, j], t_te, e_te, cause, h), h)

        add("ibs", ibs_cr(cif, horizons, t_te, e_te, cause, g_t, g_v))
        add("fit_seconds", fit_s)
        add("predict_seconds", predict_s)

    out = pd.DataFrame(rows)
    out["n_train"], out["n_test"] = len(tr), len(te)
    out["hardware"] = f"{platform.processor() or platform.machine()} / {platform.system()}"
    return out


def run_all(models: list[SurvivalModel], df: pd.DataFrame, splits: list[Split],
            features: list[str]) -> pd.DataFrame:
    """Full grid: every model on every split. Returns one tidy frame."""
    parts = []
    for split in splits:
        for model in models:
            print(f"  {model.name} on {split.name} ...", flush=True)
            parts.append(evaluate(model, df, split, features))
    return pd.concat(parts, ignore_index=True)


def summarise(results: pd.DataFrame, metric: str, cause: str = "default") -> pd.DataFrame:
    """Mean and sd of a metric per model and protocol -- the shape of the results table."""
    sel = results[(results.metric == metric) & (results.cause == cause)]
    return (sel.groupby(["model", "protocol"])["value"]
               .agg(["mean", "std", "count"]).round(4).reset_index())
