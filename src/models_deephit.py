"""
DeepHit (Lee et al., AAAI 2018) for competing risks, via pycox.

Architecture is the standard competing-risks form: a shared trunk, then one
sub-network per cause, with a softmax over the joint (cause, time-bin) grid. Time
is discretised into equidistant bins; predictions are step-interpolated onto the
harness horizons.

The alpha parameter is the point of this whole project
-------------------------------------------------------
pycox's loss is

    loss = alpha * negative_log_likelihood + (1 - alpha) * ranking_loss

and its default is **alpha = 0.2**, i.e. 80% of the objective is the ranking term.
DeepHit's own paper tunes this weight on validation *discrimination*, which is
exactly the practice the review identifies: the model is trained and selected for
ranking, and its probabilities are whatever falls out.

We therefore make alpha explicit and sweep it, rather than inheriting a default.
alpha = 1.0 is pure likelihood (no ranking term) and is the natural comparison
point for "what would DeepHit look like if it were not optimised for ranking".

Anchors to check against
------------------------
Blumenstock et al. (2022): DeepHit achieved the highest concordance index in every
experiment on Freddie Mac mortgages, significantly beating cause-specific Cox and
Fine-Gray. If our DeepHit does not beat those on AUC, the implementation is wrong.

Alberge et al. (2026): on SEER, DeepHit failed both competing-risks calibration
tests in all five runs while Aalen-Johansen passed every time. That is the
expected direction for calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from metrics import CENSORED
from models_classical import _Standardiser


class _CauseSpecificNet(torch.nn.Module):
    """Shared trunk -> one head per cause -> (batch, n_causes, n_bins) logits."""

    def __init__(self, in_features: int, shared: list[int], indiv: list[int],
                 n_causes: int, out_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, shared[:-1], shared[-1], batch_norm=True,
            dropout=dropout, output_bias=False)
        self.risk_nets = torch.nn.ModuleList([
            tt.practical.MLPVanilla(shared[-1], indiv, out_features,
                                    batch_norm=True, dropout=dropout)
            for _ in range(n_causes)
        ])

    def forward(self, x):
        h = self.shared_net(x)
        return torch.stack([net(h) for net in self.risk_nets], dim=1)


class DeepHitModel:
    """
    Parameters
    ----------
    alpha : weight on the likelihood term. pycox default is 0.2 (ranking-dominated);
            1.0 is pure likelihood. Recorded in `name` so results stay traceable.
    """

    def __init__(self, alpha: float = 0.2, n_bins: int = 30, epochs: int = 50,
                 batch_size: int = 1024, lr: float = 1e-3, sigma: float = 0.1,
                 dropout: float = 0.1, seed: int = 0, verbose: bool = False) -> None:
        self.alpha, self.n_bins, self.epochs = alpha, n_bins, epochs
        self.batch_size, self.lr, self.sigma = batch_size, lr, sigma
        self.dropout, self.seed, self.verbose = dropout, seed, verbose
        self.name = f"deephit_a{alpha:g}"

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.std_ = _Standardiser().fit(X)
        Z = self.std_.transform(X).astype("float32")

        self.causes_ = tuple(int(c) for c in np.unique(event) if c != CENSORED)
        n_causes = len(self.causes_)

        self.labtrans_ = LabTransDiscreteTime(self.n_bins)
        idx, _ = self.labtrans_.fit_transform(time, (event != CENSORED).astype(int))
        y = (idx.astype("int64"), event.astype("int64"))   # event: 0 censored, 1..K cause

        net = _CauseSpecificNet(Z.shape[1], shared=[64, 64], indiv=[64],
                                n_causes=n_causes, out_features=self.n_bins,
                                dropout=self.dropout)
        self.model_ = DeepHit(net, tt.optim.Adam(self.lr), alpha=self.alpha,
                              sigma=self.sigma, duration_index=self.labtrans_.cuts)

        n_val = max(1, int(0.1 * len(Z)))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(Z))
        vi, ti = perm[:n_val], perm[n_val:]
        val = (Z[vi], (y[0][vi], y[1][vi]))

        self.model_.fit(Z[ti], (y[0][ti], y[1][ti]),
                        batch_size=self.batch_size, epochs=self.epochs,
                        callbacks=[tt.callbacks.EarlyStopping(patience=8)],
                        val_data=val, verbose=self.verbose)
        return self

    def predict_cif(self, X: pd.DataFrame, horizons: np.ndarray, cause: int) -> np.ndarray:
        Z = self.std_.transform(X).astype("float32")
        cif = self.model_.predict_cif(Z)          # (n_causes, n_bins, n_subjects)
        k = self.causes_.index(int(cause))
        curve = np.asarray(cif[k]).T              # (n_subjects, n_bins)

        cuts = np.asarray(self.labtrans_.cuts, dtype=float)
        idx = np.searchsorted(cuts, np.asarray(horizons, float), side="right") - 1
        out = np.zeros((len(Z), len(horizons)))
        ok = idx >= 0
        out[:, ok] = curve[:, idx[ok]]
        return np.clip(out, 0.0, 1.0)
