"""
Train/test splits.

Two protocols, deliberately both:

  vintage   train on earlier origination years, test on later ones. This is how a
            lender actually uses a model, and it is the protocol the reviewed credit
            studies almost never use (3 of 6).

  random    stratified random split, ignoring vintage. This is what Blumenstock et al.
            (2022) used -- 10 random train/test splits -- and what most of the
            literature reports. Kept as the contrast, not as the main protocol.

Reporting both is the point: the gap between them is evidence about whether random
splits flatter a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TRAIN_VINTAGES = (2013, 2014, 2015)
DEFAULT_TEST_VINTAGES = (2016, 2017, 2018)


@dataclass(frozen=True)
class Split:
    name: str
    protocol: str          # "vintage" | "random"
    seed: int | None
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __repr__(self) -> str:
        return (f"Split({self.name}, protocol={self.protocol}, seed={self.seed}, "
                f"n_train={len(self.train_idx):,}, n_test={len(self.test_idx):,})")


def vintage_split(
    df: pd.DataFrame,
    train_vintages: tuple[int, ...] = DEFAULT_TRAIN_VINTAGES,
    test_vintages: tuple[int, ...] = DEFAULT_TEST_VINTAGES,
) -> Split:
    """Time-ordered split by origination year."""
    overlap = set(train_vintages) & set(test_vintages)
    if overlap:
        raise ValueError(f"train and test vintages overlap: {sorted(overlap)}")
    if max(train_vintages) >= min(test_vintages):
        raise ValueError("every training vintage must precede every test vintage")

    tr = np.flatnonzero(df.vintage.isin(train_vintages).to_numpy())
    te = np.flatnonzero(df.vintage.isin(test_vintages).to_numpy())
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("a vintage split produced an empty side")

    name = f"vintage_{min(train_vintages)}-{max(train_vintages)}_to_{min(test_vintages)}-{max(test_vintages)}"
    return Split(name, "vintage", None, tr, te)


def random_split(
    df: pd.DataFrame,
    seed: int,
    test_frac: float = 0.5,
    stratify_on: str = "event",
) -> Split:
    """Stratified random split, ignoring vintage. The literature's usual protocol."""
    rng = np.random.default_rng(seed)
    n = len(df)
    test_mask = np.zeros(n, dtype=bool)

    strata = df[stratify_on].to_numpy()
    for value in np.unique(strata):
        pos = np.flatnonzero(strata == value)
        rng.shuffle(pos)
        k = int(round(test_frac * len(pos)))
        test_mask[pos[:k]] = True

    return Split(f"random_seed{seed}", "random", seed,
                 np.flatnonzero(~test_mask), np.flatnonzero(test_mask))


def make_splits(df: pd.DataFrame, n_random_seeds: int = 5) -> list[Split]:
    """The full set evaluated in the study: one vintage split, n random splits."""
    splits = [vintage_split(df)]
    splits += [random_split(df, seed=s) for s in range(n_random_seeds)]
    return splits


def describe(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Event composition and follow-up on each side -- print this in the report."""
    out = []
    for side, idx in (("train", split.train_idx), ("test", split.test_idx)):
        g = df.iloc[idx]
        shares = g.event.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
        out.append({
            "split": split.name,
            "side": side,
            "n": len(g),
            "vintages": f"{g.vintage.min()}-{g.vintage.max()}",
            "censored": round(shares[0], 4),
            "default": round(shares[1], 4),
            "prepaid": round(shares[2], 4),
            "median_followup_mo": float(g.last_age.median()),
        })
    return pd.DataFrame(out)
