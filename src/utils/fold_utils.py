"""
Build the evaluation folds once, verify them, and save them.

Every model in the benchmark must see exactly the same splits, or comparisons
between models are not paired and the bootstrap is invalid. So folds are built
here, written to disk, and read by the runner. Nothing downstream re-splits.

THREE ARMS
    kfold      stratified 5-fold on group x event. Answers the main question.
    temporal   expanding window on GROUP boundaries (origination vintage, or
               NHANES survey cycle). Train on all earlier groups, test on the
               next one.
    matched    a stratified random split matched to its temporal partner on
               BOTH training-set size and test-set size. Used only to compare
               with the temporal arm.

WHY NOT sklearn's TimeSeriesSplit
    It splits on row count, not on group. On our calm cohort (55,000 loans over
    six vintages) four of its five folds put part of a vintage in training and
    the rest of the same vintage in test -- exactly the leakage the temporal arm
    exists to prevent. Splitting on group boundaries is six lines and cannot
    leak.

WHY THE ARMS MUST BE SIZE-MATCHED
    An expanding window trains on 17% of the data in its first fold and 83% in
    its last. A plain 5-fold split trains on a constant 80%. Comparing the two
    would confound the split rule with the amount of training data, and the
    split rule is the thing under test. Matching sizes per fold makes the
    difference within a pair attributable to the assignment rule alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from . import report_utils

ARMS = ("kfold", "temporal", "matched")

# Folds and smallest stratum allowed for the kfold arm
KFOLD_SPLITS = 5
MIN_STRATUM = 5


def temporal_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding window on group boundaries.

    Fold k trains on every group before level k and tests on level k alone, so
    no group can appear on both sides. With G groups this yields G-1 folds.
    """
    levels = sorted(pd.unique(groups))
    out = []
    for k in range(1, len(levels)):
        train = np.flatnonzero(np.isin(groups, levels[:k]))
        test = np.flatnonzero(groups == levels[k])
        out.append((train, test))
    return out


def matched_random_folds(
    groups: np.ndarray,
    events: np.ndarray,
    temporal: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    One random fold per temporal fold, matched on train and test size.

    Stratified on group x event so that each random fold holds the same mix of
    entry periods and outcomes as the pool. Without stratifying on event, a
    rare cause can land unevenly across folds and the calibration numbers move
    for that reason alone.
    """
    strata = pd.Series(groups).astype(str) + "_" + pd.Series(events).astype(str)
    out = []
    for i, (tr, te) in enumerate(temporal):
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=len(tr),
            test_size=len(te),
            random_state=seed + i,
        )
        r_tr, r_te = next(sss.split(np.zeros(len(groups)), strata))
        out.append((r_tr, r_te))
    return out


def kfold_folds(
    groups: np.ndarray, events: np.ndarray, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold on group x event, shuffled and seeded.

    If any stratum has fewer than MIN_STRATUM rows the strata fall back to the
    event alone, which is printed.
    """
    strata = pd.Series(groups).astype(str) + "_" + pd.Series(events).astype(str)
    smallest = int(strata.value_counts().min())
    if smallest < MIN_STRATUM:
        report_utils.log(
            f"kfold: the smallest group x event stratum has {smallest} rows, "
            f"fewer than {MIN_STRATUM}. Falling back to event-only strata."
        )
        strata = pd.Series(events).astype(str)
    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(groups)), strata)]


def verify(
    name: str,
    groups: np.ndarray,
    events: np.ndarray,
    temporal: list,
    matched: list,
    kfold: list,
) -> bool:
    """
    Check the folds before any model is fitted.

    A silently wrong fold produces results that look fine and answer a
    different question, so every property the design depends on is asserted
    here rather than assumed.
    """
    problems: list[str] = []

    for i, (tr, te) in enumerate(temporal, 1):
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            problems.append(
                f"temporal fold {i}: group(s) {sorted(overlap)} on both sides"
            )
        if set(tr) & set(te):
            problems.append(
                f"temporal fold {i}: {len(set(tr) & set(te))} rows on both sides"
            )
        if max(groups[tr]) >= min(groups[te]):
            problems.append(
                f"temporal fold {i}: trains on a group at or after the test group"
            )

    for i, ((tt, te_t), (rt, re_r)) in enumerate(zip(temporal, matched), 1):
        if len(tt) != len(rt):
            problems.append(f"pair {i}: train sizes differ, {len(tt):,} vs {len(rt):,}")
        if len(te_t) != len(re_r):
            problems.append(
                f"pair {i}: test sizes differ, {len(te_t):,} vs {len(re_r):,}"
            )
        if set(rt) & set(re_r):
            problems.append(f"matched fold {i}: rows on both sides")

    n = len(groups)
    covered = np.concatenate([te for _, te in kfold])
    if len(covered) != n or len(np.unique(covered)) != n:
        problems.append("kfold: the test sets are not disjoint or miss some rows")
    for i, (tr, te) in enumerate(kfold, 1):
        if set(tr) & set(te):
            problems.append(f"kfold fold {i}: rows on both sides")
        if len(tr) + len(te) != n:
            problems.append(f"kfold fold {i}: train and test do not cover every row")

    if problems:
        report_utils.log("Fold check FAILED:")
        for p in problems:
            report_utils.log(f"  {p}")
    else:
        report_utils.log(
            "Fold check passed: no group leakage, sizes matched, no row in both sides"
        )
    return not problems


def build(
    df: pd.DataFrame, group_col: str, event_col: str, name: str, seed: int = 0
) -> dict | None:
    g = df[group_col].to_numpy()
    e = df[event_col].to_numpy()
    tf = temporal_folds(g)
    mf = matched_random_folds(g, e, tf, seed)
    kf = kfold_folds(g, e, seed)
    if not verify(name, g, e, tf, mf, kf):
        return None
    return {
        "name": name,
        "group_col": group_col,
        "seed": seed,
        "n": len(df),
        "kfold": [{"train": t.tolist(), "test": s.tolist()} for t, s in kf],
        "temporal": [{"train": t.tolist(), "test": s.tolist()} for t, s in tf],
        "matched": [{"train": t.tolist(), "test": s.tolist()} for t, s in mf],
    }


def make_folds(
    table_file: Path,
    group_col: str,
    id_col: str,
    event_col: str,
    name: str,
    folds_file: Path,
    seed: int = 0,
) -> None:
    """Read a prepared table, build and verify its folds, and save them as JSON."""
    df = pd.read_csv(table_file).sort_values([group_col, id_col]).reset_index(drop=True)
    spec = build(df, group_col, event_col, name, seed)
    if spec is None:
        sys.exit("\nfold verification failed, nothing written")

    folds_file.parent.mkdir(parents=True, exist_ok=True)
    folds_file.write_text(json.dumps(spec))
    report_utils.log(f"Saved {folds_file}")


def show_folds(table_file: Path, id_col: str, event_col: str, folds_file: Path) -> None:
    """Print the size of every arm and the temporal folds from the saved folds."""
    spec = json.loads(Path(folds_file).read_text())
    df = (
        pd.read_csv(table_file)
        .sort_values([spec["group_col"], id_col])
        .reset_index(drop=True)
    )
    groups = df[spec["group_col"]].to_numpy()
    events = df[event_col].to_numpy()
    folds = {
        arm: [(np.array(f["train"]), np.array(f["test"])) for f in spec[arm]]
        for arm in ARMS
    }

    def span(sizes):
        low, high = min(sizes), max(sizes)
        return f"{low:,}" if low == high else f"{low:,} to {high:,}"

    arms = [
        {
            "Arm": arm,
            "Folds": len(pairs),
            "Train rows": span([len(tr) for tr, _ in pairs]),
            "Test rows": span([len(te) for _, te in pairs]),
        }
        for arm, pairs in folds.items()
    ]
    report_utils.table(pd.DataFrame(arms), right=("Train rows", "Test rows"))

    report_utils.log("Temporal folds (the matched arm has the same sizes)")
    rows = []
    for i, (tr, te) in enumerate(folds["temporal"], 1):
        tg = sorted(set(groups[tr]))
        shown = f"{tg[0]}-{tg[-1]}" if len(tg) > 1 else str(tg[0])
        ev = pd.Series(events[te]).value_counts().sort_index()
        rows.append(
            {
                "Fold": i,
                "Train": len(tr),
                "Test": len(te),
                "Train groups": shown,
                "Test group": groups[te][0],
                "Events in test": " ".join(f"{k}:{v}" for k, v in ev.items()),
            }
        )
    report_utils.table(pd.DataFrame(rows))
