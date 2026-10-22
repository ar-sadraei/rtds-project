"""
Build the evaluation folds once, verify them, and save them.

Every model in the benchmark must see exactly the same splits, or comparisons
between models are not paired and the bootstrap is invalid. So folds are built
here, written to disk, and read by the runner. Nothing downstream re-splits.

TWO ARMS
    temporal   expanding window on GROUP boundaries (origination vintage, or
               NHANES survey cycle). Train on all earlier groups, test on the
               next one.
    random     a stratified random split matched to its temporal partner on
               BOTH training-set size and test-set size.

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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


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


def verify(
    name: str, groups: np.ndarray, events: np.ndarray, temporal: list, random_: list
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

    for i, ((tt, te_t), (rt, re_r)) in enumerate(zip(temporal, random_), 1):
        if len(tt) != len(rt):
            problems.append(f"pair {i}: train sizes differ, {len(tt):,} vs {len(rt):,}")
        if len(te_t) != len(re_r):
            problems.append(
                f"pair {i}: test sizes differ, {len(te_t):,} vs {len(re_r):,}"
            )
        if set(rt) & set(re_r):
            problems.append(f"random fold {i}: rows on both sides")

    print(f"\n=== {name} ===")
    print(
        f"  {'fold':>4} {'train':>8} {'test':>7}  {'train groups':<28} {'test group':>10}"
        f"  {'events in test':>16}"
    )
    for i, (tr, te) in enumerate(temporal, 1):
        tg = sorted(set(groups[tr]))
        shown = f"{tg[0]}-{tg[-1]}" if len(tg) > 1 else str(tg[0])
        ev = pd.Series(events[te]).value_counts().sort_index()
        evs = " ".join(f"{k}:{v}" for k, v in ev.items())
        print(
            f"  {i:>4} {len(tr):>8,} {len(te):>7,}  {shown:<28} {groups[te][0]:>10}  {evs:>16}"
        )

    if problems:
        print("  FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print("  all checks passed: no group leakage, sizes matched, no row in both sides")
    return True


def build(
    df: pd.DataFrame, group_col: str, event_col: str, name: str, seed: int = 0
) -> dict | None:
    g = df[group_col].to_numpy()
    e = df[event_col].to_numpy()
    tf = temporal_folds(g)
    rf = matched_random_folds(g, e, tf, seed)
    if not verify(name, g, e, tf, rf):
        return None
    return {
        "name": name,
        "group_col": group_col,
        "seed": seed,
        "n": len(df),
        "temporal": [{"train": t.tolist(), "test": s.tolist()} for t, s in tf],
        "random": [{"train": t.tolist(), "test": s.tolist()} for t, s in rf],
    }


def main() -> None:
    out_dir = Path("results/folds")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = []

    for fname, label in (
        ("financial_calm_2013_2018.csv", "financial-calm"),
        ("financial_crisis_2005_2009.csv", "financial-crisis"),
    ):
        p = Path("data/banking/1_merged") / fname
        if not p.exists():
            print(f"  skipping {label}: {p} not found")
            continue
        d = pd.read_csv(p).sort_values(["vintage", "loan_id"]).reset_index(drop=True)
        specs.append(build(d, "vintage", "event", label))

    p = Path("data/medical/1_merged/nhanes_merged.csv")
    if p.exists():
        d = pd.read_csv(p)
        # medical event/time are not in the merged file yet; derive here so the
        # folds can be checked. This must move into the medical pipeline.
        CUT = 72
        d["event"] = np.where(
            (d.mortality_status == 1) & (d.followup_months <= CUT),
            np.where(
                d.cause_of_death.isin([1, 5]), 1, np.where(d.cause_of_death == 2, 2, 3)
            ),
            0,
        )
        d = (
            d[d.cycle <= "2013-2014"]
            .sort_values(["cycle", "participant_id"])
            .reset_index(drop=True)
        )
        specs.append(build(d, "cycle", "event", "clinical"))

    if any(s is None for s in specs):
        raise SystemExit("\nfold verification failed, nothing written")

    for s in specs:
        path = out_dir / f"{s['name']}.json"
        path.write_text(json.dumps(s))
        print(f"\nwrote {path}  ({len(s['temporal'])} folds per arm)")


if __name__ == "__main__":
    main()
