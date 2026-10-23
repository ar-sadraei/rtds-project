#!/usr/bin/env python3
"""
Normalize the financial data inside each fold.

The medical counterpart is 2_normalize_medical.py and this script follows the same
design
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
COHORTS = (
    (
        "calm",
        Path("data/banking/1_merged/financial_calm_2013_2018.csv"),
        Path("data/banking/2_folds/financial-calm.json"),
    ),
    (
        "crisis",
        Path("data/banking/1_merged/financial_crisis_2005_2009.csv"),
        Path("data/banking/2_folds/financial-crisis.json"),
    ),
)
OUTPUT_DIR = Path("data/banking/3_normalized")

COLUMNS_TO_SCALE = [
    "fico",
    "cltv",
    "dti",
    "orig_upb",
    "orig_rate",
    "orig_term",
    "mi_pct",
    "n_units",
    "n_borrowers",
]


CATEGORY_LEVELS = {
    "first_time_buyer": ["N", "Y"],
    "occupancy": ["P", "I", "S"],
    "channel": ["R", "B", "C", "T"],
    "loan_purpose": ["P", "C", "N"],
    "property_type": ["SF", "PU", "CO", "MH", "CP"],
    "division": [
        "South Atlantic",
        "East North Central",
        "East South Central",
        "Middle Atlantic",
        "Mountain",
        "New England",
        "Pacific",
        "Territory",
        "West North Central",
        "West South Central",
    ],
}


MISSING_CODES = {
    "first_time_buyer": ["9"],
    "occupancy": ["9"],
    "channel": ["9"],
    "loan_purpose": ["9"],
    "property_type": ["99"],
}

PASSTHROUGH = ["loan_id", "vintage", "time", "event", "dti_missing"]

COLUMNS_TO_DROP = ["ltv", "state", "default_zb_alt"]

GROUP_COLUMN = "vintage"
ID_COLUMN = "loan_id"

DIVISIONS = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
    "Territory": ["GU", "PR", "VI"],
}
STATE_TO_DIVISION = {s: d for d, states in DIVISIONS.items() for s in states}

# ----------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------
def read_inputs(table: Path, folds_file: Path) -> tuple:
    for path in (table, folds_file):
        if not path.exists():
            sys.exit(f"File not found: {path.resolve()}")

    as_text = {c: str for c in CATEGORY_LEVELS if c != "division"}
    df = pd.read_csv(table, dtype=as_text)
    df = df.sort_values([GROUP_COLUMN, ID_COLUMN]).reset_index(drop=True)
    spec = json.loads(folds_file.read_text())
    print(f"Read {len(df):,} rows from {table}")
    print(f"Read folds from {folds_file}")

    if len(df) != spec["n"]:
        sys.exit(
            f"The folds were made for {spec['n']:,} rows but the table has {len(df):,}. "
            "Run 1_make_folds.py again."
        )

    unknown = set(df["state"]) - set(STATE_TO_DIVISION)
    if unknown:
        sys.exit(f"No Census division for {sorted(unknown)}; add them to DIVISIONS.")
    df["division"] = df["state"].map(STATE_TO_DIVISION)
    df = df.drop(columns=COLUMNS_TO_DROP)


    recoded = {}
    for col, levels in CATEGORY_LEVELS.items():
        if df[col].isna().any():
            sys.exit(f"{col} is blank on {int(df[col].isna().sum())} row(s); "
                     "decide how to code them before encoding.")
        codes = MISSING_CODES.get(col, [])
        n = int(df[col].isin(codes).sum())
        if n:
            recoded[col] = n
            df.loc[df[col].isin(codes), col] = levels[0]
        unexpected = set(df[col].dropna()) - set(levels)
        if unexpected:
            sys.exit(f"{col} holds undeclared level(s) {sorted(unexpected)}.")
    if recoded:
        print("  not-available codes folded into the reference level: " + ", ".join(
            f"{c} {n} row(s) -> {CATEGORY_LEVELS[c][0]}" for c, n in recoded.items()
        ))

    expected = set(PASSTHROUGH) | set(COLUMNS_TO_SCALE) | set(CATEGORY_LEVELS)
    missing, extra = expected - set(df.columns), set(df.columns) - expected
    if missing or extra:
        sys.exit(f"Column mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}.")
    return df, spec

# ----------------------------------------------------------------------------------------
# Per-fold preprocessing
# ----------------------------------------------------------------------------------------
def normalize_fold(df: pd.DataFrame, train_idx: list, test_idx: list) -> tuple:
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN])

    params: dict = {"impute": {}, "scale": {}, "encode": {}}
    notes = {"imputed": 0, "constant": [], "zero_variance": []}

    # 0_prepare_financial_data.py leaves dti, cltv and n_borrowers empty rather than
    # filling them, and says so under "NEVER IMPUTES". This is not a disagreement:
    # that script sees one cohort with no train/test boundary, so any median it
    # computed would run over rows that become test rows in every fold. "The training
    # rows" only exists once 1_make_folds.py has run, which makes this the first place
    # the fill can be done without leaking. dti keeps dti_missing as a feature; cltv
    # (9 calm loans) and n_borrowers (19 crisis) get no flag, too rare to carry one.
    for col in COLUMNS_TO_SCALE:
        n_missing = int(train[col].isna().sum() + test[col].isna().sum())
        if n_missing == 0:
            continue
        median = float(train[col].median())
        params["impute"][col] = median
        notes["imputed"] += n_missing
        for part in (train, test):
            part[col] = part[col].fillna(median)

    for col in COLUMNS_TO_SCALE:
        mean, sd = float(train[col].mean()), float(train[col].std())
        if sd == 0 or not np.isfinite(sd):
            notes["zero_variance"].append(col)
            sd = 1.0
        params["scale"][col] = {"mean": mean, "sd": sd}
        for part in (train, test):
            part[col] = ((part[col] - mean) / sd).astype("float64")

    train = train.rename(columns={c: f"{c}_std" for c in COLUMNS_TO_SCALE})
    test = test.rename(columns={c: f"{c}_std" for c in COLUMNS_TO_SCALE})

    blocks = {"train": [], "test": []}
    for col, levels in CATEGORY_LEVELS.items():
        reference, candidates = levels[0], levels[1:]
        kept = [lvl for lvl in candidates if (train[col] == lvl).any()]
        notes["constant"] += [f"{col}={lvl}" for lvl in candidates if lvl not in kept]
        params["encode"][col] = {"reference": reference, "columns": kept}
        for name, part in (("train", train), ("test", test)):
            blocks[name].append(
                pd.DataFrame(
                    {f"{col}={lvl}": (part[col] == lvl).astype("int8") for lvl in kept},
                    index=part.index,
                )
            )

    scaled = [f"{c}_std" for c in COLUMNS_TO_SCALE]
    train_out = pd.concat([train[PASSTHROUGH + scaled], *blocks["train"]], axis=1)
    test_out = pd.concat([test[PASSTHROUGH + scaled], *blocks["test"]], axis=1)
    test_out = test_out[train_out.columns]

    assert not train_out.isna().any().any(), "NaN left in the training rows"
    assert not test_out.isna().any().any(), "NaN left in the test rows"
    return train_out, test_out, params, notes


def normalize_arm(df: pd.DataFrame, folds: list, cohort: str, arm: str) -> list:
    folder = OUTPUT_DIR / cohort / arm
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {cohort} / {arm}: {len(folds)} folds, saved in {folder} ---")

    scaled = [f"{c}_std" for c in COLUMNS_TO_SCALE]
    summary, all_params, dropped = [], [], set()
    for i, fold in enumerate(folds, start=1):
        train, test, params, notes = normalize_fold(df, fold["train"], fold["test"])
        train.to_csv(folder / f"fold_{i}_train.csv", index=False)
        test.to_csv(folder / f"fold_{i}_test.csv", index=False)
        all_params.append({"fold": i, **params})
        dropped |= set(notes["constant"])


        summary.append(
            {
                "fold": i,
                "n_train": len(train),
                "n_test": len(test),
                "n_cols": train.shape[1],
                "train_mean": round(float(train[scaled].to_numpy().mean()), 4),
                "train_sd": round(float(np.mean([train[c].std() for c in scaled])), 4),
                "test_fico_mean": round(float(test["fico_std"].mean()), 3),
                "test_cltv_mean": round(float(test["cltv_std"].mean()), 3),
                "test_dti_mean": round(float(test["dti_std"].mean()), 3),
                "imputed": notes["imputed"],
            }
        )
        if notes["zero_variance"]:
            print(f"  fold {i}: no variance in the training rows for "
                  f"{notes['zero_variance']}, sd set to 1")
    print(pd.DataFrame(summary).to_string(index=False))
    if dropped:
        print(f"  no column for {sorted(dropped)}: absent from the training rows")


    per_fold = [
        tuple(sorted(col for c in CATEGORY_LEVELS for col in p["encode"][c]["columns"]))
        for p in all_params
    ]
    if len(set(per_fold)) > 1:
        sys.exit(f"  {cohort}/{arm}: the folds do not share one design matrix.")
    return all_params


# ----------------------------------------------------------------------------------------
def main() -> None:
    print(f"Scaling  {COLUMNS_TO_SCALE}")
    print(f"Encoding {list(CATEGORY_LEVELS)}")
    print("with the statistics of the training rows of each fold only, then applying")
    print("them to its test rows.\n")

    for cohort, table, folds_file in COHORTS:
        print("=" * 92)
        print(f"{cohort.upper()} COHORT")
        print("=" * 92)
        df, spec = read_inputs(table, folds_file)

        params = {}
        for arm in ("temporal", "random"):
            params[arm] = normalize_arm(df, spec[arm], cohort, arm)

        params_file = OUTPUT_DIR / cohort / "scaling.json"
        params_file.write_text(json.dumps(params, indent=1))
        print(f"\nSaved {params_file}\n")


if __name__ == "__main__":
    main()
