#!/usr/bin/env python3
"""
Normalize the medical data inside each fold.

Reads the table from 0_prepare_medical_data.py and the folds from 1_make_folds.py. For every
fold, the mean and standard deviation of each column in COLUMNS_TO_SCALE are computed on
the training rows only, then applied to the training and the test rows. The test rows
never influence the scaling.

Run it after the other two:  python src/2_normalize_medical.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# 1. Folders and settings (edit these)
# ----------------------------------------------------------------------------------------
PREPROCESSED_FILE = Path("data/medical/2_preprocessed/nhanes_preprocessed.csv")
FOLDS_FILE = Path("data/medical/3_folds/medical.json")
OUTPUT_DIR = Path("data/medical/4_normalized")

# Continuous columns to standardize; the 0/1 columns are left as they are
COLUMNS_TO_SCALE = ["age_years"]

# Name of the scaled column
SCALED_NAMES = {"age_years": "age_std"}

# Same sort as 1_make_folds.py, so the saved row positions point at the same people
GROUP_COLUMN = "cycle"
ID_COLUMN = "participant_id"


# ----------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------
def read_inputs() -> tuple:
    for path in (PREPROCESSED_FILE, FOLDS_FILE):
        if not path.exists():
            sys.exit(f"File not found: {path.resolve()}")

    df = pd.read_csv(PREPROCESSED_FILE)
    df = df.sort_values([GROUP_COLUMN, ID_COLUMN]).reset_index(drop=True)
    spec = json.loads(FOLDS_FILE.read_text())
    print(f"Read {len(df):,} rows from {PREPROCESSED_FILE}")
    print(f"Read folds from {FOLDS_FILE}")

    # The folds are row positions, so they only fit the table they were made from
    if len(df) != spec["n"]:
        sys.exit(
            f"The folds were made for {spec['n']:,} rows but the table has {len(df):,}. "
            "Run 1_make_folds.py again."
        )
    return df, spec


# ----------------------------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------------------------
def normalize_fold(df: pd.DataFrame, train_idx: list, test_idx: list) -> tuple:
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN])

    params = {}
    for col in COLUMNS_TO_SCALE:
        mean, sd = train[col].mean(), train[col].std()
        params[col] = {"mean": float(mean), "sd": float(sd)}
        for part in (train, test):
            part[col] = ((part[col] - mean) / sd).astype("float64")

    rename = {c: SCALED_NAMES[c] for c in COLUMNS_TO_SCALE}
    return train.rename(columns=rename), test.rename(columns=rename), params


def normalize_arm(df: pd.DataFrame, folds: list, arm: str) -> list:
    folder = OUTPUT_DIR / arm
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {arm}: {len(folds)} folds, saved in {folder} ---")

    summary, all_params = [], []
    for i, fold in enumerate(folds, start=1):
        train, test, params = normalize_fold(df, fold["train"], fold["test"])
        train.to_csv(folder / f"fold_{i}_train.csv", index=False)
        test.to_csv(folder / f"fold_{i}_test.csv", index=False)
        all_params.append({"fold": i, **params})

        row = {"fold": i, "n_train": len(train), "n_test": len(test)}
        for col in COLUMNS_TO_SCALE:
            new = SCALED_NAMES[col]
            row[f"{col}_mean"] = round(params[col]["mean"], 2)
            row[f"{col}_sd"] = round(params[col]["sd"], 2)
            row[f"train_{new}_mean"] = round(train[new].mean(), 3)
            row[f"train_{new}_sd"] = round(train[new].std(), 3)
            row[f"test_{new}_mean"] = round(test[new].mean(), 3)
        summary.append(row)
    print(pd.DataFrame(summary).to_string(index=False))
    return all_params


# ----------------------------------------------------------------------------------------
def main() -> None:
    df, spec = read_inputs()

    print(f"\nScaling {COLUMNS_TO_SCALE} with the mean and standard deviation of the")
    print("training rows of each fold only, then applying them to its test rows.")

    params = {}
    for arm in ("temporal", "random"):
        params[arm] = normalize_arm(df, spec[arm], arm)

    # Keep the numbers used, so the same scaling can be applied again later
    params_file = OUTPUT_DIR / "scaling.json"
    params_file.write_text(json.dumps(params, indent=1))
    print(f"\nSaved {params_file.resolve()}")


if __name__ == "__main__":
    main()
