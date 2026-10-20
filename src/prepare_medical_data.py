#!/usr/bin/env python3
"""
Merge NHANES demography, smoking and linked mortality (2019) files into one table
and print a preview. These people are removed: under MIN_AGE_YEARS, without a mortality
status, without a smoked_100_cigarettes answer, smokers without a smokes_now answer, and
people who died without a cause of death.
No other filtering and no splitting is done here.

Edit the folders and column lists below, then run:  python src/preprocessing/merge_medical.py
File names do not matter. Every .xpt (or .xpt.txt) file in a survey folder and every
.dat file in the mortality folder is read, and the cycle comes from the data itself.
"""

import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------------------
# 1. Folders (edit these)
# ----------------------------------------------------------------------------------------
DEMOGRAPHY_DIR = Path("data/medical/0_raw/demography")
SMOKING_DIR = Path("data/medical/0_raw/smoking")
MORTALITY_DIR = Path("data/medical/0_raw/mortality")
OUTPUT_FILE = Path("data/medical/1_merged/nhanes_merged.csv")

# Linked mortality exists only for adults (ELIGSTAT 2 = under 18), so children have no
# event or follow-up time and cannot be used in a survival model.
MIN_AGE_YEARS = 20

# ----------------------------------------------------------------------------------------
# 2. Columns to keep: original name -> new name. Everything else is dropped.
# ----------------------------------------------------------------------------------------
DEMOGRAPHY_COLUMNS = {
    "SEQN": "participant_id",
    "SDDSRVYR": "cycle_code",  # only used to build the 'cycle' column
    "RIDAGEYR": "age_years",
    "RIAGENDR": "sex",  # 1 male, 2 female
    "RIDRETH1": "race_ethnicity",  # 1 Mexican American, 2 other Hispanic, 3 white, 4 black, 5 other
}

SMOKING_COLUMNS = {
    "SEQN": "participant_id",
    "SMQ020": "smoked_100_cigarettes",  # 1 yes, 2 no, 7 refused, 9 don't know
    "SMQ040": "smokes_now",  # 1 every day, 2 some days, 3 not at all
}

# Fixed-width positions (1-based, inclusive) in the mortality files.
MORTALITY_COLUMNS = {
    # original: (new name, first character, last character)
    "SEQN": ("participant_id", 1, 6),
    "ELIGSTAT": (
        "mortality_eligibility",
        15,
        15,
    ),  # 1 eligible, 2 under 18, 3 not linkable
    "MORTSTAT": ("mortality_status", 16, 16),  # 0 assumed alive, 1 assumed dead
    "UCOD_LEADING": (
        "cause_of_death",
        17,
        19,
    ),  # 1 heart, 2 cancer, ..., 10 all other causes
    "PERMTH_EXM": ("followup_months", 46, 48),  # months from the exam
}

# Meaning of the survey code in the demography files (SDDSRVYR)
CYCLE_LABELS = {
    1: "1999-2000",
    2: "2001-2002",
    3: "2003-2004",
    4: "2005-2006",
    5: "2007-2008",
    6: "2009-2010",
    7: "2011-2012",
    8: "2013-2014",
    9: "2015-2016",
    10: "2017-2018",
}

# Meaning of the coded columns, printed at the end of the merge
VALUE_LABELS = {
    "sex": {1: "male", 2: "female"},
    "race_ethnicity": {
        1: "Mexican American",
        2: "other Hispanic",
        3: "non-Hispanic white",
        4: "non-Hispanic black",
        5: "other",
    },
    "mortality_status": {0: "assumed alive", 1: "assumed dead"},
    "cause_of_death": {
        1: "heart disease",
        2: "cancer",
        3: "chronic lower respiratory diseases",
        4: "accidents",
        5: "cerebrovascular diseases",
        6: "Alzheimer's disease",
        7: "diabetes",
        8: "influenza and pneumonia",
        9: "kidney diseases",
        10: "all other causes",
    },
    "smoked_100_cigarettes": {1: "yes", 2: "no", 7: "refused", 9: "don't know"},
    "smokes_now": {
        1: "every day",
        2: "some days",
        3: "not at all",
        7: "refused",
        9: "don't know",
    },
}


# ----------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------
def files_in(folder: Path, endings: tuple) -> list:
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder.resolve()}")
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(endings)
    )
    if not files:
        sys.exit(f"No {endings} files found in {folder.resolve()}")
    return files


def check_unique(df: pd.DataFrame, folder: Path) -> None:
    dup = df["participant_id"].duplicated().sum()
    if dup:
        sys.exit(
            f"{dup} repeated participant ids after reading {folder}. "
            "Files from another survey (NHIS, NHANES III) or a repeated cycle are in there."
        )


def read_survey_folder(folder: Path, columns: dict) -> pd.DataFrame:
    parts = []
    for path in files_in(folder, (".xpt", ".xpt.txt")):
        df = pd.read_sas(path, format="xport", encoding="latin-1")
        df.columns = [c.upper() for c in df.columns]
        keep = [c for c in columns if c in df.columns]
        left_out = [c for c in columns if c not in df.columns]
        if "SEQN" not in keep:
            sys.exit(f"{path.name} has no SEQN column, is it a real XPT data file?")
        if left_out:
            print(f"  note: {path.name} has no {left_out}")
        parts.append(df[keep].rename(columns=columns))
        print(f"  {folder.name}/{path.name}: {len(df):,} rows")
    out = pd.concat(parts, ignore_index=True)
    out["participant_id"] = out["participant_id"].astype("int64")
    check_unique(out, folder)
    return out


def read_mortality_folder(folder: Path) -> pd.DataFrame:
    names = [new for new, _, _ in MORTALITY_COLUMNS.values()]
    colspecs = [(a - 1, b) for _, a, b in MORTALITY_COLUMNS.values()]
    parts = []
    for path in files_in(folder, (".dat",)):
        df = pd.read_fwf(
            path,
            colspecs=colspecs,
            names=names,
            header=None,
            dtype=str,
            na_values=[".", ""],
        )
        for c in names:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        parts.append(df)
        print(f"  {folder.name}/{path.name}: {len(df):,} rows")
    out = pd.concat(parts, ignore_index=True)

    unusual = (
        out["participant_id"].isna()
        | ~out["mortality_eligibility"].isin([1, 2, 3])
        | (~out["mortality_status"].isin([0, 1]) & out["mortality_status"].notna())
        | (~out["cause_of_death"].between(1, 10) & out["cause_of_death"].notna())
        | (out["followup_months"] > 400)
    ).mean()
    if unusual > 0.01:
        sys.exit(
            f"The mortality columns look wrong ({unusual:.1%} unusual rows). "
            "Check the positions in MORTALITY_COLUMNS and that the folder only has NHANES files."
        )
    out["participant_id"] = out["participant_id"].astype("int64")
    check_unique(out, folder)
    return out


def whole_numbers_as_int(df: pd.DataFrame) -> pd.DataFrame:
    """77.0 becomes 77 and missing becomes <NA>, for columns that only hold whole numbers."""
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) and (df[c].dropna() % 1 == 0).all():
            df[c] = df[c].astype("Int64")
    return df


# ----------------------------------------------------------------------------------------
# Preview
# ----------------------------------------------------------------------------------------
def show_preview(df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Columns and value types ===")
    print(df.dtypes.to_string())

    print("\n=== 5 random rows ===")
    print(df.sample(5))

    print(f"\n=== Total entries: {len(df):,} ===")

    print("\n=== Entries per cycle ===")
    print(df["cycle"].value_counts().sort_index().to_string())

    print("\n=== Missing values per column ===")
    missing = df.isna().sum()
    print(
        pd.DataFrame(
            {"missing": missing, "missing_%": (missing / len(df) * 100).round(1)}
        ).to_string()
    )

    print("\n=== Values of the coded columns ===")
    for col, labels in VALUE_LABELS.items():
        counts = df[col].value_counts().sort_index()
        print(f"\n{col}")
        for code, n in zip(counts.index.to_list(), counts.to_list()):
            label = labels.get(code, "??? not in VALUE_LABELS")
            print(f"  {code:>3}  {label:<36} {n:>7,}")


# ----------------------------------------------------------------------------------------
def main() -> None:
    print("Reading demography files")
    demo = read_survey_folder(DEMOGRAPHY_DIR, DEMOGRAPHY_COLUMNS)
    print("Reading smoking files")
    smoke = read_survey_folder(SMOKING_DIR, SMOKING_COLUMNS)
    print("Reading mortality files")
    mort = read_mortality_folder(MORTALITY_DIR)

    # Demography is the base table, so every survey participant stays in the result.
    df = demo.merge(mort, on="participant_id", how="left", validate="one_to_one")
    df = df.merge(smoke, on="participant_id", how="left", validate="one_to_one")
    print(
        f"\nMerged {len(demo):,} participants. "
        f"Without mortality record: {(~demo['participant_id'].isin(mort['participant_id'])).sum():,}. "
        f"Without smoking record: {(~demo['participant_id'].isin(smoke['participant_id'])).sum():,}."
    )
    lost = (~mort["participant_id"].isin(demo["participant_id"])).sum()
    if lost:
        print(
            f"WARNING: {lost:,} mortality rows have no demography row and are not in the result."
        )

    # cycle column, built from the survey code inside the demography files
    df["cycle"] = df["cycle_code"].map(CYCLE_LABELS)
    df["cycle"] = df["cycle"].fillna(
        "unknown (code " + df["cycle_code"].astype(str) + ")"
    )
    df = df.drop(columns="cycle_code")
    df = df[
        ["participant_id", "cycle"]
        + [c for c in df.columns if c not in ("participant_id", "cycle")]
    ]

    n_before = len(df)
    df = df[df["age_years"] >= MIN_AGE_YEARS].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants under {MIN_AGE_YEARS}, "
        f"{len(df):,} left."
    )

    # No mortality status means NCHS could not link the person to death records
    # (mortality_eligibility 3), so we do not know if they are alive or dead.
    n_before = len(df)
    df = df[df["mortality_status"].notna()].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants without a mortality status, "
        f"{len(df):,} left."
    )

    n_before = len(df)
    df = df[df["smoked_100_cigarettes"].notna()].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants without a "
        f"smoked_100_cigarettes answer, {len(df):,} left."
    )

    # Smoked 100 cigarettes but no answer on smoking now (never-smokers are not asked, so
    # they are fine).
    n_before = len(df)
    df = df[
        ~((df["smoked_100_cigarettes"] == 1) & df["smokes_now"].isna())
    ].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants who smoked 100 cigarettes but have "
        f"no smokes_now answer, {len(df):,} left."
    )

    # Dead but no cause listed: we cannot tell which competing event it was.
    n_before = len(df)
    df = df[~((df["mortality_status"] == 1) & df["cause_of_death"].isna())].reset_index(
        drop=True
    )
    print(
        f"Removed {n_before - len(df):,} participants who died without a cause of death, "
        f"{len(df):,} left."
    )

    # After the filters above everyone is eligible (1), so the column carries no information.
    assert (df["mortality_eligibility"] == 1).all()
    df = df.drop(columns="mortality_eligibility")

    df = whole_numbers_as_int(df)

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        print(f"Dropping columns with no data at all: {empty}")
        df = df.drop(columns=empty)

    show_preview(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
