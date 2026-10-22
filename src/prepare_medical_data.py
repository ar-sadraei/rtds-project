#!/usr/bin/env python3
"""
Merge NHANES demography, smoking and linked mortality (2019) files into one table
and print a preview. These people are removed: under MIN_AGE_YEARS, without a mortality
status, without a smoked_100_cigarettes answer, smokers without a smokes_now answer, and
people who died without a cause of death, without a follow-up time, and with a refused
or "don't know" smoking answer.
Then the columns are recoded and the time and event labels are built (follow-up cut at
HORIZON_MONTHS). The age is not scaled here, it is scaled inside each fold.

Edit the folders and column lists below, then run:  python src/prepare_medical_data.py
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
MERGED_FILE = Path("data/medical/1_merged/nhanes_merged.csv")
# Recoded, age not scaled yet
PREPROCESSED_FILE = Path("data/medical/2_preprocessed/nhanes_preprocessed.csv")

# Mortality is only linked for adults
MIN_AGE_YEARS = 20

# Cut follow-up at this many months
HORIZON_MONTHS = 72

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
# Recoding
# ----------------------------------------------------------------------------------------
def recode(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Recoding ===")
    out = df[["participant_id", "cycle"]].copy()

    out["sex_female"] = (df["sex"] == 2).astype(int)
    print(
        f"sex: 1 male / 2 female -> sex_female 0/1 ({out['sex_female'].sum():,} female)"
    )

    # White is the reference group
    race_columns = {
        1: "race_mexican_american",
        2: "race_other_hispanic",
        4: "race_black",
        5: "race_other",
    }
    for code, name in race_columns.items():
        out[name] = (df["race_ethnicity"] == code).astype(int)
    print(
        "race_ethnicity: one-hot into "
        f"{', '.join(race_columns.values())} (reference: non-Hispanic white)"
    )

    # One smoking status; never-smokers are not asked the second question
    status = pd.Series("never", index=df.index)
    ever = df["smoked_100_cigarettes"] == 1
    status[ever & (df["smokes_now"] == 3)] = "former"
    status[ever & (df["smokes_now"] == 2)] = "current_some_days"
    status[ever & (df["smokes_now"] == 1)] = "current_every_day"
    for name in ["former", "current_some_days", "current_every_day"]:
        out[f"smoking_{name}"] = (status == name).astype(int)
    print(
        "smoked_100_cigarettes + smokes_now: combined into a smoking status, then one-hot "
        "(reference: never)"
    )
    print(status.value_counts().to_string())

    # Age is scaled later, inside each fold
    out["age_years"] = df["age_years"]

    # event: 0 censored, 1 heart disease or stroke, 2 cancer, 3 other
    died = df["mortality_status"] == 1
    cause = df["cause_of_death"].map(lambda c: 1 if c in (1, 5) else 2 if c == 2 else 3)
    died_in_window = died & (df["followup_months"] <= HORIZON_MONTHS)
    out["time"] = df["followup_months"].clip(upper=HORIZON_MONTHS)
    out["event"] = cause.where(died_in_window, 0).astype(int)
    print(
        f"time and event: follow-up cut at {HORIZON_MONTHS} months; "
        f"{(died & ~died_in_window).sum():,} deaths after that count as censored"
    )
    print(
        "event: 0 censored, 1 heart disease or stroke, 2 cancer, 3 all other causes\n"
        + out["event"].value_counts().sort_index().to_string()
    )
    return out


# ----------------------------------------------------------------------------------------
def main() -> None:
    print("Reading demography files")
    demo = read_survey_folder(DEMOGRAPHY_DIR, DEMOGRAPHY_COLUMNS)
    print("Reading smoking files")
    smoke = read_survey_folder(SMOKING_DIR, SMOKING_COLUMNS)
    print("Reading mortality files")
    mort = read_mortality_folder(MORTALITY_DIR)

    # Demography is the base table
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

    # Build the cycle column
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

    # No mortality status: not linkable to death records
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

    # Smoked 100 cigarettes but no answer on smoking now
    n_before = len(df)
    df = df[
        ~((df["smoked_100_cigarettes"] == 1) & df["smokes_now"].isna())
    ].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants who smoked 100 cigarettes but have "
        f"no smokes_now answer, {len(df):,} left."
    )

    # Dead but no cause of death
    n_before = len(df)
    df = df[~((df["mortality_status"] == 1) & df["cause_of_death"].isna())].reset_index(
        drop=True
    )
    print(
        f"Removed {n_before - len(df):,} participants who died without a cause of death, "
        f"{len(df):,} left."
    )

    # No follow-up time (no exam)
    n_before = len(df)
    df = df[df["followup_months"].notna()].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants without a follow-up time, "
        f"{len(df):,} left."
    )

    # Refused or don't know in a smoking question
    n_before = len(df)
    unknown = df["smoked_100_cigarettes"].isin([7, 9]) | df["smokes_now"].isin([7, 9])
    df = df[~unknown].reset_index(drop=True)
    print(
        f"Removed {n_before - len(df):,} participants who refused or did not know a "
        f"smoking answer, {len(df):,} left."
    )

    # Everyone is eligible now, so the column is not needed
    assert (df["mortality_eligibility"] == 1).all()
    df = df.drop(columns="mortality_eligibility")

    df = whole_numbers_as_int(df)

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        print(f"Dropping columns with no data at all: {empty}")
        df = df.drop(columns=empty)

    show_preview(df)

    df.to_csv(MERGED_FILE, index=False)
    print(f"\nSaved {MERGED_FILE.resolve()}")

    recoded = recode(df)
    PREPROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    recoded.to_csv(PREPROCESSED_FILE, index=False)
    print(f"\nSaved {PREPROCESSED_FILE.resolve()}")


if __name__ == "__main__":
    main()
