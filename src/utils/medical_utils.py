"""
Functions for the medical data (NHANES demography, smoking and linked mortality).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from . import fold_utils, report_utils

# ----------------------------------------------------------------------------------------
# 2. Columns to keep: original name -> new name. Everything else is dropped.
# ----------------------------------------------------------------------------------------
DEMOGRAPHY_COLUMNS = {
    "SEQN": "participant_id",
    # only used to build the 'cycle' column
    "SDDSRVYR": "cycle_code",
    "RIDAGEYR": "age_years",
    # 1 male, 2 female
    "RIAGENDR": "sex",
    # 1 Mexican American, 2 other Hispanic, 3 white, 4 black, 5 other
    "RIDRETH1": "race_ethnicity",
}

SMOKING_COLUMNS = {
    "SEQN": "participant_id",
    # 1 yes, 2 no, 7 refused, 9 don't know
    "SMQ020": "smoked_100_cigarettes",
    # 1 every day, 2 some days, 3 not at all
    "SMQ040": "smokes_now",
}

# Fixed-width positions (1-based, inclusive) in the mortality files.
MORTALITY_COLUMNS = {
    # original: (new name, first character, last character)
    "SEQN": ("participant_id", 1, 6),
    # 1 eligible, 2 under 18, 3 not linkable
    "ELIGSTAT": (
        "mortality_eligibility",
        15,
        15,
    ),
    # 0 assumed alive, 1 assumed dead
    "MORTSTAT": ("mortality_status", 16, 16),
    # 1 heart, 2 cancer, ..., 10 all other causes
    "UCOD_LEADING": (
        "cause_of_death",
        17,
        19,
    ),
    # months from the exam
    "PERMTH_EXM": ("followup_months", 46, 48),
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

# ----------------------------------------------------------------------------------------


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
            report_utils.log(f"Note: {path.name} has no {left_out}")
        parts.append(df[keep].rename(columns=columns))
    out = pd.concat(parts, ignore_index=True)
    report_utils.log(f"Read {folder.name}: {len(parts)} files, {len(out):,} rows")
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
    out = pd.concat(parts, ignore_index=True)
    report_utils.log(f"Read {folder.name}: {len(parts)} files, {len(out):,} rows")

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


# ----------------------------------------------------------------------------------------
# Merging and cleaning
# ----------------------------------------------------------------------------------------
def merge_and_clean(
    demography_dir: Path, smoking_dir: Path, mortality_dir: Path, min_age_years: int
) -> pd.DataFrame:
    demo = read_survey_folder(demography_dir, DEMOGRAPHY_COLUMNS)
    smoke = read_survey_folder(smoking_dir, SMOKING_COLUMNS)
    mort = read_mortality_folder(mortality_dir)

    # Demography is the base table
    df = demo.merge(mort, on="participant_id", how="left", validate="one_to_one")
    df = df.merge(smoke, on="participant_id", how="left", validate="one_to_one")
    report_utils.log(
        f"Merged {len(demo):,} participants. "
        f"Without mortality record: {(~demo['participant_id'].isin(mort['participant_id'])).sum():,}. "
        f"Without smoking record: {(~demo['participant_id'].isin(smoke['participant_id'])).sum():,}."
    )
    lost = (~mort["participant_id"].isin(demo["participant_id"])).sum()
    if lost:
        report_utils.log(
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
    df = df[df["age_years"] >= min_age_years].reset_index(drop=True)
    report_utils.log(
        f"Removed {n_before - len(df):,} participants under {min_age_years}, "
        f"{len(df):,} left."
    )

    # No mortality status: not linkable to death records
    n_before = len(df)
    df = df[df["mortality_status"].notna()].reset_index(drop=True)
    report_utils.log(
        f"Removed {n_before - len(df):,} participants without a mortality status, "
        f"{len(df):,} left."
    )

    n_before = len(df)
    df = df[df["smoked_100_cigarettes"].notna()].reset_index(drop=True)
    report_utils.log(
        f"Removed {n_before - len(df):,} participants without a "
        f"smoked_100_cigarettes answer, {len(df):,} left."
    )

    # Smoked 100 cigarettes but no answer on smoking now
    n_before = len(df)
    df = df[
        ~((df["smoked_100_cigarettes"] == 1) & df["smokes_now"].isna())
    ].reset_index(drop=True)
    report_utils.log(
        f"Removed {n_before - len(df):,} participants who smoked 100 cigarettes but have "
        f"no smokes_now answer, {len(df):,} left."
    )

    # Dead but no cause of death
    n_before = len(df)
    df = df[~((df["mortality_status"] == 1) & df["cause_of_death"].isna())].reset_index(
        drop=True
    )
    report_utils.log(
        f"Removed {n_before - len(df):,} participants who died without a cause of death, "
        f"{len(df):,} left."
    )

    # No follow-up time (no exam)
    n_before = len(df)
    df = df[df["followup_months"].notna()].reset_index(drop=True)
    report_utils.log(
        f"Removed {n_before - len(df):,} participants without a follow-up time, "
        f"{len(df):,} left."
    )

    # Refused or don't know in a smoking question
    n_before = len(df)
    unknown = df["smoked_100_cigarettes"].isin([7, 9]) | df["smokes_now"].isin([7, 9])
    df = df[~unknown].reset_index(drop=True)
    report_utils.log(
        f"Removed {n_before - len(df):,} participants who refused or did not know a "
        f"smoking answer, {len(df):,} left."
    )

    # Everyone is eligible now, so the column is not needed
    assert (df["mortality_eligibility"] == 1).all()
    df = df.drop(columns="mortality_eligibility")

    df = whole_numbers_as_int(df)

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        report_utils.log(f"Dropping columns with no data at all: {empty}")
        df = df.drop(columns=empty)
    return df


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
# Recoding
# ----------------------------------------------------------------------------------------
def keep_cycles(df: pd.DataFrame, cycles: list) -> pd.DataFrame:
    """
    Keep only the listed survey cycles.

    Mortality is linked through 2019, so a participant interviewed in 2015 to 2018 has
    at most 61 months of follow-up and none of them can be observed at 72 months. Their
    censoring distribution is zero at that horizon, which makes every
    inverse-probability-of-censoring weight undefined. They also leak into the kfold
    and matched arms, whose folds sample the whole table, so the arm comparison would be confounded
    with how long people were observed rather than with the assignment rule.
    """
    before = len(df)
    out = df[df[GROUP_COLUMN].isin(cycles)].reset_index(drop=True)
    dropped = sorted(set(df[GROUP_COLUMN]) - set(out[GROUP_COLUMN]))
    report_utils.log(
        f"Removed {before - len(out):,} participants from cycles outside the list "
        f"({', '.join(dropped) if dropped else 'none'}), {len(out):,} left."
    )
    return out


def add_labels(df: pd.DataFrame, follow_up_months: int) -> pd.DataFrame:
    """Add the time and event columns, cutting follow-up at the given months (0 keeps all)."""
    out = df.copy()
    cut = follow_up_months if follow_up_months > 0 else float("inf")

    # event: 0 censored, 1 heart disease, 2 cancer, 3 other
    died = df["mortality_status"] == 1
    cause = df["cause_of_death"].map(lambda c: 1 if c == 1 else 2 if c == 2 else 3)
    died_in_window = died & (df["followup_months"] <= cut)
    out["time"] = df["followup_months"].clip(upper=cut)
    out["event"] = cause.where(died_in_window, 0).astype(int)
    report_utils.log(
        f"time and event: follow-up cut at "
        f"{f'{follow_up_months} months' if follow_up_months else '(full)'}; "
        f"{(died & ~died_in_window).sum():,} deaths after that count as censored"
    )
    return out


def recode(df: pd.DataFrame) -> pd.DataFrame:
    """Recode the columns for the models. Fixed rules only, nothing is learned."""
    out = df[["participant_id", "cycle"]].copy()

    out["sex_female"] = (df["sex"] == 2).astype(int)

    # White is the reference group
    race_columns = {
        1: "race_mexican_american",
        2: "race_other_hispanic",
        4: "race_black",
        5: "race_other",
    }
    for code, name in race_columns.items():
        out[name] = (df["race_ethnicity"] == code).astype(int)

    # One smoking status; never-smokers are not asked the second question
    status = pd.Series("never", index=df.index)
    ever = df["smoked_100_cigarettes"] == 1
    status[ever & (df["smokes_now"] == 3)] = "former"
    status[ever & (df["smokes_now"] == 2)] = "current_some_days"
    status[ever & (df["smokes_now"] == 1)] = "current_every_day"
    for name in ["former", "current_some_days", "current_every_day"]:
        out[f"smoking_{name}"] = (status == name).astype(int)
    report_utils.log(
        "Recoded sex, race_ethnicity and smoking into 0/1 columns (references: male, "
        "non-Hispanic white, never smoked)"
    )

    # Age is scaled later, inside each fold
    out["age_years"] = df["age_years"]

    out["time"] = df["time"]
    out["event"] = df["event"]
    return out


# ----------------------------------------------------------------------------------------
# Normalizing
# ----------------------------------------------------------------------------------------
# Continuous columns to standardize; the 0/1 columns are left as they are
COLUMNS_TO_SCALE = ["age_years"]

# Name of the scaled column
SCALED_NAMES = {"age_years": "age_std"}

# Same sort as the folds, so the saved row positions point at the same people
GROUP_COLUMN = "cycle"
ID_COLUMN = "participant_id"


# ----------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------
def read_inputs(table_file: Path, folds_file: Path) -> tuple:
    for path in (table_file, folds_file):
        if not path.exists():
            sys.exit(f"File not found: {path.resolve()}")

    df = pd.read_csv(table_file)
    df = df.sort_values([GROUP_COLUMN, ID_COLUMN]).reset_index(drop=True)
    spec = json.loads(folds_file.read_text())

    # The folds are row positions, so they only fit the table they were made from
    if len(df) != spec["n"]:
        sys.exit(
            f"The folds were made for {spec['n']:,} rows but the table has {len(df):,}. "
            "Run the folds step again."
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


def normalize_arm(df: pd.DataFrame, folds: list, arm: str, output_dir: Path) -> list:
    folder = output_dir / arm
    folder.mkdir(parents=True, exist_ok=True)

    all_params = []
    for i, fold in enumerate(folds, start=1):
        train, test, params = normalize_fold(df, fold["train"], fold["test"])
        train.to_csv(folder / f"fold_{i}_train.csv", index=False)
        test.to_csv(folder / f"fold_{i}_test.csv", index=False)
        all_params.append({"fold": i, **params})
    return all_params


# ----------------------------------------------------------------------------------------


def normalize(table_file: Path, folds_file: Path, output_dir: Path) -> None:
    """Recode the table, scale each fold with its own training rows and save it."""
    df, spec = read_inputs(table_file, folds_file)
    df = recode(df)

    report_utils.log(
        f"Scaled {COLUMNS_TO_SCALE} inside each fold, from its training rows only"
    )

    params = {}
    for arm in fold_utils.ARMS:
        params[arm] = normalize_arm(df, spec[arm], arm, output_dir)

    # Keep the numbers used, so the same scaling can be applied again later
    params_file = output_dir / "scaling.json"
    params_file.write_text(json.dumps(params, indent=1))
    report_utils.log(f"Saved {output_dir} (scaling.json holds the statistics used)")
