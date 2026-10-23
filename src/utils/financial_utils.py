"""
Functions for the financial data (Freddie Mac single-family loans).
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# 2. File layout and columns
# ----------------------------------------------------------------------------------------
# Asserted at parse time: if Freddie Mac changes the layout, fail loudly.
N_ORIG_FIELDS = 31
N_PERF_FIELDS = 35

# Origination file: 0-based field index -> column name. These are the fields we
# keep; everything else in the 31 is dropped.
ORIG_FIELDS = {
    0: "fico",  # credit score at origination
    2: "first_time_buyer",
    5: "mi_pct",  # mortgage insurance coverage %
    6: "n_units",
    7: "occupancy",  # P owner-occupied, S second home, I investment
    8: "cltv",  # combined loan-to-value
    9: "dti",  # debt-to-income
    10: "orig_upb",  # original unpaid principal balance
    11: "ltv",  # loan-to-value
    12: "orig_rate",  # interest rate
    13: "channel",  # R retail, B broker, C correspondent
    16: "state",
    17: "property_type",
    19: "loan_id",
    20: "loan_purpose",  # P purchase, C cash-out refi, N no-cash-out refi
    21: "orig_term",  # months
    22: "n_borrowers",
}

# Performance file: 0-based field indices we read while streaming.
P_LOAN_ID, P_PERIOD, P_DELINQ, P_AGE, P_ZB_CODE = 0, 1, 3, 4, 8

DELINQ_DEFAULT = 3  # months past due that counts as default
ZB_PREPAID = "01"  # zero balance code: prepaid or matured
ZB_CREDIT = {"02", "03", "09", "15"}  # disposition codes meaning a credit loss

# ----------------------------------------------------------------------------------------
# 3. Cleaning rules
# ----------------------------------------------------------------------------------------
# sentinel: the value Freddie Mac writes to mean "not available"
# lo, hi:   the range outside which a value is impossible, not merely unusual
NUMERIC_RULES = {
    "fico": {"sentinel": {"9999"}, "lo": 300, "hi": 850},
    "cltv": {"sentinel": {"999"}, "lo": 1, "hi": 200},
    "dti": {"sentinel": {"999"}, "lo": 1, "hi": 65},
    "ltv": {"sentinel": {"999"}, "lo": 1, "hi": 200},
    "mi_pct": {"sentinel": {"999"}, "lo": 0, "hi": 55},
    "orig_upb": {"sentinel": set(), "lo": 1000, "hi": 10_000_000},
    "orig_rate": {"sentinel": set(), "lo": 0.1, "hi": 15.0},
    "orig_term": {"sentinel": set(), "lo": 1, "hi": 480},
    "n_units": {"sentinel": {"99"}, "lo": 1, "hi": 4},
    "n_borrowers": {"sentinel": {"99"}, "lo": 1, "hi": 10},
}

# A loan missing any of these is dropped: they are the core predictors, they are
# missing in well under 1% of loans, and imputing them would invent the signal we
# are trying to measure.
#
# DTI IS DELIBERATELY NOT IN THIS LIST. Freddie Mac did not report it
# consistently in the earlier years: it is missing for 28.8% of 2013 loans and
# 1.0% of 2018 loans. Dropping those rows would filter the early vintages far
# harder than the late ones, which would confound exactly the comparison this
# dataset exists to support -- training on early years and testing on later
# ones. Instead the row is kept, `dti` is left empty, and `dti_missing` records
# it. See --drop-missing-dti to override.
REQUIRED = ["fico", "ltv", "orig_upb", "orig_rate", "orig_term"]

# Meaning of the coded columns, printed at the end of the run.
CODE_LABELS = {
    "first_time_buyer": {"Y": "yes", "N": "no", "9": "not available"},
    "occupancy": {
        "P": "owner occupied",
        "S": "second home",
        "I": "investment",
        "9": "not available",
    },
    "channel": {
        "R": "retail",
        "B": "broker",
        "C": "correspondent",
        "T": "third party origination",
        "9": "not available",
    },
    "loan_purpose": {
        "P": "purchase",
        "C": "cash-out refinance",
        "N": "no cash-out refinance",
        "9": "not available",
    },
    "property_type": {
        "SF": "single family",
        "CO": "condo",
        "PU": "planned unit",
        "MH": "manufactured housing",
        "CP": "co-op",
        "99": "not available",
    },
    "dti_missing": {0: "dti reported", 1: "dti not reported"},
}

# Column order in the output CSV.
OUTPUT_COLS = [
    "loan_id",
    "vintage",
    "time",
    "event",
    "default_zb_alt",
    "fico",
    "ltv",
    "cltv",
    "dti",
    "dti_missing",
    "orig_upb",
    "orig_rate",
    "orig_term",
    "mi_pct",
    "n_units",
    "n_borrowers",
    "first_time_buyer",
    "occupancy",
    "channel",
    "loan_purpose",
    "property_type",
    "state",
]


def delinquency_to_months(raw: str) -> int | None:
    """
    Convert a delinquency status code to months past due.

    The field is mostly numeric ("0", "1", "2", ...) but carries two special
    codes: "RA" means the property was repossessed, which is as deep a default
    as exists, and "XX" means unknown. Returning None for unknown means the
    caller will not count it as a default -- we do not guess.
    """
    raw = raw.strip()
    if not raw or raw == "XX":
        return None
    if raw == "RA":
        return 99
    try:
        return int(raw)
    except ValueError:
        return None


def read_origination(zf: zipfile.ZipFile, year: int) -> pd.DataFrame:
    """Read the origination file. Small enough to load whole (one row per loan)."""
    names = [ORIG_FIELDS.get(i, f"_unused_{i}") for i in range(N_ORIG_FIELDS)]
    with zf.open(f"sample_orig_{year}.txt") as fh:
        text = io.TextIOWrapper(fh, encoding="latin-1")
        first = text.readline()
        n_found = len(first.rstrip("\n").split("|"))
        if n_found != N_ORIG_FIELDS:
            raise ValueError(
                f"{year} origination file has {n_found} fields, expected "
                f"{N_ORIG_FIELDS}. The file layout has changed -- re-check "
                f"ORIG_FIELDS against the official layout before continuing."
            )

    with zf.open(f"sample_orig_{year}.txt") as fh:
        return pd.read_csv(
            io.TextIOWrapper(fh, encoding="latin-1"),
            sep="|",
            header=None,
            names=names,
            usecols=list(ORIG_FIELDS.values()),
            dtype=str,
            keep_default_na=False,
        )


def scan_performance(zf: zipfile.ZipFile, year: int) -> tuple[dict, int]:
    """
    Stream the monthly performance file and reduce it to one record per loan.

    This file is ~450 MB uncompressed per vintage and holds one row per loan per
    month. We never hold it in memory: we read it line by line and keep only a
    handful of numbers per loan.

    WHY THIS TRACKS CALENDAR MONTHS AND NOT THE `age` FIELD
    -------------------------------------------------------
    The obvious thing is to read the Loan Age column and take the smallest age
    at which something happened. That is wrong, and quietly so.

    When a loan is modified -- re-amortised after a long delinquency -- Freddie
    Mac RESETS the age counter. The reporting period keeps increasing but the
    age field jumps backwards. About 1% of loans per vintage do this. Trusting
    the age field on such a loan turns a prepayment in month 70 into a
    prepayment in month 9, because month 9 is what the post-modification row
    says. The error always points the same way: events look EARLIER than they
    were. Measured on the 2013 and 2018 vintages, this moved 0.18% of loans,
    and cost 4.4% of all defaults -- they were being recorded as early
    prepayments instead.

    So we record each event in absolute calendar months, together with the
    loan's own anchor -- the earliest period we see for it, and the age it was
    at that period. resolve_events() converts back to loan age at the end:

        loan_age = anchor_age + (event_month - anchor_month)

    which is monotonic by construction. Note the anchor age is not always zero:
    13% of loans first appear at age 1 or later, so "months since we first saw
    it" is not the same as "loan age" and cannot be substituted for it.

    Returns the per-loan dict and the number of malformed lines skipped.
    """
    loans: dict[str, dict] = {}
    malformed = 0

    with zf.open(f"sample_perf_{year}.txt") as fh:
        for line in io.TextIOWrapper(fh, encoding="latin-1"):
            fields = line.rstrip("\n").split("|")
            if len(fields) != N_PERF_FIELDS:
                malformed += 1
                continue

            loan_id = fields[P_LOAN_ID]
            try:
                age = int(fields[P_AGE])
                period = int(fields[P_PERIOD])  # YYYYMM
            except ValueError:
                malformed += 1
                continue

            month = (period // 100) * 12 + (period % 100)  # absolute month

            rec = loans.get(loan_id)
            if rec is None:
                rec = loans[loan_id] = {
                    "anchor_month": month,
                    "anchor_age": age,
                    "last_month": month,
                    "default_month": None,
                    "prepaid_month": None,
                    "zb_credit": False,
                }
            else:
                # Do not assume the file is sorted. Re-anchor on the earliest
                # period actually seen; everything is converted at the end, so a
                # late re-anchor corrects all of this loan's times at once.
                if month < rec["anchor_month"]:
                    rec["anchor_month"], rec["anchor_age"] = month, age
                if month > rec["last_month"]:
                    rec["last_month"] = month

            months_late = delinquency_to_months(fields[P_DELINQ])
            if months_late is not None and months_late >= DELINQ_DEFAULT:
                if rec["default_month"] is None or month < rec["default_month"]:
                    rec["default_month"] = month

            zb = fields[P_ZB_CODE].strip()
            if zb == ZB_PREPAID:
                if rec["prepaid_month"] is None or month < rec["prepaid_month"]:
                    rec["prepaid_month"] = month
            elif zb in ZB_CREDIT:
                rec["zb_credit"] = True

    return loans, malformed


def resolve_events(loans: dict, max_follow_up: int) -> pd.DataFrame:
    """
    Turn each loan's observation record into a single (time, event) pair.

    Absolute calendar months become loan age here, via the loan's own anchor.
    See scan_performance() for why the raw age field is not used directly.

    Whichever event happened FIRST is the one observed; the other is then
    impossible. If neither happened, the loan is censored when we last saw it.
    """
    rows = []
    for loan_id, r in loans.items():
        m0, a0 = r["anchor_month"], r["anchor_age"]

        def to_age(month: int) -> int:
            return a0 + (month - m0)

        default_age = None if r["default_month"] is None else to_age(r["default_month"])
        prepaid_age = None if r["prepaid_month"] is None else to_age(r["prepaid_month"])
        last = to_age(r["last_month"])

        if default_age is not None and (
            prepaid_age is None or default_age <= prepaid_age
        ):
            time, event = default_age, 1
        elif prepaid_age is not None:
            time, event = prepaid_age, 2
        else:
            time, event = last, 0

        # Administrative censoring: if the event falls beyond the window we
        # chose to observe, then as far as this dataset is concerned it had not
        # happened yet.
        if max_follow_up > 0 and time > max_follow_up:
            time, event = max_follow_up, 0

        rows.append((loan_id, time, event, int(r["zb_credit"])))

    return pd.DataFrame(rows, columns=["loan_id", "time", "event", "default_zb_alt"])


def parse_vintage(raw_dir: Path, year: int, max_follow_up: int) -> pd.DataFrame:
    """Build the one-row-per-loan table for a single origination year."""
    zip_path = raw_dir / f"sample_{year}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Download the {year} sample file from "
            f"https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset "
            f"and put the zip in {raw_dir}/ without renaming it."
        )

    with zipfile.ZipFile(zip_path) as zf:
        orig = read_origination(zf, year)
        loans, malformed = scan_performance(zf, year)

    events = resolve_events(loans, max_follow_up)

    # inner join: a loan must appear in BOTH files to be usable.
    # validate="one_to_one" makes a duplicated loan_id an error rather than a
    # silent row explosion.
    df = orig.merge(events, on="loan_id", how="inner", validate="one_to_one")
    df.insert(1, "vintage", year)

    if malformed:
        print(f"    skipped {malformed:,} malformed performance rows")
    only_orig = len(orig) - len(df)
    if only_orig:
        print(f"    dropped {only_orig:,} loans with no performance records")

    return df


def clean(
    df: pd.DataFrame, drop_missing_dti: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply the cleaning rules and return the cleaned frame plus a written report.

    Order matters: sentinels become missing BEFORE range checks, otherwise 9999
    would be flagged as an out-of-range credit score rather than as "not
    available", and the report would be misleading.
    """
    report: list[str] = []
    n_start = len(df)

    # 1. sentinel codes -> missing, then coerce to numbers
    for col, rule in NUMERIC_RULES.items():
        if rule["sentinel"]:
            n_sentinel = df[col].isin(rule["sentinel"]).sum()
            if n_sentinel:
                report.append(
                    f"  {col:<13} {n_sentinel:>7,} sentinel values -> missing"
                )
            df[col] = df[col].where(~df[col].isin(rule["sentinel"]))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. impossible values -> missing (a FICO of 50 is not a real credit score)
    for col, rule in NUMERIC_RULES.items():
        outside = df[col].notna() & ~df[col].between(rule["lo"], rule["hi"])
        if outside.any():
            report.append(
                f"  {col:<13} {outside.sum():>7,} outside [{rule['lo']}, "
                f"{rule['hi']}] -> missing"
            )
            df.loc[outside, col] = pd.NA

    # 3. DTI: record missingness as its own column rather than dropping the row.
    #    The missingness is year-dependent (28.8% in 2013, 1.0% in 2018), so
    #    dropping would filter early vintages harder than late ones.
    df["dti_missing"] = df["dti"].isna().astype(int)
    n_dti = int(df["dti_missing"].sum())
    if n_dti:
        report.append(
            f"  {'dti':<13} {n_dti:>7,} missing -> kept, flagged in dti_missing "
            f"({n_dti/len(df):.1%} of rows)"
        )

    required = list(REQUIRED) + (["dti"] if drop_missing_dti else [])

    # 4. drop rows missing a required covariate
    missing_required = df[required].isna().any(axis=1)
    if missing_required.any():
        for col in required:
            n = df[col].isna().sum()
            if n:
                report.append(f"  {col:<13} {n:>7,} missing (required)")
        report.append(
            f"  -> dropped {missing_required.sum():,} rows missing a required covariate"
        )
        df = df[~missing_required]

    # 5. structural checks on the outcome itself
    bad_time = df.time < 0
    if bad_time.any():
        report.append(f"  -> dropped {bad_time.sum():,} rows with negative time")
        df = df[~bad_time]

    bad_event = ~df.event.isin([0, 1, 2])
    if bad_event.any():
        report.append(
            f"  -> dropped {bad_event.sum():,} rows with an invalid event code"
        )
        df = df[~bad_event]

    n_dup = df.loan_id.duplicated().sum()
    if n_dup:
        report.append(f"  -> dropped {n_dup:,} duplicate loan_id rows")
        df = df[~df.loan_id.duplicated()]

    report.append(
        f"  kept {len(df):,} of {n_start:,} loans " f"({len(df)/n_start:.1%})"
    )
    return df, report


def subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """
    Cut to max_rows, taking an equal share from each origination year.

    A simple random sample over the whole pool would let one vintage dominate by
    chance. Sampling within each year keeps every year represented, which is
    what makes a train-on-early / test-on-later split possible later.
    """
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    years = sorted(df.vintage.unique())
    per_year = max_rows // len(years)

    parts = []
    for y in years:
        block = df[df.vintage == y]
        take = min(per_year, len(block))
        parts.append(block.sample(n=take, random_state=seed))

    out = pd.concat(parts, ignore_index=True)

    # integer division leaves a remainder; top up from what is left so the
    # output lands exactly on max_rows rather than slightly under.
    shortfall = max_rows - len(out)
    if shortfall > 0:
        remaining = df[~df.loan_id.isin(set(out.loan_id))]
        if len(remaining):
            out = pd.concat(
                [
                    out,
                    remaining.sample(
                        n=min(shortfall, len(remaining)), random_state=seed
                    ),
                ],
                ignore_index=True,
            )

    return out.sort_values(["vintage", "loan_id"]).reset_index(drop=True)


def describe(df: pd.DataFrame) -> None:
    """Print the summary you should read before handing this file to anyone."""
    print("\n" + "=" * 66)
    print(f"FINAL DATASET: {len(df):,} loans")
    print("=" * 66)

    counts = df.event.value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
    share = counts / len(df)
    print("\noutcome distribution")
    for code, name in [(0, "censored"), (1, "default"), (2, "prepaid")]:
        print(f"  {name:<10} {counts[code]:>8,}   {share[code]:>6.2%}")

    print("\nby origination year")
    tab = pd.crosstab(
        df.vintage, df.event.map({0: "censored", 1: "default", 2: "prepaid"})
    )
    print(tab.to_string())

    print("\nfollow-up time (months)")
    print(
        f"  median {df.time.median():.0f}   mean {df.time.mean():.1f}   "
        f"max {df.time.max():.0f}"
    )

    print("\nnumeric covariates")
    num = [c for c in NUMERIC_RULES if c in df.columns]
    print(
        df[num]
        .describe()
        .T[["count", "mean", "std", "min", "max"]]
        .round(2)
        .to_string()
    )

    # Coded columns, printed the way the medical utils print their
    # VALUE_LABELS: every level a column actually takes, with its count. This is
    # what you scan to catch a sentinel that slipped through or a category that
    # appears in one vintage and not another.
    print("\ncoded columns, observed values")
    coded = [
        c
        for c in (
            "first_time_buyer",
            "occupancy",
            "channel",
            "loan_purpose",
            "property_type",
            "dti_missing",
        )
        if c in df.columns
    ]
    for col in coded:
        vc = df[col].value_counts(dropna=False)
        print(f"  {col}")
        for val, n in zip(vc.index.to_list(), vc.to_list()):
            label = CODE_LABELS.get(col, {}).get(val, "")
            shown = "(missing)" if pd.isna(val) else str(val)
            print(f"    {shown:<4} {label:<26} {n:>7,}   {n/len(df):>6.1%}")

    if "state" in df.columns:
        top = df.state.value_counts()
        print(
            f"  state\n    {top.index[0]} is the most common of "
            f"{df.state.nunique()} states, {top.iloc[0]:,} loans "
            f"({top.iloc[0]/len(df):.1%})"
        )

    n_missing = df.isna().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing):
        print("\nremaining missing values (optional covariates only)")
        for col, n in n_missing.items():
            print(f"  {col:<13} {n:>7,}  ({n/len(df):.1%})")
    else:
        print("\nno missing values remain")


def validate(df: pd.DataFrame, max_follow_up: int, drop_missing_dti: bool) -> None:
    """
    Check the output against the things the rest of the project assumes.

    This exists because a silently wrong CSV is worse than a crash. An earlier
    version of this script read event times off Freddie Mac's Loan Age field,
    which is reset when a loan is modified; the output looked entirely
    reasonable and was wrong for 0.18% of loans. Everything below is a property
    that was checked by hand at the time, written down so the next change gets
    checked for free.
    """
    problems: list[str] = []

    if df.loan_id.duplicated().any():
        problems.append(f"{df.loan_id.duplicated().sum():,} duplicate loan_id")

    bad_event = ~df.event.isin([0, 1, 2])
    if bad_event.any():
        problems.append(f"{bad_event.sum():,} rows with event not in {{0,1,2}}")

    if df.time.isna().any():
        problems.append(f"{df.time.isna().sum():,} rows with missing time")
    if (df.time < 0).any():
        problems.append(f"{(df.time < 0).sum():,} rows with negative time")
    if max_follow_up > 0 and (df.time > max_follow_up).any():
        problems.append(
            f"{(df.time > max_follow_up).sum():,} rows with time beyond the "
            f"{max_follow_up}-month cut"
        )

    required = list(REQUIRED) + (["dti"] if drop_missing_dti else [])
    for col in required:
        if col in df.columns and df[col].isna().any():
            problems.append(f"{df[col].isna().sum():,} missing values in {col}")

    # dti_missing must agree with dti exactly, or the flag is lying
    if "dti_missing" in df.columns:
        disagree = (df.dti.isna() != df.dti_missing.astype(bool)).sum()
        if disagree:
            problems.append(f"{disagree:,} rows where dti_missing disagrees with dti")

    if problems:
        raise SystemExit(
            "OUTPUT FAILED VALIDATION, nothing written:\n  - " + "\n  - ".join(problems)
        )

    # Soft checks: legal, but you probably did not mean them.
    for code, name in ((0, "censored"), (1, "default"), (2, "prepaid")):
        if (df.event == code).sum() == 0:
            print(f"  WARNING: no {name} events in the output")


# ----------------------------------------------------------------------------------------
# Preparing one cohort
# ----------------------------------------------------------------------------------------
def prepare_cohort(
    cohort: str,
    raw_dir: Path,
    years: list,
    max_rows: int,
    max_follow_up: int,
    seed: int,
    drop_missing_dti: bool,
    out_file: Path,
) -> None:
    """Read the raw archives, clean them, subsample and save one CSV for the cohort."""
    print(f"cohort        {cohort}")
    print(f"years         {years}")
    print(f"max rows      {max_rows:,}" if max_rows else "max rows      (all)")
    print(
        f"follow-up     {max_follow_up} months"
        if max_follow_up
        else "follow-up     (full)"
    )
    print(f"seed          {seed}")
    print(f"output        {out_file}\n")

    parts = []
    for year in years:
        print(f"parsing {year} ...", flush=True)
        block = parse_vintage(raw_dir, year, max_follow_up)
        print(f"    {len(block):,} loans")
        parts.append(block)

    df = pd.concat(parts, ignore_index=True)

    print(f"\ncleaning {len(df):,} loans")
    df, report = clean(df, drop_missing_dti=drop_missing_dti)
    for line in report:
        print(line)

    if max_rows and len(df) > max_rows:
        print(
            f"\nsubsampling {len(df):,} -> {max_rows:,} "
            f"(stratified by year, seed {seed})"
        )
        df = subsample(df, max_rows, seed)

    df = df[[c for c in OUTPUT_COLS if c in df.columns]]
    validate(df, max_follow_up, drop_missing_dti)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    describe(df)
    size_mb = out_file.stat().st_size / 1024 / 1024
    print(f"\nwrote {out_file}  ({size_mb:.1f} MB)")


# ----------------------------------------------------------------------------------------
# Normalizing
# ----------------------------------------------------------------------------------------
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
            "Run the folds step again."
        )

    unknown = set(df["state"]) - set(STATE_TO_DIVISION)
    if unknown:
        sys.exit(f"No Census division for {sorted(unknown)}; add them to DIVISIONS.")
    df["division"] = df["state"].map(STATE_TO_DIVISION)
    df = df.drop(columns=COLUMNS_TO_DROP)

    recoded = {}
    for col, levels in CATEGORY_LEVELS.items():
        if df[col].isna().any():
            sys.exit(
                f"{col} is blank on {int(df[col].isna().sum())} row(s); "
                "decide how to code them before encoding."
            )
        codes = MISSING_CODES.get(col, [])
        n = int(df[col].isin(codes).sum())
        if n:
            recoded[col] = n
            df.loc[df[col].isin(codes), col] = levels[0]
        unexpected = set(df[col].dropna()) - set(levels)
        if unexpected:
            sys.exit(f"{col} holds undeclared level(s) {sorted(unexpected)}.")
    if recoded:
        print(
            "  not-available codes folded into the reference level: "
            + ", ".join(
                f"{c} {n} row(s) -> {CATEGORY_LEVELS[c][0]}" for c, n in recoded.items()
            )
        )

    expected = set(PASSTHROUGH) | set(COLUMNS_TO_SCALE) | set(CATEGORY_LEVELS)
    missing, extra = expected - set(df.columns), set(df.columns) - expected
    if missing or extra:
        sys.exit(
            f"Column mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}."
        )
    return df, spec


# ----------------------------------------------------------------------------------------
# Per-fold preprocessing
# ----------------------------------------------------------------------------------------
def normalize_fold(df: pd.DataFrame, train_idx: list, test_idx: list) -> tuple:
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN])

    params: dict = {"impute": {}, "scale": {}, "encode": {}}
    notes = {"imputed": 0, "constant": [], "zero_variance": []}

    # The prepare step leaves dti, cltv and n_borrowers empty rather than
    # filling them, and says so under "NEVER IMPUTES". This is not a disagreement:
    # that script sees one cohort with no train/test boundary, so any median it
    # computed would run over rows that become test rows in every fold. "The training
    # rows" only exists once the folds are built, which makes this the first place
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


def normalize_arm(
    df: pd.DataFrame, folds: list, cohort: str, arm: str, output_dir: Path
) -> list:
    folder = output_dir / cohort / arm
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
            print(
                f"  fold {i}: no variance in the training rows for "
                f"{notes['zero_variance']}, sd set to 1"
            )
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


def normalize_cohort(
    cohort: str, table: Path, folds_file: Path, output_dir: Path
) -> None:
    """Impute, scale and encode each fold with its own training rows and save it."""
    print(f"Scaling  {COLUMNS_TO_SCALE}")
    print(f"Encoding {list(CATEGORY_LEVELS)}")
    print("with the statistics of the training rows of each fold only, then applying")
    print("them to its test rows.\n")

    print("=" * 92)
    print(f"{cohort.upper()} COHORT")
    print("=" * 92)
    df, spec = read_inputs(table, folds_file)

    params = {}
    for arm in ("temporal", "random"):
        params[arm] = normalize_arm(df, spec[arm], cohort, arm, output_dir)

    params_file = output_dir / cohort / "scaling.json"
    params_file.write_text(json.dumps(params, indent=1))
    print(f"\nSaved {params_file}\n")
