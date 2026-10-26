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

from . import fold_utils, report_utils

# ----------------------------------------------------------------------------------------
# 2. File layout and columns
# ----------------------------------------------------------------------------------------
# Asserted at parse time: if Freddie Mac changes the layout, fail loudly.
N_ORIG_FIELDS = 31
N_PERF_FIELDS = 35

# Origination file: 0-based field index -> column name. These are the fields we
# keep; everything else in the 31 is dropped.
ORIG_FIELDS = {
    # credit score at origination
    0: "fico",
    2: "first_time_buyer",
    # mortgage insurance coverage %
    5: "mi_pct",
    6: "n_units",
    # P owner-occupied, S second home, I investment
    7: "occupancy",
    # combined loan-to-value
    8: "cltv",
    # debt-to-income
    9: "dti",
    # original unpaid principal balance
    10: "orig_upb",
    # loan-to-value
    11: "ltv",
    # interest rate
    12: "orig_rate",
    # R retail, B broker, C correspondent
    13: "channel",
    16: "state",
    17: "property_type",
    19: "loan_id",
    # P purchase, C cash-out refi, N no-cash-out refi
    20: "loan_purpose",
    # months
    21: "orig_term",
    22: "n_borrowers",
}

# Performance file: 0-based field indices we read while streaming.
P_LOAN_ID, P_PERIOD, P_DELINQ, P_AGE, P_ZB_CODE = 0, 1, 3, 4, 8

# months past due that counts as default
DELINQ_DEFAULT = 3
# zero balance code: prepaid or matured
ZB_PREPAID = "01"
# disposition codes meaning a credit loss
ZB_CREDIT = {"02", "03", "09", "15"}

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
                # YYYYMM
                period = int(fields[P_PERIOD])
            except ValueError:
                malformed += 1
                continue

            # absolute month
            month = (period // 100) * 12 + (period % 100)

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


def resolve_events(loans: dict, follow_up_months: int) -> pd.DataFrame:
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
        if follow_up_months > 0 and time > follow_up_months:
            time, event = follow_up_months, 0

        rows.append((loan_id, time, event, int(r["zb_credit"])))

    return pd.DataFrame(rows, columns=["loan_id", "time", "event", "default_zb_alt"])


def parse_vintage(raw_dir: Path, year: int, follow_up_months: int) -> pd.DataFrame:
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

    events = resolve_events(loans, follow_up_months)

    # inner join: a loan must appear in BOTH files to be usable.
    # validate="one_to_one" makes a duplicated loan_id an error rather than a
    # silent row explosion.
    df = orig.merge(events, on="loan_id", how="inner", validate="one_to_one")
    df.insert(1, "vintage", year)

    if malformed:
        report_utils.log(f"Skipped {malformed:,} malformed performance rows")
    only_orig = len(orig) - len(df)
    if only_orig:
        report_utils.log(f"Dropped {only_orig:,} loans with no performance records")

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
                code = ", ".join(sorted(rule["sentinel"]))
                report.append(
                    f"  {col}: {n_sentinel:,} loans carry the code {code} "
                    "(not available), set to missing"
                )
            df[col] = df[col].where(~df[col].isin(rule["sentinel"]))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. impossible values -> missing (a FICO of 50 is not a real credit score)
    for col, rule in NUMERIC_RULES.items():
        outside = df[col].notna() & ~df[col].between(rule["lo"], rule["hi"])
        if outside.any():
            report.append(
                f"  {col}: {outside.sum():,} loans outside the valid range "
                f"{rule['lo']} to {rule['hi']}, set to missing"
            )
            df.loc[outside, col] = pd.NA

    # 3. DTI: record missingness as its own column rather than dropping the row.
    #    The missingness is year-dependent (28.8% in 2013, 1.0% in 2018), so
    #    dropping would filter early vintages harder than late ones.
    df["dti_missing"] = df["dti"].isna().astype(int)
    n_dti = int(df["dti_missing"].sum())
    if n_dti:
        report.append(
            f"  dti: {n_dti:,} loans have no value ({n_dti/len(df):.1%}). "
            "Kept, with a dti_missing flag"
        )

    required = list(REQUIRED) + (["dti"] if drop_missing_dti else [])

    # 4. drop rows missing a required covariate
    missing_required = df[required].isna().any(axis=1)
    if missing_required.any():
        for col in required:
            n = df[col].isna().sum()
            if n:
                report.append(f"  {col}: {n:,} loans have no value")
        report.append(
            f"  Dropped {missing_required.sum():,} loans missing a required value "
            f"({', '.join(required)})"
        )
        df = df[~missing_required]

    # 5. structural checks on the outcome itself
    bad_time = df.time < 0
    if bad_time.any():
        report.append(f"  Dropped {bad_time.sum():,} loans with a negative time")
        df = df[~bad_time]

    bad_event = ~df.event.isin([0, 1, 2])
    if bad_event.any():
        report.append(f"  Dropped {bad_event.sum():,} loans with an invalid event code")
        df = df[~bad_event]

    n_dup = df.loan_id.duplicated().sum()
    if n_dup:
        report.append(f"  Dropped {n_dup:,} duplicate loan ids")
        df = df[~df.loan_id.duplicated()]

    report.append(f"  Kept {len(df):,} of {n_start:,} loans ({len(df)/n_start:.1%})")
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


def validate(df: pd.DataFrame, follow_up_months: int, drop_missing_dti: bool) -> None:
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
    if follow_up_months > 0 and (df.time > follow_up_months).any():
        problems.append(
            f"{(df.time > follow_up_months).sum():,} rows with time beyond the "
            f"{follow_up_months}-month cut"
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
            report_utils.log(f"WARNING: no {name} events in the output")


# ----------------------------------------------------------------------------------------
# Preparing one cohort
# ----------------------------------------------------------------------------------------
def prepare_cohort(
    raw_dir: Path,
    years: list,
    max_rows: int,
    follow_up_months: int,
    seed: int,
    drop_missing_dti: bool,
    out_file: Path,
) -> None:
    """Read the raw archives, clean them, subsample and save one CSV for the cohort."""
    parts = []
    for year in years:
        parts.append(parse_vintage(raw_dir, year, follow_up_months))

    df = pd.concat(parts, ignore_index=True)
    report_utils.log(f"Parsed {len(parts)} vintages: {len(df):,} loans")

    report_utils.log(f"Cleaning {len(df):,} loans")
    df, report = clean(df, drop_missing_dti=drop_missing_dti)
    for line in report:
        report_utils.log(line.strip())

    if max_rows and len(df) > max_rows:
        report_utils.log(
            f"Subsampling {len(df):,} -> {max_rows:,} "
            f"(stratified by year, seed {seed})"
        )
        df = subsample(df, max_rows, seed)

    df = df[[c for c in OUTPUT_COLS if c in df.columns]]
    validate(df, follow_up_months, drop_missing_dti)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    counts = df.event.value_counts()
    report_utils.log(
        f"Final dataset: {len(df):,} loans, "
        f"{counts.get(0, 0):,} censored, {counts.get(1, 0):,} default, "
        f"{counts.get(2, 0):,} prepaid"
    )
    report_utils.log(f"Wrote {out_file}")


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
        report_utils.log(
            "Not-available codes folded into the reference level: "
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

    all_params, dropped = [], set()
    for i, fold in enumerate(folds, start=1):
        train, test, params, notes = normalize_fold(df, fold["train"], fold["test"])
        train.to_csv(folder / f"fold_{i}_train.csv", index=False)
        test.to_csv(folder / f"fold_{i}_test.csv", index=False)
        all_params.append({"fold": i, **params})
        dropped |= set(notes["constant"])
        if notes["zero_variance"]:
            report_utils.log(
                f"Fold {i}: no variance in the training rows for "
                f"{notes['zero_variance']}, sd set to 1"
            )

    per_fold = [
        tuple(sorted(col for c in CATEGORY_LEVELS for col in p["encode"][c]["columns"]))
        for p in all_params
    ]
    if len(set(per_fold)) > 1:
        sys.exit(f"  {cohort}/{arm}: the folds do not share one design matrix.")
    return all_params, dropped


# ----------------------------------------------------------------------------------------


def normalize_cohort(
    cohort: str, table: Path, folds_file: Path, output_dir: Path
) -> None:
    """Impute, scale and encode each fold with its own training rows and save it."""
    report_utils.log(
        f"Scaling {COLUMNS_TO_SCALE} and encoding {list(CATEGORY_LEVELS)} with the "
        "statistics of the training rows of each fold only, then applying them to "
        "its test rows."
    )
    df, spec = read_inputs(table, folds_file)

    params, dropped = {}, set()
    for arm in fold_utils.ARMS:
        params[arm], absent = normalize_arm(df, spec[arm], cohort, arm, output_dir)
        dropped |= absent
    if dropped:
        report_utils.log(
            f"No column for {', '.join(sorted(dropped))}: no loan in the training "
            "rows has this category, so there is nothing to encode"
        )

    params_file = output_dir / cohort / "scaling.json"
    params_file.write_text(json.dumps(params, indent=1))
    report_utils.log(
        f"Saved {output_dir / cohort} (scaling.json holds the statistics used)"
    )
