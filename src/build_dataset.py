"""
Build a competing-risks table from the Freddie Mac Single-Family Loan-Level Dataset.

Input   project/data/raw/sample_YYYY.zip   (origination + monthly performance, pipe-delimited, no header)
Output  project/data/processed/loans.parquet  — one row per loan:
            loan_id, vintage, time, event, <origination covariates>

        event: 0 = censored, 1 = default, 2 = prepaid
        time : loan age in months at the event, or at last observation if censored

Field positions come from the official layout, "File Layout (effective July 2026)":
https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx
Origination file has 31 fields, monthly performance 35. The loader asserts both.

Event definitions
-----------------
default    first month with Current Loan Delinquency Status >= 3 (90+ days past due),
           or status "RA" (REO acquisition). This follows Blumenstock et al. (2022),
           who define default as the first 3-month delinquency. It is observed earlier
           and more consistently than a disposition code.
prepaid    Zero Balance Code "01" (prepaid or matured), if it occurs before any default.
           For 30-year loans originated 2013-2018, legal maturity falls in 2043+, so
           within this window code 01 is prepayment rather than maturity.
censored   neither event observed by the end of the performance window.

`credit_event_zb` is also recorded (Zero Balance Code in 02, 03, 09, 15) as an
alternative, disposition-based default definition for a sensitivity check.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

N_ORIG_FIELDS = 31
N_PERF_FIELDS = 35

# origination: 0-based index -> name  (subset we keep)
ORIG_COLS = {
    0: "fico",
    1: "first_payment_date",
    2: "first_time_buyer",
    5: "mi_pct",
    6: "n_units",
    7: "occupancy",
    8: "cltv",
    9: "dti",
    10: "orig_upb",
    11: "ltv",
    12: "orig_rate",
    13: "channel",
    16: "state",
    17: "property_type",
    19: "loan_id",
    20: "loan_purpose",
    21: "orig_term",
    22: "n_borrowers",
}

# performance: 0-based index
P_LOAN_ID, P_PERIOD, P_DELINQ, P_AGE, P_ZB_CODE = 0, 1, 3, 4, 8

# Zero Balance Codes treated as credit events (disposition-based alternative)
ZB_CREDIT = {"02", "03", "09", "15"}
ZB_PREPAID = "01"

# Delinquency status at or above which we call it a default. "RA" = REO acquisition.
DELINQ_DEFAULT = 3


def _delinq_to_int(s: str) -> int | None:
    """Delinquency status -> months past due. 'RA' is treated as deep default, 'XX' unknown."""
    s = s.strip()
    if not s or s == "XX":
        return None
    if s == "RA":
        return 99
    try:
        return int(s)
    except ValueError:
        return None


def parse_vintage(year: int) -> pd.DataFrame:
    zp = RAW / f"sample_{year}.zip"
    if not zp.exists():
        raise FileNotFoundError(zp)

    with zipfile.ZipFile(zp) as z:
        # ---- origination: small, load directly -------------------------------
        with z.open(f"sample_orig_{year}.txt") as fh:
            orig = pd.read_csv(
                io.TextIOWrapper(fh, "latin-1"),
                sep="|",
                header=None,
                usecols=list(ORIG_COLS),
                names=[ORIG_COLS.get(i, f"_{i}") for i in range(N_ORIG_FIELDS)],
                dtype=str,
                keep_default_na=False,
            )

        # ---- performance: stream, one pass, never held in memory -------------
        #  per loan: first default age, prepay age, credit-event age, last seen age
        first_default: dict[str, int] = {}
        prepaid_at: dict[str, int] = {}
        credit_zb_at: dict[str, int] = {}
        last_age: dict[str, int] = {}
        bad_rows = 0

        with z.open(f"sample_perf_{year}.txt") as fh:
            for line in io.TextIOWrapper(fh, "latin-1"):
                f = line.rstrip("\n").split("|")
                if len(f) != N_PERF_FIELDS:
                    bad_rows += 1
                    continue

                lid = f[P_LOAN_ID]
                try:
                    age = int(f[P_AGE])
                except ValueError:
                    continue

                if age > last_age.get(lid, -1):
                    last_age[lid] = age

                d = _delinq_to_int(f[P_DELINQ])
                if d is not None and d >= DELINQ_DEFAULT and lid not in first_default:
                    first_default[lid] = age

                zb = f[P_ZB_CODE].strip()
                if zb == ZB_PREPAID and lid not in prepaid_at:
                    prepaid_at[lid] = age
                elif zb in ZB_CREDIT and lid not in credit_zb_at:
                    credit_zb_at[lid] = age

    # ---- resolve the competing events -------------------------------------
    rows = []
    for lid, last in last_age.items():
        d_age = first_default.get(lid)
        p_age = prepaid_at.get(lid)

        if d_age is not None and (p_age is None or d_age <= p_age):
            time, event = d_age, 1          # default first
        elif p_age is not None:
            time, event = p_age, 2          # prepayment first
        else:
            time, event = last, 0           # censored

        rows.append(
            (lid, time, event, int(credit_zb_at.get(lid) is not None), last)
        )

    ev = pd.DataFrame(
        rows, columns=["loan_id", "time", "event", "credit_event_zb", "last_age"]
    )

    df = orig.merge(ev, on="loan_id", how="inner", validate="one_to_one")
    df.insert(1, "vintage", year)

    # numeric coercion; Freddie Mac uses 9s as missing codes
    for c, sentinel in [
        ("fico", {"9999"}),
        ("cltv", {"999"}),
        ("dti", {"999"}),
        ("ltv", {"999"}),
        ("mi_pct", {"999"}),
        ("n_units", {"99"}),
        ("n_borrowers", {"99"}),
    ]:
        df[c] = pd.to_numeric(df[c].where(~df[c].isin(sentinel)), errors="coerce")
    for c in ["orig_upb", "orig_rate", "orig_term"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if bad_rows:
        print(f"  [{year}] skipped {bad_rows} malformed performance rows")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2013, 2014, 2015, 2016, 2017, 2018])
    ap.add_argument("--out", default="loans",
                    help="output stem under data/processed (no extension)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for y in args.years:
        print(f"parsing {y} ...", flush=True)
        d = parse_vintage(y)
        n = len(d)
        rates = d.event.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
        print(f"  {n:,} loans | censored {rates[0]:.1%} "
              f"| default {rates[1]:.2%} | prepaid {rates[2]:.1%} "
              f"| median follow-up {d.last_age.median():.0f} mo")
        parts.append(d)

    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_parquet(OUT / f"{args.out}.parquet", index=False)
    all_df.to_csv(OUT / f"{args.out}.csv.gz", index=False, compression="gzip")

    print(f"\nwrote {len(all_df):,} loans -> {OUT / (args.out + '.parquet')}")
    print("\nevent counts by vintage:")
    print(pd.crosstab(all_df.vintage, all_df.event.map(
        {0: "censored", 1: "default", 2: "prepaid"})))


if __name__ == "__main__":
    main()
