# Financial dataset — competing risks

Dataset preparation for the group project in *Research Topics in Data Science*
(DSV, Stockholm University, 2026). Ali Emre Nebiler, Amirreza Sadraei,
Filip Becic.

This repository holds **only the financial dataset pipeline**. It turns the raw
Freddie Mac mortgage files into one clean CSV with a competing-risks outcome,
for use alongside the clinical dataset (NHANES).

## What a competing-risks dataset is

Every loan ends in one of three states, and they exclude each other:

| `event` | meaning |
|---|---|
| `0` | **censored** — still performing when we stopped looking |
| `1` | **default** — borrower stopped paying |
| `2` | **prepaid** — paid off early (refinanced or house sold) |

`time` is the loan's age **in months** when that happened. Once a loan defaults
it can never prepay, and the reverse — that mutual exclusion is the competing
risk. Prepayment is roughly 18× more common than default here, so treating
prepaid loans as merely "censored" would discard most of the outcomes.

## Build it

```bash
python3 src/prepare_financial_data.py
```

Needs `data/raw/sample_YYYY.zip` for each year. `DATA.md` says where to get
them. Output: `data/processed/financial_competing_risks.csv`.

Defaults: 2013–2018, 50,000 loans, follow-up cut at 60 months, seed 0.

```bash
python3 src/prepare_financial_data.py --max-rows 250000     # larger
python3 src/prepare_financial_data.py --years 2016 2017 2018 --max-follow-up 36
```

The script prints a cleaning report before it writes. That report is the audit
trail — read it rather than trusting the file.

## The data is not in this repository

The Freddie Mac dataset is free for academic use but **may not be
redistributed**, so `data/` is gitignored. The processed CSV is a derived work
of licensed data, so it must not be committed or posted publicly either. Pass it
to the group directly.

## Two things to know before modelling

**DTI has missing values on purpose.** Freddie Mac did not report it
consistently before 2016 — missing for 28.8% of 2013 loans against 1.0% of 2018
loans. Those rows are kept with `dti` empty and `dti_missing` set, rather than
dropped, because dropping them would filter the early years far harder than the
late ones and confound any train-early / test-later comparison. If your model
cannot take an empty value, impute from the **training fold only** — imputing
from the whole file leaks test information into training.

**Follow-up is cut at 60 months.** A 2013 loan has been observed far longer than
a 2018 one. Without a common cut, `time` means something different for each
year. `--max-follow-up 0` keeps full follow-up.
