# Ranking, Calibration and Cost in Competing-Risks Survival Models

<https://github.com/ar-sadraei/rtds-project>

A common-protocol benchmark of cause-specific Cox, Fine–Gray, a random survival
forest and DeepHit on 550,000 Freddie Mac mortgages, measuring discrimination,
calibration and computational cost together under vintage-ordered splits.

Code for the research project component of *Research Topics in Data Science*
(DSV, Stockholm University, 2026). Amirreza Sadraei, Ali Emre Nebiler, Filip Becic.

## What is here

```
src/     pipeline, models, metrics, experiments
report/  LaTeX source; every table is generated from results/
results/ tidy CSVs, one row per model x split x metric
DATA.md  dataset card: sources, event definitions, known issues
FINDINGS.md  plain-language summary of every result
```

## Data

**Not included.** The Freddie Mac Single-Family Loan-Level Dataset is free for
academic use but may not be redistributed. Register at
<https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>, download the
50,000-loan sample files for the vintages you want, and put the archives in
`data/raw/`. `DATA.md` lists the vintages used and the official field layout.

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install pandas pyarrow scipy scikit-survival torch pycox matplotlib

.venv/bin/python src/build_dataset.py --years 2013 2014 2015 2016 2017 2018 --out loans
.venv/bin/python src/build_dataset.py --years 2005 2006 2007 2008 2009 --out loans_crisis

.venv/bin/python src/validate_harness.py   # sanity checks; run this first
.venv/bin/python src/run_all.py            # full model grid
.venv/bin/python src/tune.py               # hyperparameter search, both cohorts
.venv/bin/python src/significance_tuned.py # paired bootstrap on tuned models
.venv/bin/python src/run_extras.py         # training variability + recalibration

.venv/bin/python src/make_tables.py        # regenerate report tables from results/
.venv/bin/python src/make_figure.py        # regenerate the calibration figure
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`src/validate_harness.py` checks that the Aalen–Johansen null model scores exactly
0.5 AUC and is calibrated in-distribution. If those fail, the metrics are wrong
and nothing downstream should be trusted.

## Notes

- Cox and Fine–Gray are implemented directly; no mainstream Python library ships
  Fine–Gray. Both maximise a Breslow partial likelihood.
- The forest is a *cause-specific* random survival forest, not Ishwaran's
  composite-splitting competing-risks formulation.
- No per-cause calibration measure is implemented; see Alberge et al. (2026).
