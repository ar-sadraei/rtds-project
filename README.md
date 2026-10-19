# Ranking, Calibration and Cost in Competing-Risks Survival Models

<https://github.com/ar-sadraei/rtds-project>

A common-protocol benchmark of cause-specific Cox, Fine–Gray, a random survival
forest and DeepHit on 550,000 Freddie Mac mortgages, measuring discrimination,
calibration and computational cost together under vintage-ordered splits.

The finding: the advantage deep competing-risks models are reported to have is
substantially an artefact of the protocol used to evaluate them. Under
rolling-origin evaluation it does not survive, and a matched test — same test
loans, same training-set size, only the training *period* moved — shows that
what produces it is training on loans originated alongside the test loans
rather than the model class. On calibration, three independent measures agree
that DeepHit is the worst of the five models.

Code for the research project component of *Research Topics in Data Science*
(DSV, Stockholm University, 2026). Amirreza Sadraei, Ali Emre Nebiler, Filip Becic.

## What is here

```
src/           pipeline, models, metrics, experiments
report/        LaTeX source; every table and figure is generated from results/
results/       tidy CSVs, one row per model x split x metric; see results/README.md
DATA.md        dataset card: sources, event definitions, known issues
FINDINGS.md    plain-language summary of every result
ORIENTATION.md the whole project end to end: decisions and why they were taken
GROUP_BRIEF.md where the project stands, for the other two authors
AUDIT.md       methodological self-audit and what it changed
PROJECT_PLAN.md the original plan, kept as the record of what was pre-registered
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

# CR D-calibration is not in any released hazardous; it lives in the authors'
# open pull request. Pin the commit the reported numbers were produced with:
.venv/bin/pip install --no-cache-dir \
  "git+https://github.com/soda-inria/hazardous.git@26e64323"

.venv/bin/python src/build_dataset.py --years 2013 2014 2015 2016 2017 2018 --out loans
.venv/bin/python src/build_dataset.py --years 2005 2006 2007 2008 2009 --out loans_crisis

.venv/bin/python src/validate_harness.py          # sanity checks; run this first
.venv/bin/python src/run_all.py                   # full model grid
.venv/bin/python src/tune.py                      # hyperparameter search, both cohorts
.venv/bin/python src/significance_tuned.py        # paired bootstrap on tuned models
.venv/bin/python src/run_extras.py                # training variability + recalibration

# the protocol experiments, which carry the paper's claim
.venv/bin/python src/run_random_split.py          # random vs vintage split
.venv/bin/python src/run_protocol_decomposition.py # rolling origin + matched test
.venv/bin/python src/run_matched_bootstrap.py     # paired bootstrap + DiD on the matched test

# proper competing-risks calibration
.venv/bin/python src/run_aj_calibration.py        # AJ-calibration (marginal)
.venv/bin/python src/run_cr_dcalibration.py       # CR D-calibration (distributional)

.venv/bin/python src/dump_calibration_curves.py   # decile curves -> results/
.venv/bin/python src/make_tables.py               # regenerate report tables
.venv/bin/python src/make_figure.py               # regenerate the calibration figure
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`src/validate_harness.py` checks that the Aalen–Johansen null model scores exactly
0.5 AUC and is calibrated in-distribution. If those fail, the metrics are wrong
and nothing downstream should be trusted. The two calibration scripts apply the
same idea: the Aalen–Johansen estimator is marginally consistent by
construction, so it must score near zero on a *random* split. It does —
1.1×10⁻⁵ under AJ-calibration, and 2.5×10⁻⁵ (crisis) and 3.9×10⁻⁵
(post-crisis) under CR D-calibration — which is what licenses reading its much
larger vintage-split scores (0.0096/0.0129 and 0.0084/0.0980) as the
distribution shift rather than as a bug.

`src/make_tables.py` and `src/make_figure.py` read only `results/*.csv`, so the
report rebuilds on a clean checkout without the raw loan data.

## Notes

- Cox and Fine–Gray are implemented directly; no mainstream Python library ships
  Fine–Gray. Both maximise a Breslow partial likelihood.
- The forest is a *cause-specific* random survival forest, not Ishwaran's
  composite-splitting competing-risks formulation.
- Discrimination is IPCW time-dependent AUC, not the C-index: Blanche et al.
  show the C-index is improper for t-year risk and can prefer a worse model.
- Three calibration measures are reported — a decile-binned index (conditional,
  one horizon), AJ-calibration (marginal over events and time) and CR
  D-calibration (distributional). The first is not proper for competing risks;
  it is kept because it is what the credit literature uses, and the two proper
  measures from Alberge et al. (2026) are reported beside it.
- `src/audit_auc_weights.py` is the record of an IPCW weighting bug in `auc_td`
  that was found and fixed (its numerical effect here was nil, since G(60)=0.9986,
  but the weights were wrong). `src/audit_estimators.py` checks that the
  hand-rolled Cox and Fine–Gray estimators recover known coefficients on
  simulated data. Both are kept so the checks stay runnable.
