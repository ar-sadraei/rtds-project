# What each results file is, and which claim it supports

Every number in the report comes from one of these. The four tables in the
paper are generated from them by `src/make_tables.py`; the figure by
`src/make_figure.py`. Numbers quoted in the prose but not tabulated are listed
below against the file they come from.

| File | Produced by | Used for |
|---|---|---|
| `all.csv` | `run_all.py` | the full model grid: every model × split × metric. Source of Table I (levels). |
| `classical.csv` | `run_classical.py` | Cox and Fine–Gray alone, used while the harness was being validated. |
| `crisis.csv` | `run_crisis.py` | the crisis cohort grid; the crisis half of Table I. |
| `tuned.csv` | `tune.py` | the selected configuration per model per cohort. **Every later script reads this**, so the tuned settings are fixed in one place. |
| `tuning_trials_{crisis,postcrisis}.csv` | `tune.py` | all 40 DeepHit and 8 RSF trials. Source of the pooled α marginal (0.658 → 0.776) that §V-D shows is misleading. |
| `alpha_ablation.csv` | `run_alpha_ablation.py` | the controlled α ablation, 5 seeds, everything else fixed. Source of the +0.0386 / −0.0673 range and the 0.003–0.010 seed SDs in §V-D. |
| `deephit_seed_spread.csv` | `significance_tuned.py` | DeepHit's seed-to-seed SD, the yardstick every "is this difference real?" judgement in the paper is measured against. |
| `training_variability.csv` | `run_extras.py` | refits on 5 bootstrap resamples of the training vintages. Needed because Cox and Fine–Gray are deterministic, so seeds are the wrong device for them. |
| `significance.csv`, `significance_tuned.csv` | `significance*.py` | paired bootstrap, B=500, on the single vintage split. |
| `protocol_comparison.csv` | `run_random_split.py` | random vs vintage split, models and settings identical. Under random splits DeepHit leads in all 8 cells; under vintage ordering 3 of those leads reverse. |
| `protocol_decomposition.csv` | `run_protocol_decomposition.py` | **the paper's central experiment.** Rolling origin (4 expanding folds per cohort) and the matched test (same test loans, same training size, only the period moved). Source of Tables II and III. |
| `matched_bootstrap.csv` | `run_matched_bootstrap.py` | paired bootstrap and difference-in-differences over the shared test loans. The DiD is significant in 8 of 8 cells at Bonferroni 0.05/3, which is what makes the matched-test reading a test rather than a point estimate. |
| `aj_calibration.csv` | `run_aj_calibration.py` | AJ-calibration and KM-calibration (Alberge et al. 2026), plus the fraction of incoherent CIF vectors per model. Half of Table IV. |
| `cr_dcalibration.csv` | `run_cr_dcalibration.py` | CR D-calibration and its KS test, from the unreleased PR branch (see README). Other half of Table IV. Includes the random-split control rows. |
| `calibration_curves.csv` | `dump_calibration_curves.py` | the decile curves behind Fig. 1. Split out from plotting so the figure can be resized without refitting three models. |
| `recalibration.csv` | `run_extras.py` | per-horizon rescaling fitted on train vs on test. Source of "ten of sixteen cells worse" and "median factor of 4.2" in §V-C. |
| `featureset.csv` | `run_features.py` | feature-set ablation; its `rich` rows are also the post-crisis half of Table I. |
| `tmax_ablation.csv` | `run_tmax.py` | sensitivity to the follow-up window. |

`*.log` files are the stdout of the corresponding run, kept locally as the
record of what was executed. They are gitignored.

## Reading the calibration files

The two proper measures both include a control that is correct by
construction. The Aalen–Johansen estimator assigns every subject the same
marginal incidence, so it cannot discriminate, but it *is* marginally
calibrated. On a random split it therefore has to score near zero, and it does:

| | random split | vintage split |
|---|---|---|
| AJ-calibration | 0.000011 | 0.0096 (crisis), 0.0129 (post) |
| CR D-calibration | 0.000025 / 0.000039 | 0.0084 (crisis), 0.0980 (post) |

That three-orders-of-magnitude gap is the distribution shift measured directly
rather than inferred. It is also why a large score for a *model* on the vintage
split cannot be read as that model being badly fitted without first subtracting
what the shift alone costs.

The KS test that accompanies CR D-calibration rejects every model in every
cell, including the Aalen–Johansen estimator on the random split where it is
correct by construction. With 50,000 test subjects the test has power to reject
on negligible deviations. The score is informative; the test, at this sample
size, is not.
