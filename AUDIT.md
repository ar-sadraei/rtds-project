# Methodological audit

*12 Oct 2026. A deliberately adversarial re-reading of the pipeline, the statistics and
the claims. Everything below was checked by running code or re-deriving numbers from
`results/`, not by re-reading the PDF.*

## Verdict

**The headline findings hold.** The four defects found are presentational or definitional,
and none of them moves a reported number. One claim is weaker than the report states and
should be caveated.

The reason the findings survive is structural: the central results are *negative*
(DeepHit gains nothing over a random survival forest) and negative results rest on
effect sizes measured against training noise, not on p-values. The defects below all
live in the p-value machinery or in naming.

---

## A. Defects to fix

### A1. The IPCW control weights in `auc_td` are not Blanche's

`src/metrics.py:auc_td` gives **every** control weight `1/G(horizon)`. Blanche et al.
(2013) weight the two kinds of control differently:

| control | correct weight | as coded |
|---|---|---|
| event-free at *t* | 1/G(*t*) | 1/G(*t*) ✓ |
| competing event at *T* ≤ *t* | 1/G(*T*−) | 1/G(*t*) ✗ |

A subject who prepaid at month 12 is known to be a non-defaulter only because they were
uncensored up to month 12, so the weight belongs at *T*, not at the horizon.

`brier_cr` in the same file **does** handle this case correctly (weight 1/G(*T*−), target 0),
which is what exposed the inconsistency.

**Effect on results: none.** The corrected estimator was implemented and run on both
cohorts, both causes, Cox and RSF. All eight cells agree to four decimals:

```
postcrisis Cox default   as-coded 0.7372   Blanche 0.7372   diff 0.0000
postcrisis Cox prepaid   as-coded 0.6729   Blanche 0.6729   diff 0.0000
postcrisis RSF default   as-coded 0.7300   Blanche 0.7300   diff 0.0000
postcrisis RSF prepaid   as-coded 0.6706   Blanche 0.6706   diff 0.0000
crisis     Cox default   as-coded 0.8346   Blanche 0.8346   diff 0.0000
crisis     Cox prepaid   as-coded 0.6192   Blanche 0.6192   diff 0.0000
crisis     RSF default   as-coded 0.8516   Blanche 0.8516   diff 0.0000
crisis     RSF prepaid   as-coded 0.6858   Blanche 0.6858   diff 0.0000
```

Because **G(60) = 0.9986 in both cohorts** — see B2 — every weight is within 0.14% of 1,
so the distinction cannot bite. Fix the code for correctness; no table changes.

Note that `validate_harness.py` cannot catch this: the null model ties every pair (AUC
0.5 under any weighting) and the oracle separates every pair (AUC 1.0 under any
weighting). The suite is blind to weighting errors by construction.

### A2. Bootstrap p-values are reported at their resolution floor

`p = max(p, 1/B)` with `B = 500` floors every p-value at 0.002. **31 of the 36 computed
tests sit exactly on that floor.** They are printed as `p = 0.002` in Table IV; they
should read `p < 0.002`.

This is not weak evidence — p at the floor means all 500 resamples agreed in sign — but
the number as printed overstates precision, and no threshold below 0.002 can be assessed
at B = 500.

### A3. The multiplicity denominator does not match the number of tests

The protocol applies Bonferroni α/3 = 0.0167 for "three metric families". The pipeline
computes **36 tests** (2 cohorts × 2 causes × 3 comparisons × 3 metrics). Correcting for
all 36 gives a threshold of 0.00139, which **no result can reach**, purely because of the
floor in A2.

Two honest repairs, either acceptable:

1. Raise `B` to 10,000 (floor 1e-4) and correct for the full 36. Cost: the bootstrap
   loop is ~20× longer, roughly 2 hours.
2. Keep B = 500 and state plainly that a family is *the three metrics within one
   pre-specified comparison in one cohort*, that 36 tests were computed, and that
   inference rests on effect sizes against training noise rather than on the p-values.

Option 2 is what the report already argues in the Discussion; it just needs saying in
§III-C.

### A4. The abstract contradicts the method section

Abstract: "an **equal**, documented tuning budget."
§III-C: "twenty random configurations for DeepHit and eight for the forest."

Twenty is not eight. The budget is documented, pre-specified and *favours the deep
model*, which is the conservative direction for a conclusion that the deep model does not
win — so the fix is one word, not a re-run. Suggest "a documented tuning budget of twenty
configurations for DeepHit and eight for the forest".

---

## B. Precision, not error

### B1. "Integrated calibration index" is someone else's defined term

Austin & Steyerberg (2019) define the ICI as the mean absolute difference between a
**loess- or hazard-regression-smoothed** calibration curve and the predicted probability,
averaged **over subjects**. What `metrics.ici` computes is an *n*-weighted mean absolute
gap between predicted risk and the Aalen–Johansen estimate **within deciles** — a binned
calibration error.

The report describes the implemented method accurately ("comparing mean predicted risk
against the Aalen–Johansen estimate within deciles of predicted risk") and does **not**
cite Austin, so nothing is misattributed. But the name is taken and an examiner who knows
it will object. Rename to *decile calibration error* throughout, or keep the name, cite
Austin & Steyerberg, and state the deviation in one clause.

### B2. Censoring is almost absent at the horizons actually evaluated

G(60) = 0.9986 in both cohorts: **0.14% of subjects are censored before month 60.** The
20.6% headline is censoring across the whole observation window, essentially all of it
after month 60, because `time` is loan *age* and every vintage has at least eight years of
follow-up.

Every individual statement in the report is true. But the IPCW apparatus is doing nearly
nothing at the reported horizons, and the force that actually shapes the results is the
**competing risk** — prepayment removing three quarters of the loans from the risk set —
not censoring. One sentence in §III-B would make the paper harder to attack and costs
nothing, since the competing-risk framing is the paper's real contribution anyway.

### B3. "Falls inside both ranges" is true at the boundary

Our 20.6% against Dirick's 20–85%. Inside, but at the very edge. The 95.5% against
94.56–98.16% is comfortably inside. Worth phrasing as "at the lower edge of Dirick's
range and inside the upper one".

---

## C. The one claim that is weaker than stated

**The calibration penalty in the crisis cohort.**

§IV-A: "In both cohorts DeepHit's integrated calibration index is 0.0040 worse than
Cox's." The point estimates are right. Against training-resample variability
(`results/training_variability.csv`):

| cohort | Cox ICI sd | DeepHit ICI sd | combined | +0.0040 as multiples |
|---|---|---|---|---|
| post-crisis | 0.0001 | 0.0002 | 0.0002 | **17.7 σ** |
| crisis | 0.0020 | 0.0012 | 0.0023 | **1.7 σ** |

The post-crisis penalty is overwhelming. The crisis penalty is about 1.7 combined standard
deviations — the same order as the crisis AUC win that finding 8 already forced a caveat
onto. The **direction** is consistent across both cohorts, which is what "does not
reverse" claims and that claim survives. The **crisis magnitude** should carry the same
hedge as the crisis AUC does.

This is the third instance in this project of a difference being read without checking
both models' noise. The first two were caught and fixed; this one was not.

---

## D. Checked and sound

| check | result |
|---|---|
| Cox coefficient recovery, 40,000 simulated subjects | max error 0.011 (cause 1), 0.021 (cause 2) against known β |
| CIF construction, Cox and RSF | CIF_k = Σ S(u−) dΛ_k(u) with S from **all** causes — the textbook form, not 1 − exp(−H_k) |
| CIF₁ + CIF₂ ≤ 1 | max 1.0003 on simulated data (3e-4 grid artefact) |
| Fine–Gray vs cause-specific Cox, same simulated data | 0.3453 vs 0.3457 at t = 200 — independent agreement |
| Train/test leakage | none: tuning selects on an inner validation vintage inside the training years; test vintages enter only at final evaluation |
| Brier competing-event handling | correct (weight 1/G(T−), target 0) |
| `validate_harness.py` | real: null AUC exactly 0.5, null calibrated on a random split and not on a vintage split, oracle AUC ≈ 1, CIFs sum ≤ 1 |
| α ablation "exceeds seed sd" | three cells correspond to t ≈ 3.6, 9.9, 10.3 — not overstated |
| Negative headline results | p = 0.968 and 0.788, far from the bootstrap floor, untouched by A2/A3 |
| Seed vs bootstrap variability | correctly distinguished (0.0014/0.0010 seed; 0.0035 training resample) |
| 550,000 loans | 150k+150k post-crisis, 100k+150k crisis — all of it used |

## E. Honest framing already in place

Fine–Gray CIFs not constrained to sum below 1, declared. RSF is cause-specific rather than
Ishwaran's competing-risks formulation, declared. Per-cause D-calibration deliberately not
implemented rather than approximated. The oracle recalibration labelled unattainable
rather than reported as a result. Novelty assessed as confirmatory rather than claimed.

## F. What the audit cannot fix

One dataset family, no clinical replication, origination-time covariates only, five
bootstrap replicates. All already declared in the limitations.

---

## Priority before submission

1. **C** — hedge the crisis calibration penalty (changes what the paper claims).
2. **A4** — one word in the abstract.
3. **A2** — `p < 0.002` in Table IV.
4. **B1** — rename the metric or cite and qualify.
5. **A3** — one sentence in §III-C.
6. **B2** — one sentence in §III-B.
7. **A1** — fix the code; no numbers change.
