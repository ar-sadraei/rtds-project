# What we have found

*As of 11 Oct 2026. For Amirreza, Ali Emre and Filip. Every number here is traceable to a file in `results/`.*

## The one-sentence answer

Under a common protocol with time-ordered splits, **DeepHit's advantage over classical competing-risks models depends entirely on the regime — it wins on crisis-era mortgages and loses on recent ones — while being consistently worse calibrated in both, and never beating a random survival forest.**

---

## Solid findings

### 1. The literature review's headline number replicates
Applying Dirick's two event definitions to 300,000 Freddie Mac loans:

| | censoring rate |
|---|---|
| prepayment counted as an event | **20.6%** |
| prepayment censored | **95.5%** |

Dirick et al. report 20–85% and 94.56–98.16% on ten European consumer-credit portfolios. Our figures fall inside both ranges on US agency mortgages, a different decade and a different product. An independent replication of the number the review was built on.

### 2. Prepayment dominates the data
75.0% prepaid, 4.5% default, 20.6% censored — prepayment outnumbers default **17 to 1**. Censoring it throws away three quarters of the observed outcomes.

### 3. The definition of default changes the event rate 14×
90+ days past due gives 4.47%; the disposition-based zero-balance credit event gives 0.31%. A definitional choice, not a modelling one.

### 4. The advantage reverses with regime
Tuned DeepHit vs cause-specific Cox, AUC at 60 months, default:

| cohort | DeepHit | Cox | diff | significant |
|---|---|---|---|---|
| crisis (test 2007–09) | **0.8538** | 0.8346 | **+0.0192** | yes |
| post-crisis (test 2016–18) | 0.7280 | **0.7372** | **−0.0091** | yes |

Both differences are outside DeepHit's seed-to-seed SD (0.0014 / 0.0010).

### 5. The calibration penalty does *not* reverse
ICI is **+0.0040 worse than Cox in both cohorts**. Whatever the regime, the deep model's probabilities are further from observed frequencies.

### 6. Nothing is gained over a random survival forest
AUC is statistically indistinguishable: p = 0.968 (crisis), p = 0.788 (post-crisis), both inside seed noise. On **crisis prepayment** RSF is clearly better: AUC 0.685 vs 0.620, ICI 0.154 vs 0.236.

### 7. DeepHit's ranking term is harmful — the strongest single result
Across 40 configurations and both cohorts, increasing the weight on the ranking loss makes **both** discrimination and calibration worse, monotonically:

| α | val AUC | val IBS |
|---|---|---|
| 0.10 | 0.658 | 0.0334 |
| 0.20 *(pycox default)* | 0.749 | 0.0286 |
| 0.50 | 0.776 | 0.0269 |
| 1.00 *(pure likelihood)* | **0.776** | 0.0271 |

pycox's default sits inside the harmful range. DeepHit's own paper tunes this weight on validation *discrimination*.

### 8. Cost spans three orders of magnitude
Fine–Gray 0.03 s, Cox 0.42 s, RSF 36 s, DeepHit 59 s per fit on 150,000 loans — roughly **2,000×** for no reliable gain.

---

## Hypotheses we tested and did **not** confirm

These are results, not failures. Lecture 8 treats failing to reject as an outcome.

**Selecting on discrimination costs calibration.** Pre-specified, tested both cohorts and both model families. Three of four cells fall within seed noise; only the crisis AUC effect (+0.0101) is robust, and that is simply selection working. **Not supported.**

**DeepHit has a ranking advantage that calibration erodes** — the original niche. Untuned, there was no advantage to erode. Tuned, the advantage exists only in the crisis regime.

---

## Alternative explanations we closed off

| Explanation | Test | Verdict |
|---|---|---|
| It's noise | paired bootstrap, B=500, Bonferroni α/3 | closed — effects are real |
| Too few / too linear features | added 54 states + product type (9 → ~70 features) | closed — **Cox gained most** |
| Not enough signal or shift | crisis cohort, 10.5% default vs 4.5%, structural break | closed — DeepHit still doesn't dominate |
| DeepHit was untuned | 20-config search, out-of-time inner validation | **partly valid** — tuning won the crisis cohort |

---

## What is shaky — say these out loud in the report

1. **Only DeepHit has a seed study.** Cox, Fine–Gray and RSF are single-fit, so they have no training-variability estimate.
2. **We never recalibrated.** Alberge et al. (2026) show AJ-recalibration and temperature scaling improve competing-risks probabilities *while preserving discrimination*. Our calibration penalty describes DeepHit **as trained**, not a ceiling.
3. **One dataset family.** US agency mortgages only; no clinical replication, despite two of our three reviews being clinical.
4. **Origination-time covariates only.** No time-varying macro variables — which is precisely Bellotti & Crook's contribution.
5. **With 150,000 test subjects almost everything is "significant".** Effect sizes against seed noise are what matter; several significant differences are practically negligible.
6. **Our RSF is cause-specific, not Ishwaran's** competing-risks formulation that Frydman & Matuszyk used.
7. **No per-cause D-calibration implemented.** Alberge's two measures exist and would be the right thing to add.

---

## Two things that changed during the work

Worth recording, because both were caught by checks rather than by luck.

**"DeepHit simply loses" was premature.** That claim came from untuned models. Tuning reversed it in the crisis cohort. The protocol lesson is real: an untuned deep baseline is not evidence.

**The selection-criterion effect was overstated.** It looked consistent across four cells until seed variability was quantified, at which point three of them fell inside the noise.

---

## Where this leaves the paper

The defensible headline is narrower and more interesting than the original niche:

> On mortgage data, under one protocol with time-ordered splits and an equal tuning budget, deep competing-risks models offer no reliable advantage over classical or tree-based baselines, are consistently worse calibrated, cost ~2,000× more to fit, and their signature ranking loss is counterproductive.

It engages all three reviews: Filip's tuning-budget and cost gap, Ali's calibration measures, Amirreza's competing-risks credit setting.
