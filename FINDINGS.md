# What we have found

*Updated 12 Oct 2026. For Amirreza, Ali Emre and Filip. Every number is traceable to a file in `results/`.*

## The one-sentence answer

Under a common protocol with time-ordered splits and an equal tuning budget, **DeepHit's advantage over classical competing-risks models exists only in the crisis regime, is no larger than the baseline's own training noise, never beats a random survival forest, and comes with a calibration penalty in every condition we tested.**

---

## Solid findings

### 1. The literature review's headline number replicates
Applying Dirick's two event definitions to 300,000 Freddie Mac loans:

| | censoring rate |
|---|---|
| prepayment counted as an event | **20.6%** |
| prepayment censored | **95.5%** |

Dirick et al. report 20–85% and 94.56–98.16% on ten European consumer-credit portfolios. Our figures fall inside both ranges on US agency mortgages, a different decade and a different product.

### 2. Prepayment dominates the data
75.0% prepaid, 4.5% default, 20.6% censored — prepayment outnumbers default **17 to 1**. Censoring it discards three quarters of the observed outcomes.

### 3. The definition of default changes the event rate 14×
90+ days past due gives 4.47%; the disposition-based zero-balance credit event gives 0.31%.

### 4. The advantage reverses with regime — but read the margin carefully
Tuned DeepHit vs cause-specific Cox, AUC at 60 months, default:

| cohort | DeepHit | Cox | diff | significant |
|---|---|---|---|---|
| crisis (test 2007–09) | **0.8538** | 0.8346 | **+0.0192** | yes |
| post-crisis (test 2016–18) | 0.7280 | **0.7372** | **−0.0091** | yes |

**Caveat that matters** (see finding 8): the crisis gap of +0.0192 is barely one standard deviation of the Cox model's *own* training variability in that cohort (±0.0159). The post-crisis result is clean — every training SD there is below 0.0022.

### 5. The calibration penalty does *not* reverse
ICI is **+0.0040 worse than Cox in both cohorts**. Whatever the regime, the deep model's probabilities sit further from observed frequencies.

### 6. Nothing is gained over a random survival forest
AUC statistically indistinguishable: **p = 0.968** (crisis), **p = 0.788** (post-crisis), both inside seed noise. On **crisis prepayment** the forest is clearly better: AUC 0.685 vs 0.620, ICI 0.154 vs 0.236.

### 7. DeepHit's ranking term is harmful — the strongest single result
Across 40 configurations and both cohorts, increasing the weight on the ranking loss makes **both** discrimination and calibration worse, monotonically:

| α | val AUC | val IBS |
|---|---|---|
| 0.10 | 0.658 | 0.0334 |
| 0.20 *(pycox default)* | 0.749 | 0.0286 |
| 0.50 | 0.776 | 0.0269 |
| 1.00 *(pure likelihood)* | **0.776** | 0.0271 |

The reference implementation's default sits inside the harmful range, and DeepHit's own paper selects this weight on validation *discrimination*.

### 8. The classical models are the unstable ones — not the deep one
Five bootstrap resamples of the training vintages, evaluated on the fixed test set, default AUC at 60 months:

| cohort | model | mean | **SD** |
|---|---|---|---|
| crisis | Cox | 0.8167 | **0.0159** |
| | Fine–Gray | 0.8316 | **0.0206** |
| | RSF | 0.8513 | 0.0009 |
| | DeepHit | 0.8502 | 0.0035 |
| post-crisis | all four | — | 0.0004–0.0022 |

In the high-default crisis regime, Cox and Fine–Gray vary four to six times more than DeepHit across resampled training sets. This was the opposite of what we expected and it is what forced the caveat in finding 4.

### 9. Recalibration cannot fix the calibration penalty
A marginal per-horizon rescaling, estimated two ways. ICI at 60 months:

| | raw | recal (train) | recal (oracle) |
|---|---|---|---|
| crisis, default, DeepHit | 0.0209 | 0.0164 | **0.0053** |
| post-crisis, default, DeepHit | 0.0318 | 0.0297 | **0.0082** |
| crisis, prepaid, RSF | 0.1539 | 0.1669 | **0.0310** |

Fitted on the **training** vintages — all a lender has at scoring time — it changes almost nothing and in **10 of 16 cells makes calibration worse**. Fitted on the **test** vintages, an oracle nobody can build, it cuts error by a median factor of **4.2** (range 1.0–14.7).

**So the penalty is not a broken calibration map. It is the vintage shift itself: correctable in principle, unavailable in practice.** AUC is identical to four decimals raw vs recalibrated in all 16 cells, confirming the rescaling is monotone.

### 10. Cost spans three orders of magnitude
Fine–Gray 0.03 s, Cox 0.42 s, RSF 36 s, DeepHit 59 s per fit on 150,000 loans — roughly **2,000×** — plus 14.4 minutes of hyperparameter search the classical models never need.

---

## Hypotheses tested and **not** confirmed

**Selecting on discrimination costs calibration.** Pre-registered, tested on both cohorts and both model families. Three of four cells fall within seed noise; only the crisis AUC effect (+0.0101) is robust, and that is selection working as intended. **Not supported.**

**DeepHit has a ranking advantage that calibration erodes** — the original niche. Untuned there was no advantage to erode; tuned, it exists only in the crisis regime and is within the baseline's training noise.

---

## Alternative explanations closed off

| Explanation | Test | Verdict |
|---|---|---|
| It's noise | paired bootstrap, B=500, Bonferroni α/3 | closed |
| Too few / too linear features | 9 → ~70 features (54 states + product type) | closed — **Cox gained most** |
| Not enough signal or shift | crisis cohort, 10.5% default vs 4.5% | closed |
| DeepHit was untuned | 20-config search, out-of-time inner validation | **partly valid** — tuning won the crisis cohort |
| Miscalibration is a fixable calibration map | recalibration, train vs oracle | closed — it is the shift |

---

## What is still shaky

1. **One dataset family.** US agency mortgages only; no clinical replication, despite two of three reviews being clinical. SEER access was requested but never arrived.
2. **Five bootstrap replicates** is a coarse variability estimate, and it conflates sampling with optimisation noise.
3. **Origination-time covariates only.** No time-varying macroeconomic variables — verified as Bellotti & Crook's contribution, so this is a real gap against the literature.
4. **Our recalibration is a marginal rescaling**, not the per-cause procedures of Alberge et al. (2026). A stronger correction might close more, though the oracle column suggests the ceiling is set by what is knowable at scoring time.
5. **No per-cause calibration measure implemented.** Alberge's CR-D-calibration and cal-αK-calibration exist and would be the right addition.
6. **With 150,000 test subjects almost everything is "significant".** Effect sizes against training noise are what matter.
7. **Our RSF is cause-specific**, not Ishwaran's competing-risks formulation used by Frydman & Matuszyk.

---

## Three things that changed during the work

All three were caught by checks rather than luck, which is itself an argument for the protocol.

**"DeepHit simply loses" was premature.** That came from untuned models; tuning reversed it in the crisis cohort. An untuned deep baseline is not evidence.

**The selection-criterion effect was overstated.** It looked consistent across four cells until seed variability was quantified, at which point three fell inside the noise.

**"Well outside seed variation" was wrong.** The crisis win was measured against DeepHit's seed noise while ignoring Cox's, which turned out to be 4.5× larger. Finding 8 exists because of that error.

---

## Where this leaves the paper

> On mortgage data, under one protocol with time-ordered splits and an equal tuning budget, deep competing-risks models offer no reliable advantage over classical or tree-based baselines, are consistently worse calibrated, cost roughly 2,000× more to fit, and their signature ranking loss degrades both of the things it is meant to improve. The calibration gap is not a correctable artefact of the model but a consequence of the regime shift between origination vintages.

It engages all three reviews: Filip's tuning-budget and cost gap, Ali's calibration measures, Amirreza's competing-risks credit setting.
