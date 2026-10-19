# What we have found

*Updated 12 Oct 2026 (α ablation added). For Amirreza, Ali Emre and Filip. Every number is traceable to a file in `results/`.*

## The one-sentence answer

Under time-ordered evaluation **DeepHit has no reliable advantage over classical or tree-based competing-risks models in any cell**, and the advantage it shows under the literature's protocol is **substantially a property of being trained on loans originated alongside the loans it is tested on**. Its calibration penalty, unlike its ranking advantage, does not depend on the protocol.

*Superseded 19 Oct: the earlier one-sentence answer said the advantage "exists only in the crisis regime". Rolling origin shows that was one fold.*

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

### 4. ~~The advantage reverses with regime~~ — RETRACTED 19 Oct

The original claim rested on **one time-ordered split per cohort**. Four rolling-origin folds per cohort give, for DeepHit minus cause-specific Cox on crisis default:

| test vintage | 2006 | 2007 | 2008 | 2009 |
|---|---|---|---|---|
| ΔAUC | +0.0013 | +0.0034 | +0.0061 | **+0.0185** |

Mean +0.0073, sd 0.0077. **The published +0.0192 was the 2009 fold** — the most extreme of four.

Across all eight cause-by-cohort cells the ratio of mean advantage to its spread across folds lies between −1.24 and +0.95. **No model leads reliably, in either direction, anywhere.** See `results/protocol_decomposition.csv`.

### 5. The calibration penalty does *not* reverse — but only the sign replicates
ICI is **+0.0040 worse than Cox in both cohorts**. Whatever the regime, the deep model's probabilities sit further from observed frequencies.

**Caveat added by the audit of 12 Oct.** Measured against the two models' combined training variability, the two cohorts carry this very differently:

| cohort | Cox ICI sd | DeepHit ICI sd | combined | +0.0040 is |
|---|---|---|---|---|
| post-crisis | 0.0001 | 0.0002 | 0.0002 | **17.7 σ** |
| crisis | 0.0020 | 0.0012 | 0.0023 | **1.7 σ** |

The direction holds in both, which is what "does not reverse" claims. The crisis *magnitude* is about as weakly supported as the crisis AUC win in finding 4, and now carries the same hedge in the report.

**Confirmed by proper measures, 20 Oct.** The decile index above is not a proper calibration measure for competing risks — it is per-cause and evaluated at one horizon, whereas what must be calibrated is the whole CIF vector across events and time. Both proper measures from Alberge et al. (2026) are now computed, and **all three instruments independently rank DeepHit worst in both cohorts**:

| model | AJ-cal crisis | AJ-cal post | CR-Dcal crisis | CR-Dcal post |
|---|---|---|---|---|
| Aalen–Johansen | 0.0096 | 0.0129 | 0.0084 | 0.0980 |
| Cox | 0.0173 | 0.0075 | 0.0224 | 0.0795 |
| Fine–Gray | 0.0135 | 0.0081 | 0.0180 | 0.0902 |
| RSF | 0.0108 | 0.0092 | 0.0101 | 0.1027 |
| **DeepHit** | **0.0251** | **0.0125** | **0.0270** | **0.1286** |

On crisis data DeepHit's CR D-calibration is **3.2× the Aalen–Johansen estimator's** — a model with no covariates at all produces better probabilities than the deep network.

**Two things not to misread.** The Aalen–Johansen row is not a modelling failure: that estimator is marginally calibrated by construction and scores 1.1×10⁻⁵ (AJ-cal) and 2.5–3.9×10⁻⁵ (CR-Dcal) on a *random* split, so its vintage-split values are the distribution shift measured directly — a ~1000× and ~340–2500× excess. And the KS test accompanying CR D-calibration **rejects every model in every cell**, including that estimator on the random split where it is correct by construction; with 50,000 test subjects it has power to reject on negligible deviations. The score is informative; the test, at this n, is not.

CR D-calibration is not in any released version of `hazardous` — it comes from the authors' open pull request (`soda-inria/hazardous` PR 100, branch `dcal`, commit `26e6432`), which the report states explicitly. This is the one result in the paper that depends on unreleased code.

### 6. Nothing is gained over a random survival forest
AUC statistically indistinguishable: **p = 0.968** (crisis), **p = 0.788** (post-crisis), both inside seed noise. On **crisis prepayment** the forest is clearly better: AUC 0.685 vs 0.620, ICI 0.154 vs 0.236.

### 7. The loss weight is not a reliable knob — and a lesson about marginals
Pooling the 40 tuning configurations by α appeared to show a strong monotone effect: validation AUC rising 0.658 → 0.776 as α goes 0.1 → 1.0. **That did not survive control.** In the search, α varied alongside depth, dropout, learning rate, batch size and bin count.

A controlled ablation — everything else fixed, 5 seeds, both cohorts, test vintages:

| cause | cohort | AUC α=0.1 → α=1 | Δ | seed sd | ICI Δ |
|---|---|---|---|---|---|
| default | post-crisis | 0.7263 → 0.7267 | +0.0005 | 0.0018 | +0.0114 |
| default | crisis | 0.8417 → 0.8492 | +0.0075 | 0.0033 | −0.0010 |
| prepaid | post-crisis | 0.6352 → 0.6738 | +0.0386 | 0.0062 | −0.0397 |
| prepaid | crisis | 0.6509 → 0.5836 | **−0.0673** | 0.0103 | +0.0345 |

**The effect reverses sign across cohorts and causes.** Three of four cells exceed seed noise and they disagree on direction. There is no setting of α we can recommend.

We also ruled out the competing explanation that DeepHit's discretisation horizon drives this: varying t_max over 72, 120 and 160 months leaves the α effect flat (`results/tmax_ablation.csv`).

The more useful result is methodological: **a marginal taken over a random search is not an ablation.** Reading one as the other is how a hyperparameter acquires a reputation it has not earned.

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

### 10. The advantage comes from training alongside the test loans

Same test loans, same training-set size; only the training *period* moves. One arm trains on earlier vintages, the other on the remaining half of the later ones.

**Training contemporaneously is worth +0.006 to +0.130 AUC.** All sixteen bonuses are positive with bootstrap intervals clear of zero, and DeepHit's is the largest of the four models in three of four cells.

The sharper quantity is the **difference in differences** — how much DeepHit's lead over a baseline changes between arms:

| cohort | cause | vs Cox | vs RSF |
|---|---|---|---|
| post-crisis | default | +0.0143 | +0.0100 |
| post-crisis | prepaid | +0.0142 | +0.0084 |
| crisis | default | **−0.0045** | +0.0108 |
| crisis | prepaid | +0.0192 | +0.0787 |

**All eight significant** at Bonferroni 0.05/3, paired bootstrap over the shared test loans, B=500. Against the forest it is positive in every cell: DeepHit's standing improves with contemporaneous training, and that is not a sampling artefact. The one negative cell is crisis default against Cox, where the classical model gains more.

So the deep model is not better at predicting forward; it is better at exploiting period-specific structure. See `results/matched_bootstrap.csv`, `src/run_matched_bootstrap.py`.

### 11. Cost spans three orders of magnitude
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
5. ~~No per-cause calibration measure implemented.~~ **Resolved 20 Oct** — both AJ-calibration and CR D-calibration are now computed and reported. What remains unimplemented is Alberge's *correction* methods (AJ-recalibration, temperature scaling), which is point 4 above. One residual dependency: CR D-calibration runs on an unreleased PR branch, pinned by commit.
6. **With 150,000 test subjects almost everything is "significant".** Effect sizes against training noise are what matter.
7. **Our RSF is cause-specific**, not Ishwaran's competing-risks formulation used by Frydman & Matuszyk.
8. **The bootstrap cannot resolve p below 1/500.** 31 of the 36 computed tests sit on that floor; they are reported as `< 0.002`. Bonferroni was applied within a family of three metrics, not across all 36 tests. Neither touches the negative findings, which rest on p = 0.968 and 0.788.
9. **Our calibration index is decile-binned**, not the smoothed integrated calibration index of Austin & Steyerberg (2019) or its competing-risks form (Austin et al. 2022). The report now says so and cites both.
10. **Censoring is nearly absent at the horizons we evaluate.** G(60) = 0.9986 in both cohorts, so the IPCW weights are all near one. The competing risk, not censoring, is what shapes these results — see `AUDIT.md`.

---

## Three things that changed during the work

All three were caught by checks rather than luck, which is itself an argument for the protocol.

**"DeepHit simply loses" was premature.** That came from untuned models; tuning reversed it in the crisis cohort. An untuned deep baseline is not evidence.

**The selection-criterion effect was overstated.** It looked consistent across four cells until seed variability was quantified, at which point three fell inside the noise.

**The α result was confounded.** The pooled tuning table suggested the ranking loss weight was the dominant hyperparameter and monotonically harmful. A controlled ablation showed the effect is small and sign-inconsistent — the pooled view had confounded α with the architectures it co-occurred with.

**"Well outside seed variation" was wrong.** The crisis win was measured against DeepHit's seed noise while ignoring Cox's, which turned out to be 4.5× larger. Finding 8 exists because of that error.

---

## Where this leaves the paper

> On mortgage data, under one protocol with time-ordered splits and an equal tuning budget, deep competing-risks models offer no reliable advantage over classical or tree-based baselines, are consistently worse calibrated, cost roughly 2,000× more to fit, and the strong effect their loss weight appears to have in a pooled hyperparameter search does not survive controlled variation. The calibration gap is not a correctable artefact of the model but a consequence of the regime shift between origination vintages.

It engages all three reviews: Filip's tuning-budget and cost gap, Ali's calibration measures, Amirreza's competing-risks credit setting.
