# Where the project stands

*For Filip, Ali Emre and Amirreza. 18 Oct 2026. Checked against
`RTDS_Project_Overview.pdf` (15 Oct).*

---

## 1. What we are doing, in plain terms

**The question, unchanged from our overview:** do deep survival models really
beat classical ones once they are judged on more than ranking?

A model can order borrowers perfectly and still attach the wrong probabilities
to them. Banks act on the probability, so we measure ranking, calibration and
compute cost together, and we test on loans the model has not seen *in time* —
not on a random slice of the same period.

**Data.** 550,000 Freddie Mac mortgages. Two outcomes that compete: default and
early repayment. Once one happens the other cannot, which is the whole reason
competing-risks models exist. Two eras: 2005–09 (crisis) and 2013–18 (calm).

**Models.** Cause-specific Cox, Fine–Gray, random survival forest, DeepHit.
Equal interface, documented tuning budget, everything fixed before we looked at
test results.

---

## 2. The three problems from our overview, and where each stands

### Problem 1 — deep vs classical models
**Done.** All four models, same data, same budget, several seeds, runtime
measured.

**Answer: no reliable advantage for the deep model.** Under proper evaluation
(see below) DeepHit does not reliably beat the Cox model or the forest in any
of eight cause-by-cohort cells. It costs roughly 2,000× more to train.

### Problem 2 — testing on later data
**Done, and this became the centre of the project.** We ran three things:

1. Random split vs time-ordered split (as planned).
2. **Rolling origin** — four time-ordered splits per era instead of one, so we
   can see how much the answer wobbles.
3. **A matched test** — same test loans and same training size in both arms,
   with only the *training period* moving.

**Answer: our hypothesis "all models drop, deep models drop more" is
supported.** Training on loans originated alongside the test loans is worth
between +0.006 and +0.130 AUC, and the bonus is bigger for DeepHit than for Cox
in three of four cells. DeepHit beats the forest in 4 of 4 cells when trained on
contemporaneous loans, and trails or draws in 4 of 4 when trained on earlier
ones.

### Problem 3 — ranking vs probabilities
**Mostly done.** We report IPCW time-dependent AUC, an integrated Brier score
on the cumulative incidence, and calibration curves against the Aalen–Johansen
estimate.

**Answer: the calibration penalty is real and does not reverse** — DeepHit's
calibration error is worse than Cox's in both eras. But our hypothesis that
*the ranking of models changes when calibration is added* was **not confirmed**.

---

## 3. Our three hypotheses, answered

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Deep models keep their ranking advantage on later data | **Refuted.** No reliable advantage in any cell. |
| H2 | Deep models lose more on calibration | **Supported.** Worse in both eras. |
| H3 | Any gain costs clearly more runtime | **Supported.** ~2,000×. |

Our overview said "any outcome is a result". Two of three hypotheses survived
and one was refuted, which is a result.

---

## 4. Where we follow the overview, and where we do not

### Followed
- The main question, word for word.
- The three-problem structure: model type, split type, metric type.
- Cause-specific Cox, Fine–Gray, random survival forest, DeepHit.
- Same tuning budget for all, fixed in advance, validated out-of-time.
- Time-ordered split **plus random split as control**.
- Several seeds, bootstrap intervals on the test set.
- Brier on cumulative incidence, calibration curves, runtime.
- All three hypotheses actually tested.

### Not followed — and Ali should see this list
1. **SEER never arrived.** The overview put SEER first and mortgage data second,
   with the fallback being "SEER only". We ended up with the exact inverse:
   mortgage only, SEER as future work. This is the biggest deviation.
2. **Competing-events D-calibration (Alberge et al. 2026) is not implemented.**
   It is in the overview's metric list. We use a decile-binned calibration index
   instead and say so in the paper. This is the clearest thing still missing,
   and it sits in Ali's area.
3. **No C-index.** The overview lists "standard and time-dependent C-index per
   event". We deliberately use IPCW time-dependent AUC instead, because Blanche,
   Kattan and Gerds (2019) show the C-index is improper for t-year risk — it can
   prefer a worse model. This is a defensible change but it *is* a change, and
   it needs a group decision rather than a silent substitution.
4. **Only one deep model.** The overview allowed for Neural Fine-Gray, Deep
   Survival Machines or SurvTRACE as well. We ran DeepHit alone.
5. **No bar chart of the drop per model.** We have the numbers in a table.

---

## 5. Two things that changed late, which everyone should know

**We found a factual error in our own related work.** We had written that
Blumenstock et al. (2022) evaluated on "ten random train/test splits". Reading
the actual paper, they use *temporal* splitting with random subsampling inside
it, and one of their experiments already trains before the crisis and tests in
2009. So time-ordered evaluation is **not** what makes our study different.

What does make it different: in the full text of that paper, "calibration"
appears **zero** times and "Brier" appears **zero** times. They measure
concordance alone. The calibration axis is genuinely ours.

**We had to retract our own headline.** We previously reported a significant
+0.0192 AUC advantage for DeepHit on crisis-era defaults. Running four rolling
splits instead of one shows the per-fold values are +0.0013, +0.0034, +0.0061
and +0.0185 — our published number was the single most extreme fold. The honest
summary is that there is no reliable advantage in either direction.

This is uncomfortable but it is the right outcome: it is direct evidence for the
project's own argument that evaluation protocol decides the answer.

---

## 6. What is left

- Rewrite the paper around the protocol finding as the single claim (in
  progress).
- **Group decision needed:** do we implement Alberge's D-calibration, or state
  it as future work? It is in our overview and it is currently missing.
- **Group decision needed:** do we add back a C-index for comparability with the
  literature, or keep the AUC-only choice and justify it?
- Everyone reads the final draft. The α result and the crisis-advantage result
  both changed after earlier drafts, so anything remembered from before is out
  of date.
- Author contributions statement needs to be made accurate before submission.

**Deliverable:** 6-page report, references excluded from the limit, IEEE
template. No poster.
