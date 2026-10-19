# The whole project, from scratch

*Written 16 Oct 2026 for Amirreza, Ali Emre and Filip. Assumes no prior knowledge.
Read it top to bottom once; after that the other documents make sense.*

---

# Part 1 — The ideas you need before anything else

## 1.1 Credit scoring used to be a yes/no question

The classic question a bank asks is: **will this borrower default in the next
twelve months?** That is a classification problem. You train a model, it outputs
a probability, you set a cutoff, done.

Accounting rules changed the question. Under **IFRS 9**, a lender has to set
aside money today for losses expected over the *remaining lifetime* of a loan.
That needs something harder: the probability that the loan defaults **by each
future date** — month 12, month 24, month 60 — not a single number for one
horizon.

That is what **survival analysis** gives you: models of *time until an event*
rather than *whether an event happens*.

## 1.2 Censoring: the thing survival analysis exists to handle

Your data ends at some point. Loans that are still alive when the data ends
haven't defaulted *yet* — you don't know what they will do. Throwing them away
biases everything (you'd keep only loans that ended). Treating them as
"no default" is also wrong.

A **censored** observation means: "this loan survived at least this long, and
then we stopped watching." Survival analysis uses that partial information
correctly. This is the single idea the whole field is built on.

## 1.3 Competing risks: the idea this project is about

A mortgage can end in three ways:

1. **Default** — the borrower stops paying.
2. **Prepayment** — the borrower pays it off early (sells the house,
   refinances).
3. **Maturity** — it runs its full term.

**Once one happens, the others cannot.** A loan that was paid off in month 20
can never default in month 40. It is gone.

These are **competing risks**. And here is the mistake the credit literature
keeps making:

> Most studies treat prepayment as **censoring** — as if it meant "we stopped
> watching, the loan might still default later."

It doesn't mean that. The loan is *finished*. Treating it as censoring tells the
model that those loans were still at risk of default when they weren't, which
**biases the estimated default probability upward**.

### The number that makes this concrete

Dirick et al. (2017) benchmarked eight survival models on ten European
consumer-credit portfolios. Seven of the eight censor early repayment. On the
very same loans, that one modelling choice moves the censoring rate from
**20–85%** (prepayment counted as a real event) to **94.56–98.16%**
(prepayment censored).

Read that again: under the common approach, **95%+ of your data is "censored"**
— almost no information. That is the motivating fact of this entire project.

## 1.4 The right quantity: the cumulative incidence function

In ordinary survival analysis, P(event by time *t*) = 1 − S(*t*) where S is the
survival curve. **Under competing risks that formula is wrong.**

What you want instead is the **cumulative incidence function (CIF)**:

> CIF_default(*t*) = P(the loan defaults by time *t*, **and** default is what
> happened to it first)

Every model in this project is required to output a CIF per cause. That is the
common currency.

## 1.5 Ranking versus calibration — the heart of the project

There are two completely different ways a prediction can be good.

**Ranking (discrimination).** Does the model put riskier borrowers above safer
ones? Measured by **AUC**: pick a loan that defaulted and one that didn't; AUC
is the probability the model scored the defaulter higher. 0.5 = coin flip,
1.0 = perfect ordering.

**Calibration.** Are the numbers *true*? If the model says 5%, do about 5% of
those loans actually default?

**These are independent.** Take any well-calibrated model and halve every
prediction. The ranking is identical — AUC unchanged. The probabilities are now
all wrong by a factor of two.

Why it matters here: **a lender computes its reserves from the probability, not
from the ordering.** A model that ranks beautifully and says 2% when the truth
is 5% will under-reserve, and no amount of good ranking fixes that.

And yet: **the deep-learning survival literature reports ranking almost
exclusively.** That is the gap this project attacks.

## 1.6 The models

| model | what it is |
|---|---|
| **Aalen–Johansen** | Not a model — the non-parametric estimate of the CIF with no covariates. Our *null baseline*: it cannot rank at all (AUC exactly 0.5) but is calibrated by construction. Having it catches bugs. |
| **Cause-specific Cox** | One classical Cox regression per cause, then combined into a CIF. The workhorse. |
| **Fine–Gray (1999)** | Models the "subdistribution hazard" directly — keeps competing-event subjects in the risk set with a decaying weight. The standard competing-risks regression. |
| **Random survival forest (RSF)** | A tree ensemble. Non-linear, no parametric assumptions. |
| **DeepHit (Lee et al. 2018)** | The neural network. Outputs a joint distribution over (cause, time) directly. |

### DeepHit's loss — remember this, it becomes finding 7

DeepHit trains on a weighted sum of two objectives:

> **loss = α · (likelihood term) + (1 − α) · (ranking term)**

The α knob trades "get the probabilities right" against "get the order right".
α = 1 removes the ranking term entirely. The library's default is α = 0.2, which
puts most of the weight on *ranking*.

That is a telling design choice given §1.5.

---

# Part 2 — Assessment 1: the literature review

**Submitted around 5 October 2026.** A 5-page mini systematic review on
competing-risk survival models in credit risk.

It went through a lot of shaping: 12 papers down to 9, several LaTeX templates,
a full rewrite into plain language, and a hunt for whether anyone had already
written the same review. The final claim structure identified gaps in the
literature, including:

- Studies that censor prepayment rather than model it as a competing event.
- Models selected and reported on ranking, with calibration rarely checked.
- No standard measure for per-cause CIF calibration.

**One gap was falsified during checking.** The original claim — "there is no
off-the-shelf measure for per-cause CIF calibration" — turned out to be false:
Gerds-style pseudo-value calibration curves exist, and some studies do report
calibration-sensitive measures. It was rewritten as a weaker, true *separation*
claim: ranking and calibration are rarely reported **together**.

That episode matters because it set the standard for everything after: **check
your own claims hard enough to break them.**

---

# Part 3 — Assessment 2: what we are actually doing

## 3.1 The brief

From Lecture 1: 2–3 students, combine your findings, **fill one of the
identified gaps**, a concrete research question, a report and a poster.

**The official brief was published mid-project** (`docs_brief_RTDS2026.pdf`) and
changed two things: the deliverable is a **6-page report** with references
excluded from the limit, and **there is no poster**. The poster that had been
drafted was removed on 20 Oct; it is in git history if it is ever wanted again.
Everything before that point was built from Lecture 1 and Lecture 8.

## 3.2 How three reviews became one question

All three of you had written separate reviews. They converged:

- **Amirreza** — competing risks in credit.
- **Ali Emre** — calibration measures.
- **Filip** — not a fourth topic, but the *method* for testing the other two:
  protocol, metric sets, cost.

Which gave one question:

> **Deep competing-risks models are selected and reported on ranking. Does their
> advantage survive when calibration and cost are measured too?**

Ownership: Filip the protocol and cost, Ali the calibration measures, Amirreza
the competing risks and the credit setting.

## 3.3 The data

Two datasets were wanted — one clinical, one credit.

- **SEER** (cancer registry): requested, **never arrived**. The clinical half of
  the group's reviews therefore has no empirical representation. This is declared
  as a limitation rather than papered over.
- **Freddie Mac Single-Family Loan-Level Dataset**: obtained. Free for academic
  use, **cannot be redistributed** — which is why `data/` is excluded from git.

Two cohorts were built deliberately:

| | vintages | loans | default rate |
|---|---|---|---|
| **Post-crisis** | 2013–2018 | 300,000 | 4.5% |
| **Crisis** | 2005–2009 | 250,000 | 10.5% |

Two cohorts, because one of them is a *stress test*: if deep models need a
high-signal, high-default regime to shine, that will show up as a difference
between the two.

**Event definitions:** default = first month at 90+ days past due; prepayment =
zero-balance code 01; time = **loan age in months**, so the clock starts at
origination for every loan.

### The replication that justifies the whole framing

Applying Dirick's two definitions to our 300,000 US mortgages:

| | censoring rate |
|---|---|
| prepayment counted as an event | **20.6%** |
| prepayment censored | **95.5%** |

Dirick reported 20–85% and 94.56–98.16% on European *consumer credit*. Ours sit
at the lower edge of the first range and inside the second — a different
product, a different country, a different decade, and the same effect. That is
real independent replication, and it is what the report leads with.

---

# Part 4 — The decisions, and why each one was made

This is the part that is hardest to reconstruct later, so here is the reasoning
behind each choice.

## Decision 1: build the evaluation harness *before* any model

**Why.** If you build models first and metrics second, you will — honestly and
unconsciously — pick metrics that make your results look clean. Fixing the
measuring stick first removes that freedom.

Concretely: `src/metrics.py` and `src/harness.py` were written, unit-tested and
validated against a null model before a single Cox model was fitted.

## Decision 2: one common interface for every model

Every model must implement `fit(X, time, event)` and
`predict_cif(X, horizons, cause)`. Nothing else.

**Why.** It makes it structurally impossible to give one model a metric the
others can't have. If DeepHit got a metric Cox couldn't be scored on, any
comparison would be meaningless.

## Decision 3: vintage-ordered splits, not random splits

Train on loans originated 2013–2015, test on 2016–2018. **Never** shuffle.

**Why.** A lender fits a model on loans it has already seen and applies it to
loans it hasn't originated yet. A random split leaks the future into the
training set: the model sees 2018 loans while predicting other 2018 loans.

Most of the literature uses random splits — Blumenstock et al. used ten of them.
So the gap between the two protocols is itself evidence about whether random
splits flatter deep models.

## Decision 4: a documented, pre-committed tuning budget

Twenty random configurations for DeepHit, eight for the forest, chosen on an
**out-of-time inner validation vintage held inside the training years** — so the
test vintages are never touched during tuning. Cox and Fine–Gray have no
comparable hyperparameters and are untuned.

**Why.** The standard way to make a deep model win is to tune it hard and leave
the baselines at defaults. Fixing the budget in advance and reporting it closes
that door. Note the budget is *not equal* — the deep model gets **more** tries,
which is the conservative direction for a conclusion that it doesn't win.

## Decision 5: pre-register the hypotheses

Before looking at any test result: two-sided nulls, α = 0.05, Bonferroni
correction, and the specific comparisons written down (DeepHit vs Cox; DeepHit
vs RSF; DeepHit selected on AUC vs on Brier).

**Why.** Otherwise you look at the results and then decide which comparison was
"the interesting one". That is how noise becomes a finding.

## Decision 6: implement Fine–Gray by hand

No mainstream Python library has it.

**Why it was feasible:** the naive implementation expands the dataset and is
O(n²). The version here splits the risk-set sum into a reverse and a forward
cumulative sum, making it O(n) — it fits all 300,000 loans in 0.03 seconds with
no data expansion.

## Decision 7: measure variability three different ways

This turned out to be the most important decision in the project.

1. **Bootstrap over the test set** — "would a different sample of test loans
   change the answer?"
2. **Seed variation** — refit the same model on the same data with different
   random seeds.
3. **Training resamples** — refit each model on bootstrap resamples of the
   *training* vintages.

**Why all three.** A p-value from (1) holds the fitted model fixed, so it
*understates* uncertainty. (2) and (3) catch the rest. And as you'll see in
Part 5, (3) overturned a headline claim.

---

# Part 5 — What we found

## Finding 1 — The censoring contrast replicates

20.6% vs 95.5%, inside Dirick's ranges, on completely different data. (§3.3)

## Finding 2 — Prepayment dominates

75% of post-crisis loans prepay; 4.5% default. **Prepayment outnumbers default
17 to 1.** Censoring it throws away three quarters of the observed outcomes.

## Finding 3 — The definition of "default" changes the rate 14×

90+ days past due gives 4.47%. The disposition-based definition gives 0.31%.
Same loans. A definitional choice moves the headline number by an order of
magnitude.

## Finding 4 — ~~DeepHit's advantage depends on the economic regime~~ **RETRACTED 19 Oct**

This was the headline for weeks and it did not survive. It rested on **one
time-ordered split per cohort**. Running four rolling-origin folds instead —
expanding the training window a vintage at a time — gives, for DeepHit minus
cause-specific Cox on crisis defaults:

| test vintage | 2006 | 2007 | 2008 | 2009 |
|---|---|---|---|---|
| ΔAUC | +0.0013 | +0.0034 | +0.0061 | **+0.0185** |

Mean +0.0073, standard deviation 0.0077 across folds. **The +0.0192 we
published was essentially the 2009 fold** — the most extreme of four.

Across all eight cause-by-cohort cells, the ratio of mean advantage to its
spread across folds lies between −1.24 and +0.95. **No model leads reliably in
any cell, in either direction.** The regime story was an artefact of sampling
one split.

## Finding 5 — The calibration penalty does not reverse

DeepHit's calibration index is **0.0040 worse than Cox in both cohorts**.
Whatever the regime, its probabilities sit further from observed frequencies.

**Hedged after the audit:** post-crisis that gap is ~18× the models' combined
training variability; on crisis data only ~1.7×. **The sign replicates across
regimes; the size does not.**

**Strengthened 20 Oct.** The decile index is not a proper measure for competing
risks, so this finding originally rested on the wrong instrument. Both proper
measures from Alberge et al. (2026) now sit beside it — AJ-calibration
(marginal over events and time) and CR D-calibration (distributional) — and
**all three rank DeepHit worst in both cohorts**. On crisis data DeepHit's CR
D-calibration is 0.0270 against 0.0084 for the Aalen–Johansen estimator: a
model with no covariates produces better probabilities than the deep network.
This is now the most robust finding in the paper, because it is the only one
confirmed by three independent instruments.

## Finding 6 — Nothing is gained over a random survival forest

| cohort | p-value (DeepHit vs tuned RSF, AUC) |
|---|---|
| crisis | **0.968** |
| post-crisis | **0.788** |

Statistically indistinguishable, and both differences are smaller than DeepHit's
own seed-to-seed variation. On **crisis prepayment** the forest is clearly
better: AUC 0.685 vs 0.620, calibration 0.154 vs 0.236.

This is the most damaging result for the deep model, and it is a *negative*
result — which is why it survives every later correction untouched.

## Finding 7 — The loss weight α is not a reliable knob (and a lesson)

Pooling the 40 tuning configurations by α appeared to show a clean monotone
effect: validation AUC rising **0.658 → 0.776** as α goes 0.1 → 1.0.

**It did not survive control.** In the tuning search, α varied *alongside* depth,
dropout, learning rate, batch size and bin count. So that curve was measuring all
of them at once.

A controlled ablation — everything else fixed, 5 seeds, both cohorts:

| cause | cohort | ΔAUC (α 0.1→1) | seed sd |
|---|---|---|---|
| default | crisis | +0.0075 | 0.0033 |
| default | post | +0.0005 | 0.0018 |
| prepaid | crisis | **−0.0673** | 0.0103 |
| prepaid | post | **+0.0386** | 0.0062 |

**The effect reverses sign across causes.** There is no α we can recommend.

> **The transferable lesson: a marginal taken over a random search is not an
> ablation.** Reading one as the other is how a hyperparameter acquires a
> reputation it has not earned.

We also ruled out the competing explanation that DeepHit's discretisation
horizon drives this (flat across t_max of 72, 120, 160 months).

## Finding 8 — The classical models are the unstable ones

Crisis cohort, five training resamples, default AUC:

| model | SD |
|---|---|
| Cox | **0.0159** |
| Fine–Gray | **0.0206** |
| RSF | 0.0009 |
| DeepHit | 0.0035 |

Cox and Fine–Gray vary **four to six times more** than DeepHit across resampled
training sets. This was the opposite of what we expected, and it is what forced
the caveat on finding 4.

## Finding 9 — Recalibration cannot fix the calibration gap

We fitted a per-horizon rescaling two ways.

- Fitted on the **training** vintages (all a lender actually has at scoring
  time): changes almost nothing, and makes calibration **worse in 10 of 16
  cells**.
- Fitted on the **test** vintages (an oracle nobody can build): cuts error by a
  median factor of **4.2**.

**So the penalty is not a broken calibration map.** If it were, the training-
fitted correction would work. It is **the vintage shift itself** — a rescaling
estimated on 2005–06 cannot know that 2007–09 would turn out worse. Correctable
in principle, unavailable in practice.

## Finding 10 — Cost spans three orders of magnitude

Fine–Gray 0.03 s, Cox 0.42 s, RSF 36 s, DeepHit 59 s per fit on 150,000 loans —
roughly **2,000×** — plus 14.4 minutes of hyperparameter search the classical
models never need.

## Finding 11 — The advantage comes from training alongside the test loans

**This is now the paper.** Same test loans, same training-set size; only the
training *period* moves. One arm trains on earlier vintages, the other on the
remaining half of the later ones.

Training contemporaneously is worth **+0.006 to +0.130 AUC**, and DeepHit's
bonus is the largest of the four models in three of four cells.

The sharper number is the **difference in differences** — how far DeepHit's lead
over a baseline moves between the two arms:

| cohort | cause | vs Cox | vs RSF |
|---|---|---|---|
| post-crisis | default | +0.0143 | +0.0100 |
| post-crisis | prepayment | +0.0142 | +0.0084 |
| crisis | default | **−0.0045** | +0.0108 |
| crisis | prepayment | +0.0192 | **+0.0787** |

**All eight are significant** (paired bootstrap over the shared test loans,
B = 500, Bonferroni 0.05/3). Against the forest it is positive in every cell.

So the deep model is not better at predicting forward. It is better at
exploiting structure specific to the period it trained on — and any protocol
that hands it that structure will report an advantage a lender never receives.

## The one-sentence answer

> Under time-ordered evaluation, deep competing-risks models have **no reliable
> advantage** over classical or tree-based baselines on mortgage data. The
> advantage they are reported to have is **substantially a property of being
> trained on loans originated alongside the loans they are tested on**. Their
> calibration penalty, unlike their ranking advantage, does not depend on the
> protocol.

# Part 6 — Alternative explanations we closed off

A negative result is only worth something if you've ruled out the boring reasons
for it. Each row here is an experiment that was run specifically to try to break
our own conclusion.

| "Your result is just…" | what we did | outcome |
|---|---|---|
| …noise | paired bootstrap, B = 500, Bonferroni | closed |
| …too few / too linear features | 9 → ~70 features (54 states + product type) | closed — **Cox gained the most** |
| …not enough signal or shift | added the crisis cohort (10.5% default) | closed |
| …an untuned deep model | 20-config search, out-of-time validation | **partly valid** — tuning won the crisis cohort |
| …a fixable calibration map | recalibration, train vs oracle | closed — it's the shift |
| …a discretisation artefact | t_max ablation (72/120/160 months) | closed |

---

# Part 7 — Is it novel?

Honestly: **no, it is confirmatory and incremental.** That was checked
deliberately rather than assumed.

- **da Silva & Alvares, "The C-index illusion"** overlaps heavily with the
  ranking-measure argument.
- **Alberge et al. (2026, AISTATS)** pre-empts the calibration finding — they
  show existing calibration measures are unsuited to competing risks, introduce
  two proper ones, and give recalibration methods.
- **Alberge et al. (2025)** does *not* analyse α, so finding 7 is not pre-empted
  there.

The project's defensible contributions are: the **credit setting**, the
**vintage-ordered protocol with cost measured alongside**, the **two-regime
comparison**, and **finding 7**, which is a methodological point that generalises
past this benchmark.

Saying this plainly in the report is better than being caught claiming novelty
that isn't there.

---

# Part 8 — Everything that went wrong, and how it was caught

This section exists because it is the most valuable thing in the project. **Eight
claims were wrong at some point.** None was caught by rereading the PDF; every
one was caught by recomputing from the data.

### Code bugs

1. **AUC was O(cases × controls)** — about 4 billion pairs for prepayment; it
   hung. Rewritten as a weighted Mann–Whitney in O(n log n).
2. **The Cox linear predictor was mis-scaled** — a max-subtraction for numerical
   stability was returned *after* subtraction, so it was always zero and the
   baseline hazard silently absorbed the error. Found by testing against
   simulated data with known coefficients.
3. **A step function returned 1.0 before its first knot** — correct for a
   survival curve, wrong for a cumulative hazard. It inflated one model's
   calibration error to 0.67.
4. **The build script silently overwrote the first dataset** with the second.
5. **The AUC's censoring weights were wrong** (found in the final audit) —
   competing-event controls should be weighted by G(T−), not G(horizon).
   Verified to change nothing here, because G(60) = 0.9986.

### Claims that were wrong

6. **"DeepHit simply loses."** Premature — that was from *untuned* models.
   Tuning reversed it in the crisis cohort. An untuned deep baseline is not
   evidence of anything.
7. **"The selection criterion consistently costs calibration."** Looked
   consistent across four cells until seed variability was quantified, at which
   point three of them fell inside the noise.
8. **"Well outside seed variation."** The crisis win was measured against
   *DeepHit's* noise while ignoring *Cox's*, which turned out to be 4.5× larger.
   Finding 8 exists because of that mistake.
9. **The α result was confounded** — the pooled tuning table had α varying
   together with the architectures it co-occurred with.
10. **§IV-E merged two different table cells** into one sentence (the factor of
    14.7 belongs to Fine–Gray on post-crisis default, not to the forest on crisis
    prepayment).

### The biggest one (19 Oct)

11. **"The advantage reverses with regime" was wrong.** It came from one
    time-ordered split per cohort. Four rolling folds showed the published
    +0.0192 was the most extreme fold of four, and that no model leads reliably
    anywhere. This is the single most important correction in the project, and
    it arrived only because a referee-style read asked what the uncertainty on
    a sample of one was supposed to be.

### The audit (12 Oct)

A deliberately adversarial re-reading found four more defects — the AUC weights
above, p-values printed at the bootstrap's resolution floor, a Bonferroni
denominator that didn't match the 36 tests actually computed, and an abstract
claiming an "equal" tuning budget that the method section contradicted. All are
fixed. **None of them moved a reported number**, because the central results are
negative ones that rest on effect sizes rather than p-values.

The audit also established two honest framing points:

- Our calibration index is **decile-binned**, not the smoothed integrated
  calibration index of Austin & Steyerberg. The report now says so and cites it.
- **Censoring is nearly absent at the horizons we evaluate** (G(60) = 0.9986).
  All the censoring machinery is correct but barely active; what actually shapes
  the results is the **competing risk**.

---

# Part 9 — What exists right now

| artefact | state |
|---|---|
| `report/main.pdf` | **6 body pages** (the limit, exactly), IEEE format, 4 tables + 1 figure |
| `src/` | full pipeline: data build, splits, metrics, harness, 4 models, tuning, significance, ablations, table/figure generation, 2 audit scripts |
| `results/` | every CSV behind every number in the paper |
| GitHub | https://github.com/ar-sadraei/rtds-project — pushed, data excluded |
| `FINDINGS.md` | the findings with their caveats |
| `AUDIT.md` | the methodological audit and what was fixed |
| `DATA.md` | dataset card |
| `PROJECT_PLAN.md` | phase status |

**Rule to keep:** every number in the report is generated by
`src/make_tables.py` and `src/make_figure.py` from `results/*.csv`. Nothing is
typed by hand. If a result changes, regenerate — never transcribe.

---

# Part 10 — What is left

### Must happen before submission

1. **Rewrite the prose in your own voice.** The draft is complete and every
   number is verified, but the rubric penalises text that reads as generated.
   §V-B especially. Read it aloud — that finds it faster than reading silently.
2. **Merge Filip's draft**: his §I–§II with our §III–§V. His file uses the 10pt
   `[conference]` class where ours is `[11pt,journal]`, and his DeepHit citation
   carries page numbers 2314–2321 that need checking against the proceedings.
3. **Ali and Filip must review everything**, especially:
   - the **α correction** (an earlier version claimed the opposite)
   - the **retraction of the regime headline** (19 Oct)
   - the **calibration section**, which was rebuilt on 20 Oct
   Each of you should be able to answer *"what is a calibration index, and why
   isn't it enough here?"* and *"why is the Fine–Gray risk set weighted that
   way?"* without reading off the paper.
4. **Make the Author Contributions statement true.** It describes a division of
   labour; whoever's name is on a sentence should confirm it.
5. **Confirm Alberge's page numbers** against Ali's copy.

### Worth doing if there is time

- ~~Apply Alberge's per-cause calibration measures~~ — **done 20 Oct**, both of
  them. What is still not applied is their *correction* methods
  (AJ-recalibration, temperature scaling); our recalibration remains a marginal
  per-horizon rescaling, and the paper says so.
- More than five bootstrap replicates.
- A clinical dataset, if SEER ever arrives.

### Note on the page budget

The body is at 6 pages exactly. Floats pin the page break, so cutting prose on
an earlier page does nothing — anything added now has to displace something on
the page it lands on.

---

# Part 11 — If someone challenges the work

**"Isn't your crisis win within noise?"**
Yes — say so first. It's ~1 SD of Cox's own training variability. The post-crisis
loss is the clean result, and the negative result against the forest is cleaner
still.

**"Why didn't you tune Cox?"**
It has no comparable hyperparameters. And the deep model got *more* budget than
the forest, which works against our own conclusion.

**"Your calibration measure isn't the real ICI."**
Correct, and the report says so: it's the decile-binned form, and Austin &
Steyerberg's smoothed version is cited.

**"This isn't novel."**
Agreed, and we say so in the paper. Alberge et al. (2026) pre-empts the
calibration argument. What we add is the credit setting, the cost axis, the
two-regime comparison, and the α result.

**"Everything is significant with 150,000 loans."**
Exactly — which is why we report effect sizes against training noise throughout,
and why the p-values are the least interesting column in the table.
