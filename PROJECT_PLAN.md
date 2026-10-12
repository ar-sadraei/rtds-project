# Assessment 2 — step-by-step plan

*Written 11 Oct 2026. Three people: Amirreza, Ali Emre, Filip.*

**Status of the brief.** The official guidelines are not posted yet (forum, 8 Oct: "I will make the instructions available by this week… the instructions were also provided in the lectures"). Everything here is built from Lecture 1 (what the deliverable is) and Lecture 8 (how to run and write it). Phases are numbered, not dated — re-anchor to real dates the moment the brief lands.

**What we know for certain**

- 2–3 students, combine your findings, **fill one of the identified gaps**
- A concrete research question, **theoretical or empirical**
- **5-page report + poster presentation**
- Same grading criteria as the literature review

**What we don't know:** deadline, poster size/format, whether references count toward 5 pages, whether a template is mandated, and whether a design without results is acceptable. Four forum questions in Phase 0 cover these.

---

## Phase 0 — Lock the scope (do this week)

Nothing downstream is safe until these are fixed.

### 0.1 Pick the one gap

All three reviews converge on the same thing. Written side by side:

| | Niche as submitted |
|---|---|
| Amirreza | Do deep **competing-risk** models keep their ranking advantage when judged on calibration of the per-cause CIF, on **credit** data, vintage-ordered splits? |
| Ali Emre | Do deep survival models keep their ranking advantage over Cox and RSF when also judged on **calibration**, on **clinical** data, time-ordered splits? |
| Filip | When do deep models outperform classical ones on **discrimination, calibration and runtime**, under a **common evaluation protocol**? |

Amirreza's and Ali's are the same gap in two domains. Filip's is not a fourth gap — it is the *method* for addressing them.

**The one gap: deep survival models are selected and reported on ranking, and it is not established that their advantage survives when calibration is also measured.**

### 0.2 Write the core-idea sentence

Lecture 8: "Be clear about the core idea of the paper and communicate this clearly." One sentence, agreed by all three, written down. Everything in the report must serve it. If a section doesn't, cut it.

Working version to argue about and replace:

> Under one fixed evaluation protocol, we test whether the ranking advantage of deep survival models over classical and tree-based baselines survives when calibration and runtime are measured alongside it.

### 0.3 Post the forum questions

He invited specific queries. These four change the plan:

1. Is the 5-page limit exclusive of references, as in Assessment 1?
2. Is a rigorous project *design* acceptable under "theoretical or empirical", or must we report results?
3. Our three reviews span clinical and credit data. May we address one gap across both domains, or should we stay in one?
4. Poster format and dimensions, and the submission deadline?

**Question 2 is the one that matters most.** If a design counts, Phase 2's data risk disappears.

### 0.4 Agree the division of labour

Each person owns the part their review already covers:

| Owner | Owns |
|---|---|
| **Filip** | The protocol: metric set, tuning procedure, runtime and hardware instrumentation |
| **Ali Emre** | Calibration: CR D-calibration (Alberge et al. 2026), D-calibration, IBS, censoring handling |
| **Amirreza** | Competing risks: cause-specific vs subdistribution setup, CIF, the credit dataset |

Shared: statistics, writing, poster.

**Done when:** one gap written down, core-idea sentence agreed, forum post submitted, owners assigned.

---

## Phase 1 — Write the protocol before any code

Lecture 8: "Start a project by writing the paper. Do not wait till last minute." That is literal. Write the Method section now; it will not change much, and writing it exposes decisions you'd otherwise discover late.

### 1.1 The protocol table

Fill this in completely before touching data. It becomes Table 1 of the report.

| Axis | Decision |
|---|---|
| Models | cause-specific Cox, Fine–Gray / Aalen–Johansen, Random Survival Forest, DeepHit (+ SurvTRACE if time) |
| Discrimination | concordance index (state the variant and, per Sonabend, the distribution→risk reduction used) |
| Calibration | CR D-calibration; IBS; one calibration plot |
| Cost | training time, inference time, hardware |
| Splits | time-ordered (diagnosis year / origination vintage), plus a random split as a contrast |
| Tuning | identical budget for every model; state the grid and the seeds |
| Repeats | ≥5 seeds, same folds for every model |

### 1.2 Pre-register the statistics

Lecture 8 is explicit that α is "fixed before looking."

- **H₀:** deep and classical models do not differ on the metric under test
- **H₁:** they differ — **two-sided**, unless you genuinely pre-commit to a direction
- **α = 0.05**, fixed now, written into the Method
- **Paired tests** across the same folds/datasets — paired because some datasets are hard for every model
- **Bonferroni** across the three metric families: test each at α/3 ≈ 0.0167. Lecture 8's own example shows FWER reaching 14.3% at three tests.
- Fallback when normality fails: Wilcoxon signed-rank or a permutation test

**Done when:** the protocol table is complete, the hypotheses and α are written into a draft Method, and all three have signed off.

---

## Phase 2 — Data access (start in parallel with Phase 1, not after)

This has the longest lead time and is the usual reason projects like this stall.

**Start immediately, same day as Phase 0:**

- **SEER** — needs a signed research request; allow days to weeks. Ali's review shows diagnoses spanning 1992–2017, which is what makes time-ordered splits possible.
- **Freddie Mac single-family loan-level** — free registration, fast download, ~600k loans available with origination vintages.

**Work on these while waiting** (no application needed, available today):

- `pycox` ships SUPPORT, METABRIC, GBSG, FLCHAIN
- `scikit-survival` ships several clinical datasets
- Lending Club (Kaggle) for a credit dataset with default and prepayment

Build and debug the whole pipeline on these. Swap in the gated data when it arrives.

**Done when:** applications submitted, at least one competing-risks dataset loading locally, and the pipeline running end-to-end on it.

---

## Phase 3 — Baselines and the known anchor

Implement classical models first. They are fast, they expose data bugs, and they give the comparison its floor.

Order: cause-specific Cox → Fine–Gray / Aalen–Johansen → RSF → DeepHit.

**Two anchors to check yourself against — use them:**

- **Blumenstock et al. (2022)** found DeepHit achieved the highest concordance in *every* experiment and significantly outperformed cause-specific Cox and Fine–Gray on Freddie Mac mortgages. If your DeepHit doesn't beat Cox on ranking, your implementation is wrong, not the literature.
- **Alberge et al. (2026)** found DeepHit and SurvTRACE failed both calibration tests in all five SEER runs, while Aalen–Johansen passed every time. That is the expected direction of your headline result.

**Plan for the compute wall now.** Blumenstock had ~600,000 loans and trained on 10,000 per fold because RSF "proved computationally demanding." Decide your subsample size up front and report it.

**Done when:** all four models train and predict on one dataset, and the ranking ordering roughly matches Blumenstock.

---

## Phase 4 — Run the grid

Every model × every dataset × every split type × ≥5 seeds. Log training time, inference time and hardware for every run — Filip's gap is that nobody reports this, so the project must.

Write results to one tidy CSV: `model, dataset, split_type, seed, metric, value`. Every number in the report and poster comes from that file.

**Done when:** the CSV is complete and no run is missing.

---

## Phase 5 — Statistics

Apply exactly what Phase 1.2 pre-registered. Do not change the test after seeing results.

- Paired test per metric family, across folds/datasets
- Bonferroni at α/3
- Report effect sizes and the per-fold differences, not only p-values
- State plainly where you **fail to reject** — a negative result is a result, and pretending otherwise is the failure mode Lecture 8 warns about

**Done when:** every headline claim has a test, a corrected p-value and an effect size.

---

## Phase 6 — The 5-page report

Reuse what worked in Assessment 1: a fixed template, every count traceable to a table, limitations stated plainly.

Suggested shape:

1. **Introduction** — the problem and the core-idea sentence. *Write this last* (Lecture 8).
2. **Related work** — one paragraph drawn from the three reviews, positioning against Shang et al. (2024), Wiegrebe et al. (2024) and Alberge et al. (2026).
3. **Protocol** — the Phase 1 table, plus hypotheses and α.
4. **Results** — one results table, one figure. Significance marked.
5. **Discussion and limitations** — including compute limits, subsampling and dataset coverage.

Check the page rule from forum question 1 before you write to length.

**Done when:** 5 pages, every number traceable to the CSV, no claim without a test.

---

## Phase 7 — Poster

Format unknown until the brief lands, so design content-first: one core question, the protocol table, one results figure, one takeaway line. Same discipline as the one-slide spotlight — large text, few words, counts not adjectives.

---

## Phase 8 — Pre-submission checks

- [ ] 5 pages, template as specified
- [ ] Every citation verified against the publisher record — a single wrong reference was an instant fail in Assessment 1; assume the same
- [ ] Code public (Lecture 8), anonymised if required
- [ ] Every number in the report and poster reconciles with the CSV
- [ ] α and hypotheses stated as pre-registered, correction applied
- [ ] Limitations section present
- [ ] If any joint screening or labelling was done, report Fleiss' kappa (three annotators — Lecture 8 covered this two weeks before a three-person project)

---

## The main risks

| Risk | Mitigation |
|---|---|
| SEER access is slow or refused | Build on pycox/scikit-survival data from day one; SEER is an upgrade, not a dependency |
| RSF/DeepHit too slow on full data | Subsample per fold as Blumenstock did; report the size |
| Two domains don't fit 5 pages | Make one domain primary, the second a stated extension |
| No results in time | Forum question 2 — if a design is acceptable, degrade to design + pilot rather than failing |
| Scope creep from three niches | The core-idea sentence is the test. If it doesn't serve that sentence, cut it. |

---

## What to do today

1. Agree the one gap and write the core-idea sentence (30 min, all three)
2. Post the four forum questions
3. Submit the SEER and Freddie Mac access requests
4. Create the shared repo and the results CSV schema

Steps 2 and 3 are the ones with external latency. Do them first.
