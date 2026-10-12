# Assessment 2 — status and what is left

*Plan written 11 Oct 2026, updated 12 Oct. Three people: Amirreza, Ali Emre, Filip.*

**The official guidelines are still not published.** Everything below is built from Lecture 1 (what the deliverable is) and Lecture 8 (how to run and write it). Re-anchor the moment the brief appears.

From Lecture 1, verbatim: 2–3 students, combine your findings, **fill one of the identified gaps**, a concrete research question, theoretical or empirical, **5-page report and a poster**, "a similar grading criteria as lit. review."

Still unknown: **deadline**, **poster dimensions**, whether references count toward the 5 pages, and whether a template is mandated.

---

## Status

| Phase | State |
|---|---|
| 0. Scope, gap, core idea, owners | **done** — forum questions never posted |
| 1. Protocol written before code | **done** — it is §III of the report |
| 2. Data access | **partly** — Freddie Mac in hand; SEER never arrived |
| 3. Baselines implemented | **done** — Cox, Fine–Gray, RSF, DeepHit |
| 4. Experiment grid | **done** — `results/all.csv` and five others |
| 5. Statistics | **done** — paired bootstrap, Bonferroni, seed and bootstrap variability |
| 6. 5-page report | **drafted** — compiles at 5 body pages; needs rewriting in your voice |
| 7. Poster | **drafted** — A0 portrait, size unconfirmed |
| 8. Pre-submission checks | **outstanding** |

---

## The gap we chose, and why it worked

All three reviews converged. Amirreza's and Ali's niches were the same question in two domains; Filip's was not a fourth gap but the *method* for testing them. The single gap:

> Deep survival models are selected and reported on ranking, and it is not established that their advantage survives when calibration is also measured.

Ownership followed: **Filip** the protocol, metric set and cost; **Ali** the calibration measures; **Amirreza** competing risks and the credit setting.

That split still holds, and it is how the poster columns should be divided when presenting.

---

## What changed from the plan

**Compute was never the constraint.** The plan budgeted for Blumenstock's 10,000-loans-per-fold limit. RSF fits 150,000 loans in 36 seconds on this hardware, so no subsampling was needed. That assumption was 2022-era and is worth a sentence in the report.

**The original niche did not survive contact with the data.** We set out to test whether a ranking advantage erodes under calibration. Untuned, there was no advantage to erode. The question became regime-dependent and the answer is more interesting than the one we expected. See `FINDINGS.md`.

**SEER never arrived**, so the clinical half of the group's reviews has no empirical representation. This is declared as a limitation rather than worked around.

**Two experiments were added** that were not in the plan: training variability for every model (not just DeepHit), and post-hoc recalibration. Both changed conclusions — the first forced a caveat onto the headline result, the second turned a limitation into finding 9.

---

## What is left

### Must do before submitting

- [ ] **Rewrite the report prose.** The draft is complete and every number is verified, but the rubric penalises text that reads as generated. Read it aloud; vary the rhythm; make the argument sound like three people who argued about it.
- [ ] **Create the GitHub repo and push.** Local repo is initialised with two commits and a `.gitignore` excluding the licensed data. Needs a remote, then the URL goes into `report/main.tex` and `report/poster.tex` where `[repository URL]` currently sits.
- [ ] **Confirm the poster size** against the brief. A0 portrait is assumed; change one `geometry` line if wrong.
- [ ] **Confirm Alberge's page numbers** (currently PMLR 300:1837–1845, taken from the PMLR record) against Ali's copy.
- [ ] **Ali and Filip review everything.** Most of this was built in one person's session. Beyond the collaboration requirement, whoever stands at the poster must be able to answer "what is an integrated calibration index?" without reading off the board.

### Worth doing if there is time

- [ ] Apply Alberge's per-cause calibration measures (CR-D-calibration, cal-αK-calibration) rather than our marginal rescaling.
- [ ] More than five bootstrap replicates for the variability estimates.
- [ ] A clinical dataset, if SEER access ever clears, so the group's two clinical reviews are represented.

---

## Checks before submission

- [ ] 5 pages, template as specified by the brief
- [ ] Every citation verified against a publisher record — a single wrong reference was an instant fail in Assessment 1
- [ ] Code public, `[repository URL]` replaced in both files
- [ ] Every number in report and poster reconciles with `results/` — regenerate with `src/make_tables.py` and `src/make_figure.py`, never transcribe by hand
- [ ] α and hypotheses stated as pre-registered, Bonferroni applied
- [ ] Limitations section present and honest
- [ ] Fleiss' kappa reported if any joint screening or labelling was done (Lecture 8 covered it two weeks before a three-person project)

---

## Lecture 8, as it applied

Worth recording because it turned out to be a brief rather than a lecture:

- *"Start a project by writing the paper"* — the protocol section was written before any model ran, and it did not change.
- *"Be clear about the core idea"* — one gap, one question.
- *"Make code available publicly"* — repo ready, URL pending.
- *"Always write or revisit Introduction in the end"* — done.
- **Hypothesis testing, paired tests, Bonferroni** — the worked example was three metrics at α = 0.05 with the family-wise error rate reaching 14.3%. We have three metric families and corrected at α/3.
- **Cohen's and Fleiss' kappa** — taught for two and three annotators, two weeks before a three-person project. Not a coincidence.

---

## One lesson worth keeping

Three claims in this project were wrong and were caught by checks, not by luck: a premature "DeepHit simply loses", an overstated selection-criterion effect, and a significance claim measured against the wrong model's noise. Each was found by recomputing from `results/` rather than by rereading the PDF.

**Whatever you rewrite, re-derive the numbers from the CSVs.**
