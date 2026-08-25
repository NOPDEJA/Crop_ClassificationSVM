# Plan — making the S2-only 3-date SVM arm presentable beside the RF paper and the XGBoost arm

Drafted 2026-08-25; revised the same day after an independent review (Codex) of the
first draft; amended again 2026-08-25 after a verification pass that re-derived every
number in §0 from the run artifacts and the RF paper's Table V. Vocabulary as in
`CONTEXT.md`. Builds on `PLAN.md` (executed), `docs/REPORT_2026-08-25.md`, and the
finished run `runs/s2_2018_3date_parcel/`.

The first three amendments were: §0's comparison table now carries an explicit cap
column and states which of its cells may be subtracted from which (they mostly may
not); §0's weighted-F1 claim now quotes rubber's 85.1% support share, which is the
actual reason 0.8018 and 0.714 are not comparable; and §3.3-3.4 are re-aimed at the
ad-hoc analysis scripts, where a masking bug had manufactured a result that does not
exist.

**Amended a third time, 2026-08-25, after the review `docs/HANDOFF_PLAN_AMEND_REVIEW.md`
asked for.** That review's repro reproduced exactly (strict 0.2248 / 0.8018, masked
0.2405 / 0.8309, 278,123 dropped false positives = 7.72% of all crop predictions), so
the plan's own figures stand. Three things changed:

- **§0 consequence 4 was overclaiming and has been cut back.** "Parcel-disjointness
  costs nothing" is not supported; see below.
- **§3.4's blast radius was swept rather than assumed** — the review's biggest open
  item. Three of six analysis scorers carried the defect, not zero, and the
  consequence is now measured rather than feared.
- **A fourth amendment was added to M2**, naming the Stage-3 denominator hazard the
  review found and left out.

**Execution status.** §2.1's `CONTEXT.md` corrections and all five §3 protocol repairs
are implemented; §4's M1-M6 are not started, and M3 onward need a retrain. What each
repair changed is marked inline below.

## 0. The comparison targets, corrected

The first draft called the RF paper's 0.714 a macro F1. **That was wrong.** The
paper's Table IV "F1-score 0.7139" is the support-weighted average; computed from its
own Table V per-class figures, the RF model's **macro F1 is ≈ 0.625 over its 15
classes and ≈ 0.602 over the 13 crops**. The comparison the joint paper must survive:

| study | model | eval population | cap | crop-13 macro F1 | weighted F1 |
|---|---|---|---|---|---|
| RF paper | flat 15-class RF | pixel 80/10/10, rare crops supplemented from the 2020 epoch | ~390k/class | ≈ 0.602 (from Table V) | 0.714 |
| Collaborator | flat 14-class XGBoost | pixel 60/20/20, scored on the CV partition | 200k/class | (their repo) | — |
| This arm | 3-stage SVM cascade | **parcel-disjoint** test fold | none (natural priors) | 0.2248 | 0.8018† |
| This arm, rescored under a 200k cap | *leaky v2* model | pixel-split-trained, capped | 200k/class | 0.3965\* | 0.5520\* |

*\*from `rescore_collaborator_protocol.py` applied to the **v2 (pixel-split, leaky)**
predictions — not the parcel run. It shows the shape of the population effect only, and
must be recomputed on the parcel run before it appears in any joint table.*

**The cap column is not decoration.** The three cap regimes (~390k / 200k / none) are
different populations, so the macro column is *not* a ranking and no two of its cells
are directly subtractable. Only cells sharing a cap may be compared. The one
comparison this plan will actually earn is parcel-run-at-200k vs collaborator-at-200k,
and it does not exist yet.

Four consequences:

1. Under matched 200k caps the gap looks like roughly 0.40 vs 0.60 — large, but half
   the size the mislabeled 0.714 implied, and the RF side still carries pixel-split
   leakage and 2020 supplementation that this arm removed. **Caveat that presently
   voids the number:** the 0.40 is the *leaky* v2 model's capped rescore, and 0.60 is
   RF at a ~390k cap, not 200k. Neither half of "0.40 vs 0.60" is yet an honest,
   like-for-like figure; M5 produces the first one that is. Do not put this sentence in
   an email or a paper until then.
2. This arm's weighted F1 (0.8018, parcel-disjoint) exceeds the RF headline (0.714,
   capped) — but see †. This is the weakest claim in the plan, not the strongest.
3. **No table may put crop-13, flat-14 and flat-15 macro figures in one column.** The
   agreed joint taxonomy (`CONTEXT.md`: 13 crops + reservoir + others) is the common
   endpoint; each study's native summary appears separately.
4. **We have no measurement of what leakage cost, and must not imply one.**
   *(Amended. This read "parcel-disjointness costs nothing — a tie, not a win", leaning
   on the parcel run's 0.2248 against the leaky v2's 0.2245.)* Those two numbers are not
   a comparison. They score different populations — fold 2's 3.80 M crop rows against
   the whole tile's 15.16 M — and the two runs differ in five configuration changes at
   once: split, balancing, sink taxonomy, Stage-3 population and calibration. A gap of
   +0.0003 between non-comparable quantities is not evidence of a tie; it is the absence
   of evidence in either direction, and calling it a tie smuggles in a result. An
   earlier note claiming the parcel run *beat* the leaky one by ~0.016 was worse still,
   and was a scoring artifact — see §3.4.

   What may honestly be said, and it is still worth saying: this arm's headline is
   measured on ground the model never saw, and the RF arm's is not. That is a claim
   about protocol, not about accuracy. A controlled estimate of leakage cost would need
   the same configuration trained twice differing only in the split — the experiment
   `PLAN.md` cut as run (a). It remains uncosted, and until it is run the honest answer
   to "what did leakage buy the old numbers?" is that we do not know.

†**Rubber is 85.1% of this arm's crop-13 test support** (3,235,887 of 3,800,567 rows),
so the 0.8018 weighted F1 is close to a restatement of rubber's own F1 (0.8767). The RF
population is rebalanced to ~390k per class, where rubber is ~15%. The two weighted
numbers therefore measure near-different things, and the concentration share — not a
vague "populations differ" — is what must be quoted beside 0.8018 every single time.

## 1. Constraints (unchanged)

Three-stage hierarchy · SVM learner · Sentinel-2 only · 3-date (Oct/Nov/Dec) ·
pixel-level classification · one 32 GB machine, serial runs. The 5-date arm is out of
scope for this plan.

## 2. Step 0 — protocol agreement first (was Track B, moved to the front)

The taxonomy, epoch and shared population determine what every later step optimises,
and the email does not depend on any run finishing.

1. **Verify the RF paper's numbers against the published PDF** (done above; record in
   the joint-paper notes) and fix the stale `CONTEXT.md` description of the
   collaborator's XGBoost as a cascade — it is a flat 14-class model. This is **three
   lines, not one**: `CONTEXT.md:39` ("The SVM cascade uses none; the collaborator's
   XGBoost cascade does") and `CONTEXT.md:52-53` ("Both this project's SVM and the
   collaborator's XGBoost are cascades, but they are *different* cascades and their
   internal stages do not correspond"). Line 39 carries a second claim in the same
   breath — that this arm uses no temporal information while theirs does — which must
   be re-checked on its own merits rather than deleted along with the cascade error.
2. **Settle the epoch question with the collaborator before proposing pixel IDs.**
   Their model is 2024-trained (2020-supplemented); this arm is 2018. A shared pixel
   list is only meaningful once both sides score the same epoch, or each epoch gets
   its own shared list.
3. **Propose the two-column joint table**: for each model, (a) the shared capped
   population and (b) a parcel-disjoint population — the latter only for models
   *retrained* under the parcel split. A saved pixel-trained prediction array cannot
   be made parcel-disjoint by rescoring; if the collaborator does not retrain, their
   capped score is labelled a prevalence-normalised descriptive rescore, not an
   honesty audit. Ask for the retrain; do not assume it.

## 3. Protocol repairs before any new number (cheap, mostly code)

1. **Fold-1 dual use.** *(IMPLEMENTED.)* The Platt sigmoids are fitted on fold 1;
   tuning anything else on fold 1 is optimistic model-selection reuse. Fold 1 is now
   halved **by parcel** into a calibration half and a tuning half by
   `halve_by_parcel()`, and every stage calibrates on the calibration half only. The
   halving is **stratified by each parcel's label**, which the plan did not ask for but
   which the rare-crop supports force: an unstratified coin flip over 5,818 validation
   parcels can hand a whole class to one half, leaving the calibrator with no positives
   to fit a sigmoid against. Halves are written as `val_cal_idx.npy` / `val_tune_idx.npy`.
   Rare-crop tuning support is still thin (Langsat has 13 val rows in total, so its
   tuning half holds a handful) — any operating point selected there is exploratory.
2. **Save validation-fold probabilities.** *(IMPLEMENTED.)* Every stage now writes
   `*_prob_val.npy` alongside `*_prob_test.npy`, with the matching index arrays. This
   costs a second prediction pass over the validation candidates, which is the price of
   having anywhere to tune that is not fold 2.
3. **Add the missing assertion**: train–val parcel disjointness. *(IMPLEMENTED.)*
   *Hardening, not a bug fix* — the property already holds in the current split
   (`np.intersect1d(parcels[tr], parcels[va]).size == 0`, verified, and re-verified by
   the smoke run). The assertion exists so that a future change to the split builder
   cannot break it silently, which matters more now that §3.1 subdivides fold 1; a
   `cal`/`tune` disjointness assertion was added beside it for the same reason.
4. **Fix the ad-hoc scorers — this is where the real bug was.**
   `train_parcel_cascade.py` is already safe: it passes `labels=CROPS`, a fixed list,
   so no class can drop out of its macro average. The danger is the one-off analysis
   scripts, and it has already bitten once. `parcel_agg.py` masked to
   `(asg == 2) & isin(y, CODES)` **before** computing precision, so the 278,123
   non-crop test rows that the cascade predicted as a crop — 7.7% of all its crop
   predictions — never entered any false-positive count. That produced a macro F1 of
   0.2405 where the run's own `report_hard.csv` says 0.2248, and the inflated figure
   was then compared against a leaky-v2 number computed the stricter way, manufacturing
   a ~0.016 "gain" that does not exist (§0, consequence 4). Therefore every scorer,
   ad-hoc ones included, must fix **both**:
   - the **class list** (`labels=CROPS`), so a rare class cannot vanish; and
   - the **population convention** — score the full test fold with non-crop truth
     mapped to 0, exactly as `train_parcel_cascade.py` and `evaluate_end_to_end.py`
     do. Never mask to crop-truth rows before computing precision.

   Before any cross-run comparison, confirm both sides used this convention. Two
   numbers computed under different conventions are not comparable no matter how
   similar they look.

   **Swept and fixed, 2026-08-25.** The blast radius was checked rather than assumed.
   Three of the six analysis scorers carried the defect; three did not.

   | script | verdict |
   |---|---|
   | `sweep_prior_alpha.py:235` | **had it** — and on `econ13_macro_f1`, the column that *ranked* the 256-cell sweep |
   | `apply_prior_correction.py:159` | **had it** — on the per-crop econ-13 table |
   | `validate_alpha_split.py:61` | **had it** — inside `econ_scores()`, so both the A (selection) and B (reporting) halves |
   | `audit_parcel_disjoint.py` | clean — `np.where(isin(y, CODES), y, 0)` with `labels=CODES` |
   | `rescore_collaborator_protocol.py` | clean — maps both truth and prediction into `others` |
   | `diagnose_error_budget.py` | clean — computes no precision at all; its true-econ mask is the correct unit for tracing where recall goes |

   All three are now fixed to the strict convention. The consequence is **measured, not
   feared**: the v2 sweep happens to carry both a masked column (`econ13_macro_f1`) and a
   strict one (`crop_macro_f1_flat`) for all 256 cells, so the two can be compared
   directly. Masking ran a **stable +0.034 high** (mean 0.0344, range 0.0249–0.0389) and
   moved the argmax by **one adjacent cell**, (0.80, 0.70) → (0.80, 0.80).

   So the sweep's *ranking* survives the bug and its *levels* do not. Restated honestly,
   prior correction's gain over the uncorrected baseline is **+0.0015** (strict:
   0.2036 → 0.2051), not the **+0.0054** the masked column showed (0.2374 → 0.2428) —
   both of which are small enough that §4's M2 should be framed as a diagnostic, not as
   a source of headline gain. Note the shape of the original error: two of the three
   scripts already computed a clean metric next to the masked one. The masked one was
   simply the one being read.

   **Every `econ13_*` column in a `sweep_results.csv` written before this fix is
   inflated, and must not be placed beside one written after it.**
5. **Cross-fitted Stage-1 routes for Stage-2 selection.** Stage-2 training candidates
   are chosen by a Stage-1 model fitted on those same rows; before any Stage-2
   tuning, generate Stage-1 routes for training parcels out-of-fold (parcel-grouped).
6. **Test-fold discipline.** Fold 2 has already been read once (the 0.2248 report),
   so it is now an exploratory test set. Everything below selects on the tuning
   half of fold 1 or on grouped out-of-fold predictions; **fold 2 is scored once, at
   the end, on a predeclared configuration**, with a paired parcel bootstrap against
   the 0.2248 baseline on identical parcels.

## 4. The modelling sequence (each step selected off-test, one final test read)

### M0 — The prerequisite run *(~7 h; NOT in the original sequence)*

**Added 2026-08-25. The sequence below had an unstated dependency.** M1 audits
calibration "on the calibration half" and M2 selects "on the tuning half of fold 1" —
but neither half exists in any run yet, and `runs/s2_2018_3date_parcel/` saves no
validation probabilities at all. §3.1 and §3.2 are edits to
`train_parcel_cascade.py`; they take effect only when that script is next run. So M1
and M2 cannot start on existing artifacts, and the plan as written implied they could.

There is no honest shortcut. Dumping validation probabilities from the *saved* models
would produce arrays whose sigmoids were fitted on a random 300,000-row subsample of
the whole of fold 1 — so the "tuning half" would be contaminated by calibration,
which is precisely the reuse §3.1 exists to remove. Scoring fold 2 instead is barred
by §3.6.

M0 is therefore the current configuration plus the five §3 repairs, and nothing else:
no new features, no new hyperparameters. It earns its cost twice over, because it is
also **the baseline M5's paired parcel bootstrap must compare against**. The existing
0.2248 is not that baseline — it was produced without the repairs, so a later
comparison against it would confound the repairs with M3's features and M4's capacity.

*Verify:* `val_cal_idx.npy`, `val_tune_idx.npy` and a `*_prob_val.npy` per stage exist;
the cal/tune parcel-disjointness assertion passed; the logged out-of-fold vs in-sample
Stage-1 route agreement is recorded in the manifest (a low figure is not a failure —
it is the size of the problem §3.5 fixes, and should be reported).

*Cost note:* §3.5's cross-fitting adds `CROSSFIT_S1` extra Stage-1 base fits (default 3)
and §3.2 adds a prediction pass over the validation candidates, taking the chain from
roughly 4.7 h to roughly 7 h. `CROSSFIT_S1=0` restores the old in-sample routes and the
old runtime, and logs a warning — acceptable only for a smoke test, never for M0 or M5.

#### M0 RESULT — run 2026-08-25 02:31–08:09, `runs/s2_2018_3date_parcel_m0/`

Every M0 criterion passed. Actual cost 5 h 38 min, not the 7 h estimated: cross-fitting
was dearer than expected (each part fits on ~2.85 M rows against the full fit's 2.88 M,
because the per-superclass caps bind on two thirds of fold 0 almost as hard as on all of
it, so cross-fitting roughly *doubles* Stage 1 rather than adding a third), and the
prediction passes were correspondingly cheaper.

**1. The §3.5 defect, measured properly.** Out-of-fold and in-sample Stage-1 routes agree
on **94.27%** of fold 0. The agreement scalar alone does not say what that costs, so the
run now also reports both candidate populations from the same model:

| fold-0 routing | Stage-2 candidates | sink share |
|---|---|---|
| in-sample (what the baseline did) | 9,806,157 | 15.657% |
| out-of-fold (repaired) | 9,944,240 | **17.030%** |

The in-sample routes were hiding **10.3% of the contamination** Stage 2 exists to reject.
Same rows, same Stage-1 model, only the routing differs — so nothing else varies. Note
that comparing M0's sink share against the *baseline run's* would NOT have been valid,
because M0 also changed the calibration set; that comparison was drafted and discarded.

**2. Accuracy: the repairs cost nothing measurable.** Both runs scored the identical fold
2 (5,500,269 rows, 5,824 parcels), so unlike §0's 0.2248-against-0.2245 this is a
legitimate paired comparison. Paired parcel bootstrap, 400 replicates:

| | macro F1 | 95% CI |
|---|---|---|
| baseline `s2_2018_3date_parcel` | 0.2248 | — |
| M0 `s2_2018_3date_parcel_m0` | **0.2283** | [0.2119, 0.2404] |
| paired delta | **+0.0032** | [+0.0001, +0.0070] |

**Do not report this as a gain.** The interval's lower bound is +0.0001, indistinguishable
from zero, and a tail percentile from 400 replicates is itself noisy. The supportable
claim is that removing the leak from model selection did not cost accuracy — which is what
one wants from a correctness fix, and is the *second* time this project has found that
tightening the protocol is close to free.

The wide CI is the more important number: **±0.014 around 0.2283 at the parcel level.**
Every macro F1 this project has published without such an interval is less precise than
its four decimal places imply, and M5's improvement must clear this band to mean anything.

Per crop, no class regressed materially and none revived: Oil palm +0.0277 and Mango
+0.0190 are the largest moves, Durian −0.0113 the only notable loss, and the same **7 of
13** crops sit above F1 0.01 in both runs. Joint routing improved (0.1949 → 0.2025) but
remains clearly worse than hard routing, reproducing the report's §5 finding on a second
independent run.

**3. The rare-crop risk did not materialise.** Every one of the 13 crops appears in both
halves of fold 1. Langsat, the binding case, got **4 calibration pixels and 9 tuning
pixels** — above the 2-positive floor, so a real sigmoid was fitted and no fallback fired
at full scale. Label-stratified halving is what made that hold; the smoke run, where
stratification had too few parcels to work with, did drop Langsat and exercised the
fallback path instead.

**4. Amendment 4 independently confirmed.** All three Stage-3 fit sizes reproduced the v2
run exactly — orchards 172,371, plantation 148,344, field 210,000 = 3 × `PER_LU_CAP`. So
field's denominator is uniform and the other two are not, and reusing a v2 `ratio3_den`
artifact would misdenominate two of the three experts. This is now verified on two runs,
not inferred from one.

*Not carried by this run:* M0 does not persist the out-of-fold routes, so the in-sample
side of table 1 had to be recomputed afterwards from `stage1_pred.npy`. The script now
saves `stage1_route_oof_train.npy` and logs both sink shares directly, so M5 will not need
the reconstruction.

### M1 — Calibration audit *(hours, no training)*
Reliability curves / Brier per stage on the calibration half, predicted vs observed
prevalence per class. This decides M2's framing: the run was calibrated on
natural-prior validation rows, so a mathematically correct correction toward those
priors should be a near no-op. If the calibrators are already good, exponent tuning
is **operating-point tuning**, not prior correction, and the plan says so.

#### M1 RESULT — run 2026-08-25, `m1_calibration_audit.py`, no training

Artifacts: `runs/s2_2018_3date_parcel_m0/m1_calibration.csv`, `m1_reliability.csv`,
`m1_prior_stability.csv`.

*Departure from the plan, deliberate.* M1 as written audits "on the calibration half" —
the rows the sigmoids were fitted on, where scores are optimistic by construction. Both
halves are reported; the tuning half is the honest number and the gap is the overfit.

**Stages 1 and 2 are well calibrated.** Predicted-over-observed prevalence ratios sit at
0.86–1.29 (Stage 1) and 0.91–1.26 (Stage 2) with ECE 0.002–0.052, on *both* halves. There
is no systematic distortion for a prior correction to repair at the routing stages.

**Stage 3 looks miscalibrated out-of-sample, but the cause is not the calibrator.** On the
tuning half the experts over-predict rare crops badly — Longan 3.58×, Mango 2.43×, Langsat
2.08×, Jackfruit 1.80× — while under-predicting Oil palm at 0.39×. The obvious reading is
that Platt scaling overfitted the calibration half. The prior-stability table says
otherwise.

**The finding that matters: there is no stable `π_true` to correct toward.** The folds are
disjoint *parcel* samples of one tile, and class prevalence swings between them. The table
is split by support, because the ratio is only meaningful where the counts are:

*Swings on ample support — these carry the finding:*

| class | within | cal | tune | test | max/min | smallest count |
|---|---|---|---|---|---|---|
| Oil palm | plantation | 0.03731 | 0.13014 | 0.02685 | **4.9×** | 42,868 px |
| Mango | orchards | 0.16305 | 0.04404 | 0.12820 | **3.7×** | 3,045 px |
| Rice | field | 0.09772 | 0.27900 | 0.13533 | **2.9×** | 21,239 px |
| Jackfruit | orchards | 0.05920 | 0.03634 | 0.09658 | 2.7× | 2,513 px |
| Pineapple | field | 0.49509 | 0.30686 | 0.38009 | 1.6× | 49,945 px |
| Rubber | plantation | 0.96099 | 0.86904 | 0.97227 | 1.1× | 732,519 px |

Even the four superclasses move by 1.15–2.10× on millions of pixels each.

*Ratios too noisy to cite — recorded so they are not mistaken for evidence:*
Langsat 25.6× rests on **4 / 9 / 191 pixels** across the three samples and on roughly one
parcel per fold; Longan (975 px minimum), Rambutan (375) and Coconut (695) are similarly
thin. **An earlier version of this section led with the Langsat 25.6× figure. That was
wrong** — it is small-count noise dressed as a result, and it is exactly the kind of
number this document exists to catch. The finding does not need it: Oil palm swinging
4.9× on 42,868–109,693 pixels establishes it on its own.

So Stage 3's calibrators are not broken: they faithfully learned the calibration half's
mix, and the tuning half is simply a different mix. The "miscalibration" is the fold
difference, not a modelling fault.

**Three consequences, and they are load-bearing.**

1. **M2 is operating-point tuning. It is not prior correction, and the report must not
   call it that.** Prior correction presumes `π_true` is a property of the world that the
   training sample distorted. Here `π_true` is a property of *which parcels you drew*.
   There is nothing to correct toward.
2. ~~**An exponent selected on the tuning half may not transfer to fold 2 at all.**~~
   **Overstated — M2 tested this and it is largely wrong.** The reasoning was that the
   tuning half's Oil-palm share is 4.8× the test fold's and its Langsat share a
   fourteenth, so a rare-class rule tuned there fits the sample. M2 measured it across
   two disjoint parcel samples and found the exponent surface **correlates at 0.9936**
   between them, picking adjacent cells at a transfer cost of −0.0014. Prevalence
   instability is real, but it does not destabilise the *choice of exponent*, because the
   response surface is a broad plateau rather than a peak. The surviving caution is
   narrower and still worth stating: per-crop claims for the rarest classes remain
   unsupported, because the tuning half contains **9 Langsat pixels**.
3. **It compounds the ±0.014 interval from M0.** Not only is a single macro F1 imprecise,
   the class mix it is computed over is itself one draw. Any per-crop claim for the seven
   rare crops rests on a prevalence that moves by 2–25× depending on which parcels landed
   in the fold.

This also retires a hypothesis the project has carried for a while: the
`prior-correction-sweep-regime` work treated the flat response surface as a property of
the correction. Part of it is more likely this — the target being corrected toward was
never stable.

### M2 — Operating-point (exponent) sweep on the parcel run *(minutes per cell)*
The (α₂, α₃) exponent sweep applied to the parcel run's probability arrays — with the
sink included in the derivation and the denominator taken from the calibration
population's priors, not the old balanced-regime constants in `sweep_prior_alpha.py`.

**Amendment 4 — be precise about *which* denominator is stale, because only one is.**
Verified against `runs/s2_2018_3date_parcel/run.log` and `config.py:50-63`:

- *Stage 2 is safe.* `sweep_prior_alpha.py:124` asserts a uniform `pi_train` because
  Stage 2 saw an equal count per subclass. It still does: the parcel run's Stage-2 base
  fit is exactly **800,000 rows = 4 × `PER_GROUP_CAP`** (200,000), so the cap bound
  uniformly in both runs and the *ratio* is unaffected. Only the comment's number is
  stale, not the arithmetic.
- *Stage 3 is the real hazard.* The script reads per-class training counts from a saved
  artifact (`ratio3_den`, lines 121-122) rather than assuming uniformity — correct
  behaviour in itself, but those counts describe the **v2** run. In the parcel run the
  Stage-3 caps did **not** all bind: field fit on 210,000 = 3 × `PER_LU_CAP` (uniform),
  but orchards fit on **172,371** and plantation on **148,344**, both under their
  ceilings and therefore non-uniform. Reusing v2's counts file would apply the wrong
  denominator to two of the three experts, and would do it silently.

M2 must therefore recompute Stage-3 denominators from the parcel run's own fitted
counts, and must fail loudly rather than fall back to an artifact from another run.
Select on the tuning half of fold 1; the hypothesis (not a promise) is that it
revives some of the seven dead crops at a macro-F1 cost/gain to be measured. Keep
hard routing (§5 of the report: joint scoring is worse).

#### M2 RESULT — run 2026-08-25, `m2_operating_point.py`, no training

Artifact: `runs/s2_2018_3date_parcel_m0/m2_sweep.csv`, 169 cells (α₂, α₃ ∈ 0.0…1.2).

**A correction to M2 as specified.** The plan says to take the denominator from the
training population's priors, as `sweep_prior_alpha.py` does. That is right for v2 —
weighted, upsampled, calibrated by internal CV on the rebalanced sample, so its
probabilities sat at a rebalanced prior. It is **wrong for M0**, which uses caps only and
fits its sigmoids on natural-prior validation rows. M1 measured the consequence:
prevalence ratios 0.86–1.29 at Stage 1 and 0.91–1.26 at Stage 2. The probabilities are
*already* at natural priors, so dividing by the capped training prior would re-apply a
correction calibration has already made, and would show a large effect that is an
artifact of double-counting. The sweep therefore reweights from the calibration
population's prior toward uniform, and α is an **operating point** — "how far toward
balanced" — not a correction toward a truth.

**Amendment 4 discharged.** Stage-3 denominators are re-derived from this run as
`min(N_c in fold 0, PER_LU_CAP)` and the totals are asserted against the run log:
orchards 172,371, plantation 148,344, field 210,000 — all three reproduce exactly. The
script refuses to sweep on a mismatch rather than falling back to another run's artifact.

**Result, selected on the tuning half only (fold 2 not read):**

| | α₂ | α₃ | tune macro F1 | crops alive |
|---|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.2067 | 7 |
| **selected on tune** | **0.2** | **0.7** | **0.2179** | **10** |
| best on cal (check) | 0.2 | 0.6 | 0.2177 | — |

**+0.0112 macro F1 and three crops brought back from zero**, which is a better outcome
than the plan's hypothesis allowed for — it framed the trade as a possible cost.

Per crop on the tuning half: Mangosteen +0.0422, Durian +0.0384, Jackfruit +0.0275,
Rambutan +0.0252, Cassava +0.0171, Rubber +0.0077. Paid for by Pineapple −0.0144 and Oil
palm −0.0056. **Langsat stays at 0.0000 and cannot move: the tuning half holds 9 Langsat
pixels.** Pushing α₃ to 1.0 and beyond collapses the metric (0.1804 at 1.0, 0.1474 at 1.2),
so the useful range is bounded.

**The selection is robust, which was not expected.** Ten of 169 cells sit within 0.002 of
the optimum (α₂ 0.1–0.4, α₃ 0.4–0.7) — a plateau, not a peak — and the two disjoint parcel
halves rank all 169 cells with correlation **0.9936**. This both reproduces the older
sweep's "flat curve" finding on an honest split and retires the worry recorded in M1's
consequence 2.

**Predeclared for M5:** operating point **(α₂ = 0.2, α₃ = 0.7)**, hard routing, selected on
the tuning half. Recorded here *before* fold 2 is read, which is the point of writing it
down.

### M3 — Feature parity: MTCI + raw B11 per date *(+6 columns, 24 → 30)*
The one index family both other studies have and this arm lacks (MTCI, red-edge) plus
the raw SWIR band their importance analyses rank highly. B05/B06/B11 already exist in
the composites. Build as a **new immutable NPZ with a feature manifest**, assert `y`,
parcel IDs and split assignment are byte-identical to the current matrix, name a new
run directory. Temporal deltas are **cut from the default bundle** — Dec−Oct /
Dec−Nov differences of existing columns are linear combinations carrying no new
information; they survive only as a preregistered training-only ablation if time
allows.

#### M3 RESULT — 2026-08-25, `m3_build_features.py`, `m3_ablation_expert.py`

**The matrix is built and is row-identical.** `aligned_features/svm_s2_3date_m3_features_labels.npz`,
(24,323,769 × 30). `y` reproduces from the label raster exactly, so
`splits/split_assign.npy`, the parcel IDs and every M0/M1/M2 index array remain valid
against it. Rasters live in `./indices_m3`, deliberately **not** `./indices`, because
`align_indices_labels.py` derives column order from `sorted(glob("./indices/*.tif"))` and
six new files there would silently reorder every existing arm.

MTCI values are physically sensible — median 2.40–2.64, p95 4.50–4.90, the textbook range
for vegetated land. Coverage among labelled pixels is 94.59% (Oct), 99.71% (Nov), 99.05%
(Dec); the October composite carries the cloud gaps and those cells are median-imputed.

**The go/no-go probe says the new columns buy almost nothing.** Orchards expert alone,
fitted on the same 172,371 rows, scored on the tuning half by raw decision function:

| crop | 24 col | 30 col | delta | tune support |
|---|---|---|---|---|
| Durian | 0.8648 | 0.8712 | +0.0064 | 60,574 |
| Rambutan | 0.2201 | 0.2379 | +0.0179 | 615 |
| Mango | 0.4214 | 0.4528 | +0.0314 | 3,045 |
| Longan | 0.0449 | 0.0066 | **−0.0383** | 975 |
| Jackfruit | 0.0727 | 0.0733 | +0.0006 | 2,513 |
| Mangosteen | 0.1165 | 0.1079 | −0.0087 | 1,414 |
| Langsat | 0.0000 | 0.0123 | +0.0123 | **9** |
| **macro-7** | **0.2486** | **0.2517** | **+0.0031** | |

**On its own this reads as a negative result.** The +0.0031 is carried by classes with
615–3,045 pixels, where a few points is noise; Langsat's +0.0123 comes off **nine pixels**
and means nothing; and Longan *loses* more than any crop gains except Mango. The most
reliable line in the table is Durian, the only class with ample support, and it moves
+0.0064. **But orchards alone was the wrong basis for a verdict** — see the other two
experts below, which change the conclusion.

The probe script originally printed an automatic "PROCEED" whenever the mean improved.
That verdict was too credulous for a +0.0031 mean against a −0.0383 single-class move, and
has been replaced by a report of the mean move beside the largest single move and its
support. An automatic PROCEED on this evidence is precisely the overclaiming this document
exists to catch.

*Caveat that limits the probe:* `P23` uses `gamma=None`, which sklearn reads as
`1/n_features`, so 24 → 30 also moves gamma 0.0417 → 0.0333. Feature effect and kernel
width are therefore confounded. That mirrors what M5 would do, so it is the right probe
for the *decision*, but it means a win could not be attributed to MTCI. Separating them is
M4's job.

**All three experts, which is what the verdict must rest on:**

| expert | 24 col | 30 col | delta | reliability |
|---|---|---|---|---|
| orchards | 0.2486 | 0.2517 | +0.0031 | poor — 5 of 7 classes under 3,045 px, signs mixed |
| field | 0.7400 | 0.7412 | +0.0011 | good — all three 45k–67k px; signs mixed |
| plantation | 0.4545 | 0.4591 | **+0.0046** | **best — 695 to 732,519 px, all three positive** |

Field is the sharpest test and returns essentially zero (+0.0011 with Pineapple −0.0052
against Rice +0.0051). Plantation is the only expert where every class moves the same way,
and it carries the largest supports in the study: Rubber +0.0038 on 732,519 px, Oil palm
+0.0088 on 109,693 px.

**The gamma confound is resolved, not merely flagged.** `gamma=None` means `1/n_features`,
so 24 → 30 also moved gamma 0.041667 → 0.033333. Refitting plantation at 30 columns with
gamma **pinned** to the 24-column value separates them:

| configuration | macro |
|---|---|
| 24 col, gamma = 1/24 | 0.4545 |
| 30 col, gamma = 1/24 *(pinned)* | **0.4581** |
| 30 col, gamma = 1/30 *(as run)* | 0.4591 |

So of the +0.0046, **+0.0036 is the features and +0.0010 the gamma side-effect**. MTCI and
B11 do carry signal. (The as-run configuration reproduced to four decimals and to identical
per-class values on a second fit, which also confirms the pipeline is deterministic under
the fixed seed.)

**Verdict: proceed to M5 at 30 columns**, on three grounds — the effect is small but
consistently positive across all three experts, it is mostly attributable to the features
rather than to gamma, and feature parity is what makes the joint comparison like-for-like.

**But state the size honestly.** Per-expert gains of +0.001 to +0.005 macro will not move
the end-to-end headline meaningfully, and they are an order of magnitude smaller than M0's
±0.014 parcel-level confidence interval. M3 was hoped to rescue the rare species; it did
the opposite, helping most where support was already largest. The seven near-zero crops
remain near zero, and nothing in M3 addresses them.

### M4 — Small, stage-specific capacity check *(bounded, after M3)*
Any gamma=None setting means 1/n_features, so tuning before the feature change would
be invalid — features first, then search. Not the 24-cell grid of the first draft:
benchmark **one** maximum-size fit for memory first, then at most two capacity points
above the current 600 per stage, parcel-grouped CV, `scoring='f1_macro'` as the
surrogate, checked with grouped out-of-fold end-to-end predictions before selection.
If the ceiling is selected again, report the bound rather than escalating.

### M5 — Retrain + the single fold-2 evaluation
Predeclare: features (M3), parameters (M4), calibration split (§3.1), operating
point (M2). Train, score fold 2 once, paired parcel bootstrap vs baseline. Rescore
the same predictions under the shared capped protocol for the joint table.

### M6 — Conditional pilot: contamination-aware Stage-3 experts
Only if the grouped out-of-fold error budget shows cross-group contamination is a
dominant loss: add a per-expert rejection class trained from cross-fitted upstream
misroutes. Stays within the three-stage SVM hierarchy; deferred by default.

## 5. Dropped or deferred

- **R4 (2020 rare-crop supplementation) is out of the execution plan.** Adding only
  rare crops from another epoch makes epoch correlated with class — the model could
  learn the year as a rare-crop cue; and the RF paper's own cross-year check showed
  its synthetic-oversampling variants failed to generalise. If the Prof wants
  protocol parity it must be a separately named multi-epoch arm with matched 2020
  imagery for those parcels, decided at a meeting, never slipped into a run.
- Temporal deltas (ablation only, see M3).
- The 24-cell hyperparameter grid.
- Numeric success promises (">0.2248", "~0.5", "all 13 alive"). The deliverables are
  the predeclared protocol, per-class results with parcel-level uncertainty, and the
  population-effect decomposition — the joint paper's actual contribution.

## 6. Risks

1. Thin tuning support for rare crops after splitting fold 1 — label those operating
   points exploratory; quote parcel-bootstrap CIs.
2. Memory: a 1M × 1500 float64 Nystroem block is ~12 GB; the benchmark fit in M4
   exists to catch this before an overnight run does.
3. The collaborator may decline retraining or a shared population — fallback is the
   two-protocol table with the measured population effect as justification.
4. One fold-2 read at the end means no second chance; that is the point, and the
   grouped out-of-fold machinery in §3 is what replaces iterating on test.
