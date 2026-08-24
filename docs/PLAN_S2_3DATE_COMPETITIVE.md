# Plan — making the S2-only 3-date SVM arm presentable beside the RF paper and the XGBoost arm

Drafted 2026-08-25; revised the same day after an independent review (Codex) of the
first draft; amended again 2026-08-25 after a verification pass that re-derived every
number in §0 from the run artifacts and the RF paper's Table V. Vocabulary as in
`CONTEXT.md`. Builds on `PLAN.md` (executed), `docs/REPORT_2026-08-25.md`, and the
finished run `runs/s2_2018_3date_parcel/`.

The amendments were three: §0's comparison table now carries an explicit cap column
and states which of its cells may be subtracted from which (they mostly may not);
§0's weighted-F1 claim now quotes rubber's 85.1% support share, which is the actual
reason 0.8018 and 0.714 are not comparable; and §3.3-3.4 are re-aimed at the ad-hoc
analysis scripts, where a masking bug had manufactured a result that does not exist.

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
4. **Parcel-disjointness costs nothing — and that is a tie, not a win.** The parcel run
   scores macro F1 0.2248; the leaky v2 scores 0.2245 (`end_to_end_report.csv`). That
   is +0.0003, and even that is loose, since the two score different populations
   (fold 2's 3.80M crop rows vs the whole tile's 15.16M). An earlier note claiming the
   parcel run *beat* the leaky one by ~0.016 was wrong — see §3.4. The defensible
   claim, and it is a good one, is that removing leakage cost no measurable accuracy;
   the RF arm cannot say the same.

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

1. **Fold-1 dual use.** The Platt sigmoids are fitted on fold 1; tuning anything else
   on fold 1 is optimistic model-selection reuse. Split fold 1 **by parcels** into a
   calibration half and a tuning half; refit the sigmoids on the calibration half
   only. Rare-crop tuning support becomes thin (Langsat has 13 val rows in total) —
   any operating point selected there is labelled exploratory.
2. **Save validation-fold probabilities.** `train_parcel_cascade.py` currently saves
   test-fold probabilities only, which leaves nowhere honest to tune.
3. **Add the missing assertion**: train–val parcel disjointness (train–test and
   val–test are already asserted at `train_parcel_cascade.py:227-228`).
   *Hardening, not a bug fix* — the property already holds in the current split
   (`np.intersect1d(parcels[tr], parcels[va]).size == 0`, verified). The assertion
   exists so that a future change to the split builder cannot break it silently,
   which matters more once §3.1 starts subdividing fold 1.
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
5. **Cross-fitted Stage-1 routes for Stage-2 selection.** Stage-2 training candidates
   are chosen by a Stage-1 model fitted on those same rows; before any Stage-2
   tuning, generate Stage-1 routes for training parcels out-of-fold (parcel-grouped).
6. **Test-fold discipline.** Fold 2 has already been read once (the 0.2248 report),
   so it is now an exploratory test set. Everything below selects on the tuning
   half of fold 1 or on grouped out-of-fold predictions; **fold 2 is scored once, at
   the end, on a predeclared configuration**, with a paired parcel bootstrap against
   the 0.2248 baseline on identical parcels.

## 4. The modelling sequence (each step selected off-test, one final test read)

### M1 — Calibration audit *(hours, no training)*
Reliability curves / Brier per stage on the calibration half, predicted vs observed
prevalence per class. This decides M2's framing: the run was calibrated on
natural-prior validation rows, so a mathematically correct correction toward those
priors should be a near no-op. If the calibrators are already good, exponent tuning
is **operating-point tuning**, not prior correction, and the plan says so.

### M2 — Operating-point (exponent) sweep on the parcel run *(minutes per cell)*
The (α₂, α₃) exponent sweep applied to the parcel run's probability arrays — with the
sink included in the derivation and the denominator taken from the calibration
population's priors, not the old balanced-regime constants in `sweep_prior_alpha.py`.
Select on the tuning half of fold 1; the hypothesis (not a promise) is that it
revives some of the seven dead crops at a macro-F1 cost/gain to be measured. Keep
hard routing (§5 of the report: joint scoring is worse).

### M3 — Feature parity: MTCI + raw B11 per date *(+6 columns, 24 → 30)*
The one index family both other studies have and this arm lacks (MTCI, red-edge) plus
the raw SWIR band their importance analyses rank highly. B05/B06/B11 already exist in
the composites. Build as a **new immutable NPZ with a feature manifest**, assert `y`,
parcel IDs and split assignment are byte-identical to the current matrix, name a new
run directory. Temporal deltas are **cut from the default bundle** — Dec−Oct /
Dec−Nov differences of existing columns are linear combinations carrying no new
information; they survive only as a preregistered training-only ablation if time
allows.

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
