# Plan — Stage-2 subtype-mass run, report repair, and delivery

**Written:** 2026-08-26 (evening), by the planning session, after the Codex review of
`docs/REPORT_2026-08-27.md` and its correction round.
**Executor:** a fresh Claude Opus 5 session. Everything here is S2-only, 3-date arm.
**Reviewer:** Codex, delta re-review only (§8).
**Hard deadline:** report to the professor **Thursday 2026-08-27 before 23:00**.

The scope is three things, in priority order: (1) make the report's claims true and its
figures traceable, (2) run the Stage-2 subtype-mass experiment the report's own §8.1
identified (+0.149 oracle headroom, ~2 h not ~7 h), (3) rebuild the leaked probe with
parcel groups. All three feed the same report. Nothing else.

---

## 0. State you inherit

- Headline: M5 test macro F1 **0.2344** (`runs/s2_2018_3date_parcel_m5/`), tune-half
  0.2294 at cal-selected (0.3, 0.6) via `sweep_operating_point.py`.
- Weighted run lost its gate (0.2272), fold 2 never end-to-end scored for it, but
  **Stage 1 did mechanically predict fold 2** (`stage1_prob_test.npy`, 88 MB, in the
  weighted and tree dirs; `run.log:124` prints test composition). This is finding F1 below.
- `docs/REPORT_2026-08-27.md` (commit `1c8993c`, since edited) is written and good, but
  carries 8 open review findings (§2). It ships only after they are fixed.
- The discovery driving this plan (report §4): Stage 2 caps balance the four GROUP labels
  at 200,000 rows each but sample uniformly inside each group, so the plantation fit is
  96.82 % rubber / 3.10 % oil palm / 0.08 % coconut (~159 coconut rows). Replacing the
  learned Stage-2 route with the true group, all models frozen, moves tune-half macro F1
  0.2294 → 0.3785. Memory: `stage2-subtype-mass`.
- Withdrawn claim (report §5): `probe_dry_season.py` splits by pixel; 87.5–95.7 % of
  rare-crop test parcels are also in training. Langsat has 10 training parcels, not 27.
- Interpreter: `C:\Conda_environment\envs\svm_env\python.exe`. 32 GB, serial runs only.
  Launch long jobs detached/unbuffered per the `long-training-workflow` memory.
- Key mechanics of `train_parcel_cascade.py` you will reuse: Stage-3 experts already
  predict on **all** validation candidates (`stage3_*_prob_val.npy` are dense over
  `stage2_val_idx.npy` rows), so a new Stage-2 routing can be scored by composing cached
  Stage-1 and Stage-3 validation probabilities with new Stage-2 probabilities — **no
  Stage-1 or Stage-3 refit or re-prediction is needed.**

## 1. Decisions already made (do not reopen)

| # | decision |
|---|---|
| D1 | The subtype experiment is scored on **fold 1 only** (cal-select α, tune-report). **Fold 2 is not read for it before the meeting**, even if it wins. Rationale: the headline must not churn hours before the deadline, and the test read is worth more after the hyperparameter retune. The professor can overrule at the meeting. |
| D2 | The experiment is **two paired Stage-2 fits**: a control refit (identical config, fresh fit — measures pure retrain noise) and a treatment fit (subtype mass). Stage 1 and Stage 3 stay frozen from M5. No full cascade run. |
| D3 | Treatment mechanism: **per-sample weights inside Stage 2's fit**, w(row) = sqrt(m_maxc / m_c) where m_c is the post-cap row count of the row's crop subtype within its group and m_maxc the largest subtype count in that group; then normalise weights within each group so the group's total mass stays exactly its row count. Sink rows keep weight 1 (they are one subtype). Group totals therefore stay balanced at 200k mass each; only the distribution inside moves. Pass via `fit(..., svc__sample_weight=w)`. |
| D4 | Pre-launch check, non-negotiable (lesson of the weighted run): **print the per-subtype masses before and after weighting for every group at full scale and confirm the treatment differs from uniform** — coconut's mass share in plantation must be visibly larger than 0.08 %. If any group's masses come out unchanged, stop; do not launch a vacuous arm. |
| D5 | Report fixes F1–F8 (§2) are all applied. None is optional; F1–F3 change meaning, F4 requires saving artifacts. |
| D6 | Probe rebuild (§5) runs with parcel-grouped splits, same uniform prior and settings otherwise, and its result goes into the report **whatever it says** — including "we were wrong about the ceiling all year". |
| D7 | Out of scope: Δ-date features, red-edge bands, texture, cap changes, any full-cascade retrain, per-crop manual thresholds, joint table, anything touching `2018/`, and any DEM/S1 work. |

## 2. Report fixes (from the review of the corrected draft)

Apply to `docs/REPORT_2026-08-27.md`, keeping Nop's voice (`C:\Users\Nop\Downloads\SKILL.md`:
first person, and/so/but, no semicolons, no em-dashes, plain words, unflattering things
said plainly).

| # | severity | where | fix |
|---|---|---|---|
| F1 | Critical | §4 gate para, §1.3, limitation 1 | The sentences "could not read the test fold even if I wanted it to" / "I did not read the test fold" / "stayed unread" are false: SKIP_TEST prevented Stage-2/3 test prediction and end-to-end scoring, but Stage 1 still predicted fold 2 mechanically and the log printed test-fold composition. Say exactly that, and add one sentence owning that the predeclaration promised more than the script enforced. |
| F2 | High | §1.3 and §4 | "+0.149 … six times the entire gain from the baseline to M5" — 0.149 / 0.0096 ≈ **15×**, not 6×. Fix the multiplier or name a different denominator explicitly. |
| F3 | High | §6 | Drop "The truth sits between the two". The report's own table shows the two proxies order oppositely on the two macro columns (13-crop: 0.2673 < 0.2757, 14-label: 0.3078 > 0.3060), so they do not bracket. Call them "two proxy sensitivity scenarios on our buffered fold-2 population" and state that an exact collaborator-protocol value is unknown. |
| F4 | Medium | footer + §2/§4/§5 | Save an artifact for every figure that lacks one, then cite each: oracle routing (+0.149) → `runs/s2_2018_3date_parcel_m5/oracle_routing.csv`; Stage-2 routing accuracy per crop → `stage2_routing_accuracy.csv`; pool composition → `stage2_pool_composition.csv`; probe replay overlap table → `runs/probe_dry_season/probe_replay_overlap.csv`; the paired bootstrap → rerun once with a fixed seed, save `m0_bootstrap.csv`, and reconcile +0.0032 (M0 handoff) vs +0.0035 (report) — quote the saved artifact's number. |
| F5 | Medium | §2 | "the gain is entirely in the rare orchard crops" contradicts the table (Rice +0.0205, Cassava +0.0220). Use: most of the rare-class recovery is in orchards, and Rice and Cassava also improve. |
| F6 | Medium | §3 | "6,685 to 1" sits next to 3,235,887/191 (= 16,942:1). Name the population behind 6,685:1 and give the test-fold ratio separately. |
| F7 | Low | §3 table | "chosen on held-out rows" → "chosen on the calibration half (which also fitted the sigmoids), evaluated once on the tuning half". |
| F8 | Low | limitation 6 | Keep the 8-row count but add the measured fact: 77 validation candidate identities changed, so the −0.0022 includes retraining variation of unmeasured size. The §3-run control fit (this plan) measures it; cite the number once you have it. |

## 3. The Stage-2 subtype-mass experiment

### Build (~30 min)

One script, `s2mass_stage2.py`, flat style, outputs to `runs/s2_2018_3date_parcel_s2mass/`:

1. Load the M5 fold assignment, Stage-1 cached routes (`stage1_route_oof_train.npy` for
   fold-0 candidates; `stage1_prob_val.npy` for validation candidates), and the m3 matrix.
2. Rebuild Stage 2's training pool exactly as `train_parcel_cascade.py` does (same seed,
   same `PER_GROUP_CAP` draw) — byte-compare the drawn index set against a recomputation
   to prove identity before fitting anything.
3. Fit **control**: identical pipeline (`Nystroem(600) → LinearSVC(C=10)`), no weights,
   fresh RNG state. Fit **treatment**: same, plus D3's sample weights.
4. D4's pre-launch check happens between 2 and 3: print the mass table, abort if uniform.
5. Sigmoid-calibrate both on the calibration half, exactly as the cascade does.
6. Predict both on all validation candidates (chunked; ~20 min each).
7. Compose each with M5's cached Stage-1 and Stage-3 validation probabilities → hard
   routing → `sweep_operating_point.py` procedure: select (α₂, α₃) on the calibration
   half, score once on the tuning half, strict convention (full half, non-crop truth → 0,
   13-crop macro).

### Predeclaration (write to the run dir BEFORE fitting)

> Two Stage-2 fits, control and treatment, differing only in D3's sample weights.
> Stage 1 and Stage 3 frozen from M5. Metrics: tune-half strict crop_macro_f1 at the
> cal-selected cell, per-crop F1, and Stage-2 routing accuracy per crop.
> Reading rules, fixed in advance: the noise floor is |control − M5| = |control − 0.2294|.
> The effect is treatment − control. The effect is claimed only if it exceeds the noise
> floor. Fold 2 is not read regardless of outcome. Expected secondary movement: coconut,
> mango and Langsat routing accuracy (13.7 %, 11.0 %, 28.6 % at M5) should rise if the
> mechanism is real; rubber routing accuracy (85.4 %) may fall — report both directions.

### Verify gates

- Pool identity check passes (step 2) — otherwise the control is not a control.
- Mass table shows treatment ≠ uniform in all three named groups (D4).
- Control tune score within ~0.005 of 0.2294 — much larger means the pool rebuild is
  wrong, stop and diagnose rather than reporting anything.
- All scoring through the same code path for control, treatment, and (already done) M5.

### Report integration

Either outcome goes in the report as a short §4 continuation ("I then ran the experiment
§8.1 proposes"): the mass table, the three tune scores (M5 / control / treatment), routing
accuracy shifts, and the reading under the predeclared rules. If treatment wins, §8.1
becomes "done, next is the test read after retuning"; if it loses, say it plainly and say
what that implies about where the +0.149 headroom actually hides (calibration of Stage-2
probabilities is the next suspect, since oracle routing bypasses them entirely).

## 4. Probe rebuild (~30 min compute)

`probe_dry_season.py` rerun with a parcel-grouped split (GroupShuffleSplit on parcel ID,
same uniform prior, same 800 components, same seed policy), saving per-class F1 to
`runs/probe_dry_season/per_class_parcel_grouped.csv`. Predeclared reading, either way:

- If rare-crop F1 survives (say, > 0.3 for most of the seven): the withdrawn claim is
  partially recoverable in parcel-grouped form, and §8.1's case strengthens. Update §5's
  last paragraph to say exactly what the new number licenses.
- If it collapses: the report says the ceiling may be spectral after all, the year's
  framing was wrong, and item 1's headroom estimate (+0.149) still stands because it is
  measured on real routes, not on the probe.

Caution: with 10 Langsat parcels a grouped split leaves 2–3 test parcels; report
rare-crop rows with parcel counts attached and do not quote Langsat alone.

## 5. Schedule

| when | what |
|---|---|
| Wed evening | commit current state; F4 artifact scripts (they also produce §3's inputs); report fixes F1–F8 |
| Wed night | §3 build + pre-launch check + predeclaration + launch (fits are ~1–2 h; can run overnight unattended) |
| Thu morning | score §3; probe rebuild §4; integrate both into the report |
| Thu midday | Codex delta re-review (§6); apply findings |
| Thu afternoon | final read-through in voice; commit; **deliver before 23:00** with hours of margin |

If §3 or §4 fails or overruns: the report ships on the fixed F1–F8 text alone. They are
next-steps items again, and that is fine — the report was always "consolidate, bonus
either way".

## 6. Codex delta re-review brief

Paste to Codex Thursday midday, repo root as cwd:

> Delta re-review only. The full review's findings were applied; verify each of F1–F8 in
> docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md §2 is actually fixed in
> docs/REPORT_2026-08-27.md, then review ONLY the new content: the Stage-2 subtype
> experiment section and the probe-rebuild paragraph. For the experiment: re-derive the
> three tune-half scores (M5, control, treatment) from the saved probability arrays with
> the strict scorer, confirm the noise-floor reading rule in the run dir's
> PREDECLARATION.md was applied as written, and recompute the mass table from the saved
> weights. For the probe: confirm the split is parcel-grouped (no parcel in both sides)
> and the quoted numbers match per_class_parcel_grouped.csv. Do not re-litigate settled
> findings. Never read fold 2. Ranked findings with re-derivation evidence, most severe
> first, and state explicitly which items you verified clean.

## 7. After delivery (not before)

Update memories: `stage2-subtype-mass` with the experiment's result, the weighted-run
memory's "next experiment" pointer, and a new memory if the probe rebuild changes the
spectral-ceiling framing. Queue for next week, in the report's own order: honest
hyperparameter retune (GroupKFold on parcel ID, scoring f1_macro, Stage 2 or orchards
alone), Δ-date features, parcel aggregation side analysis (approved by Nop 2026-08-26,
never executed — do not drop it), red-edge bands behind everything else.
