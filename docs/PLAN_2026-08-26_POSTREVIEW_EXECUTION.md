# Post-review execution plan: mid-crop headroom, falsifiers, one fold-2 read

**Date:** 2026-08-26. **For:** a fresh Claude session (Sonnet 5, medium effort) executing this
plan end to end. **Provenance:** this is §4 of `docs/HANDOFF_CODEX_2026-08-26_CEILING_VERDICT.md`
after Codex review, in Codex's order, with full specifications. Read that handoff's §2 and §8
first if any "why" is unclear; do not re-derive or re-litigate its conclusions.

---

## 0. Context capsule — read this instead of the history

The model is the parcel-disjoint three-date S2-only SVM cascade. Current baseline **M5**
(`runs/s2_2018_3date_parcel_m5/`): strict test macro F1 **0.2344** (5,500,269 px), tune half
**0.2294**. The Stage-2 subtype-mass treatment (`runs/s2_2018_3date_parcel_s2mass/`) is a
measured winner: **+0.0081** on the tune half, all 169 operating-point cells favour it. Oracle
Stage-2 routing headroom remaining: +0.1452, concentrated in rice/cassava/durian/pineapple/mango.
The rare five (coconut, longan, langsat, rambutan, mangosteen) are not expected to reach F1 0.33
under strict scoring — that is the reviewed verdict; this plan does not chase it.

**Strict scoring convention** (always, unless a step says otherwise): full population, non-crop
truth mapped to 0, so crop false positives on non-crop ground count against precision. Scorer:
`sweep_operating_point.py` — selects the operating-point cell on the **calibration half**, reads
the **tuning half once**.

**Environment and habits:**
- Python: `C:\Conda_environment\envs\svm_env\python.exe`. 32 GB RAM → **serial runs only**.
- Long fits: launch detached and unbuffered, babysit per the established workflow (memory
  `long-training-workflow`); logs into the run directory.
- CSVs `encoding="utf-8-sig"`. New artifacts under `runs/<name>/`. Never touch `2018/`.
- Commit after each completed item (message names the item); do not push.
- Write `PREDECLARATION.md` into the run directory **before** launching anything it governs.

**The iron rule:** fold 2 (the test fold) is read **exactly once** in this plan, in E7, under
E7's predeclaration. No other step may load, score, or print anything derived from fold-2
predictions. Paired comparisons are always same-rows, same-code-path.

**Numeric gates** below use the house threshold **+0.002** tune-half macro F1 with an
alive-crops guard (count of crops with F1 ≥ 0.01 must not fall). Predeclare each gate verbatim
in the run's `PREDECLARATION.md` before launching.

---

## E1 — instrument `train_parcel_cascade.py` (hygiene; do first)

**Why:** M5 never saved its Stage-2 fit indices, which cost half a day of pool-identity argument.
Accepted follow-up in `docs/REPORT_2026-08-27.md` §8.1.

**Do:** add saving of `stage2_fit_idx.npy`, `stage2_cal_idx.npy`, and per-expert
`stage3_<group>_fit_idx.npy` / `stage3_<group>_cal_idx.npy` (row indices into the NPZ), plus a
SHA-256 of each array recorded in `manifest.json`.

**Constraints:** additive only. No change to any random draw, draw order, seed, or fitted
quantity — the diff must contain only new `np.save`/hashing lines and their imports. If saving
requires *moving* a computation, stop and reconsider; the arrays to save already exist at the
point the fit happens.

**Verify:** `git diff` shows additions only; E7's run writes the files.
**Cost:** ~30 min.

---

## E2 — fused-stack parcel-grouped probe (the falsifier; run second)

**Why:** the cheapest direct test of whether the rare-crop ceiling is *optical* (S2-only)
rather than *data* (parcels). Promoted to second by Codex.

**Data (facts verified 2026-08-26 — assert them again anyway):**
- `aligned_features/svm_dem_s1_s2_features_labels.npz`: keys `X, y, feature_names`; 153 named
  features; zero VVVH_DIFF columns; 24,323,769 rows.
- `aligned_features/_unpacked/X_src.npy` is the same matrix pre-extracted, (24,323,769 × 153)
  float32 — load with `np.load(..., mmap_mode='r')`; never decompress the 15 GB npz for X.
- Mandatory assertions before fitting: `y_fused` array-equal to the S2-only npz's `y`
  (`aligned_features/svm_s2_3date_features_labels.npz` — its `y` loads cheaply), and
  `len(parcel_id_row) == X.shape[0] == 24,323,769`. **If any assertion fails: stop and report;
  do not rebuild anything silently.**
- Do **not** use `svm_add_data_features_labels.npz` (no `feature_names`; superseded).

**Design — two arms on identical rows and identical split:**
- **Arm A (control):** the 40 S2-index columns (names starting `NDVI/EVI/NDWI/BSI/NDBI/MSAVI/SWIR`).
- **Arm B (treatment):** all 153 columns (adds 5 DEM + 108 S1 columns).
- Split: copy `probe_dry_season_grouped.py`'s crop-wise parcel halving exactly — same seed 42,
  parcels halved within each crop, disjointness asserted in code.
- Everything else copied from that script: uniform prior via its per-crop caps, median
  imputation, `StandardScaler`, Nyström 800 components, its gamma rule. **Gamma must follow the
  arm's own feature count** — apply the script's rule per arm, never reuse Arm A's gamma for
  Arm B (the gamma-scale lesson is written in blood in this repo).

**Predeclare, then read:** primary = per-class F1 (parcel split), B vs A on the same test rows.
**Falsifier threshold:** any of coconut / mangosteen / rambutan / longan gains ≥ +0.10 F1 in B →
record the **optical-ceiling branch**: the fused pipeline becomes the rare-class direction and
the verdict's V2(b) weakens. Report both directions and macro either way. Langsat is reported
but excluded from claims (8 test parcels). Artifacts → `runs/probe_fused_grouped/`.

**Cost:** ~1–2 h. **Note:** this is a probe on a 5-date fused matrix — its numbers are never
comparable to cascade numbers; they compare only Arm B to Arm A.

---

## E3 — pool-draw sensitivity of the +0.0081

**Why:** the s2mass design held the 200k-per-group draw fixed, so that variation is unmeasured;
it must be before the weights ride into E7.

**Do:** repeat the paired control/treatment Stage-2 fits of `s2mass_stage2.py` over **three
fresh draws** of the 200,000-rows-per-group cap — predeclared seeds **1001, 1002, 1003** for the
draw only. Everything else exactly as in `runs/s2_2018_3date_parcel_s2mass/`: Stage 1 and
Stage 3 arrays frozen from M5, the same calibration rows, weights `sqrt(m_max/m_c)` over
post-cap within-group subtype counts renormalised per group.

**Read:** per-draw paired delta (treatment − control) at the cal-selected cell and at fixed
(0.3, 0.6); report mean and range.

**Gate G3 (predeclare):** the subtype weights enter E7 iff treatment ≥ control in **all three
draws** and the mean delta ≥ +0.002. Otherwise they stay out and that is reported, not argued.

**Cost:** ~2 h (a control+treatment pair is ~33 min of fitting plus scoring).
Artifacts → `runs/s2mass_pool_sensitivity/draw_<seed>/`.

---

## E4 — honest hyperparameter retune (Stage 2 and the orchards expert)

**Why:** current values come from a pixel-leaky search scored on accuracy — the largest known
validity defect still standing (report §8.3). Every prior search hit its capacity ceiling, so
capacity may be undersized.

**Search protocol:**
- Data: fold-1 **training** parcels only. CV: `GroupKFold(3)` grouped by `parcel_id_row`;
  `scoring='f1_macro'` in the component's own label space (Stage 2: its four group labels plus
  sink, exactly as `train_parcel_cascade.py` constructs them; orchards expert: its seven crops).
- Subsample for the search to keep it overnight-sized: Stage 2 on 50,000 rows per group
  (200,000 total, drawn with the existing seeded sampler); the orchards expert on its natural
  capped fit set.
- Grid (≤ 24 candidates per component, joint as the literature requires): C ∈ {1, 10, 30};
  gamma ∈ {0.25, 0.5, 1, 2} × the current rule's value; components ∈ {600, 1200} for Stage 2,
  {800, 1200} for the expert. Log every cell's CV score to a CSV.

**Gate G4:** refit the winning configuration at full size and swap it into the frozen cascade
(same paired machinery as s2mass — everything else M5, or M5+G3 weights if G3 passed and you
run them jointly; state which in the predeclaration). Cal-select, single tune read. Carry iff
≥ paired control + 0.002 with the alive-crops guard.

**Cost:** overnight. Artifacts → `runs/retune_stage2_orchards/`.

---

## E5 — Stage-3 subtype mass, factorial against the E4 winner

**Why:** coconut's Stage-2 routing doubled (13.7 % → 26.6 %) while its F1 stayed at 0.001 — the
binding constraint moved into the Stage-3 plantation expert. This is the direct test.

**Do:** the s2mass paired design *inside* the plantation and orchards experts: per-row weight
`sqrt(m_max/m_c)` over post-cap within-expert crop counts, renormalised so each expert's total
mass is unchanged. Stage 1 and Stage 2 frozen — use the E4 winner if G4 passed, else M5's
(factorial against the winner, per review). Control = unweighted refit of the same experts on
the same rows.

**Gate G5:** same form as G3. **Watch coconut specifically** — it is the crop this experiment
exists for; report its routing-vs-F1 pair either way.

**Cost:** ~1–2 h. Artifacts → `runs/s3mass_experts/`.

---

## E6 — date-difference features (demoted; optional)

Review verdict: Oct−Nov / Nov−Dec index deltas are deterministic linear combinations of existing
columns — no new information, only a change of feature scaling and kernel metric. Treat as a
**feature-weighting experiment** with low expected value. Run only if E2–E5 complete with time
to spare or the user asks: parcel-grouped expert-level probe first, gate ≥ +0.01 probe macro
before it may touch the cascade.

---

## E7 — one consolidated cascade and the single fold-2 read

**Config:** M5 plus every component that passed its gate (G3 subtype weights, G4
hyperparameters, G5 expert weights), trained with the E1-instrumented script. If *nothing*
passed, stop and report — do not spend the fold-2 read on a config identical to M5.

**Predeclare before launch** in `runs/<final_name>/PREDECLARATION.md`:
- the exact configuration and which gates admitted each component;
- the read rule: select the operating-point cell on fold 1's calibration half, score fold 2
  **once** with the strict scorer at that cell — no second read under any outcome;
- the planning scenario being tested: base 0.24–0.26, upside 0.28 (a scenario, not a forecast —
  report whatever lands, including a loss);
- abort conditions (crash → fix and relaunch is allowed; peeking is not).

**After the read:** run the `/regression-check` skill against M5's report; produce the per-crop
F1 table baseline-vs-M5-vs-final; update `docs/` (a short dated report) and the memory files
(`ceiling-verdict-2026-08-26`, `stage2-subtype-mass`) with the outcome; commit.

**Cost:** ~3.5 h training + scoring.

---

## Reporting tasks (during lulls or after E7)

1. **`docs/S2_SVM_ANALYSIS.md`:** rewrite the RF-paper comparison per the sustained wording —
   *"Our experiment demonstrates severe protocol sensitivity on our dataset. The RF paper's
   generalization performance remains unaudited because its split cannot be reconstructed
   here."* Move the `matched` table into an appendix / protocol-sensitivity subsection; it must
   not sit beside honest scores as equivalent evidence. While there, soften any claim that the
   0.048 parcel-half gap is "the sampling variance" — it is one observed difference.
2. **`docs/REPORT_2026-08-27.md` stays untouched** — it is the meeting record.
3. Note for the user to raise with the professor (not an agent task): the only falsifier that
   can rescue the rare five is **more parcels** — the 2020/2024 LDD surveys for 47PQQ or a
   neighbouring tile.

---

## Guardrails, restated

- Fold 2: once, in E7, predeclared. Nothing else touches it — check any script you reuse for
  what it does with `split_assign` before running it.
- Predeclaration before launch, selection on the calibration half, one tune read per gate.
- Serial execution; detached launches for anything over ~20 min; babysit, don't poll.
- If an assertion or gate fails, that is a *result* — record it and move on; never tune a gate
  after seeing its number.
- Deliberately out of scope (settled, do not revisit): pixel SMOTE, equalize-to-Langsat,
  per-crop manual thresholds, soft probability-product routing, tree merge, any second fold-2
  read, joint tables with the collaborator before a shared protocol is agreed.

## Order and budget

| step | what | cost | gate |
|---|---|---|---|
| E1 | instrument the trainer | 30 min | diff review |
| E2 | fused-stack grouped probe | 1–2 h | falsifier read, no carry |
| E3 | pool-draw sensitivity ×3 | ~2 h | G3 |
| E4 | honest retune, 2 components | overnight | G4 |
| E5 | Stage-3 subtype mass | 1–2 h | G5 |
| E6 | (optional) date differences | — | probe gate |
| E7 | consolidated cascade + the read | ~4 h | predeclared |

Total: roughly two to three working days, serial.
