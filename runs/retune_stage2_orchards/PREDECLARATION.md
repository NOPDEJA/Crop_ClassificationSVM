# PREDECLARATION — E4 honest hyperparameter retune (Stage 2 + orchards expert)

Written before launch, per docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md E4.

**Script:** `e4_retune_stage2_orchards.py`.

**Search populations** (fold-0 / training-parcel data only, never touching
fold 1's tuning half or fold 2):
- Stage 2: `cap(cand_tr, g_of, 50_000)` — 50k/group, 200k total, drawn from the
  out-of-fold econ candidates in fold 0 (M5's saved `stage1_train_idx.npy` /
  `stage1_route_oof_train.npy`).
- Orchards expert: `cap(orchard rows in fold 0, y, PER_LU_CAP=70_000)` — the
  same natural capped fit set production already uses. No further subsample:
  the search population and the final-refit population are the same rows.

**CV:** `GroupKFold(3)` grouped by `parcel_id_row`, `scoring='f1_macro'` in
each component's own label space (Stage 2: 4 group labels + sink; orchards:
its 7 crops). `N_JOBS=1` throughout.

**Grid** (24 candidates per component): C ∈ {1, 10, 30}; gamma ∈
{0.25, 0.5, 1, 2} × the current rule's value (1/n_features); n_components ∈
{600, 1200} for Stage 2, {800, 1200} for the orchards expert.

**Gate G4 refit — paired against the current best-known config** (G3 passed,
so per the plan: "M5 + G3 weights, run them jointly"):
- **control:** `runs/s2_2018_3date_parcel_s2mass`'s already-gated Stage-2
  TREATMENT arm (production hyperparams, subtype-mass weights) + M5's
  original, untuned, unweighted orchards expert.
- **treatment:** Stage 2 refit on the IDENTICAL s2mass pool / calibration
  rows / subtype-mass weights (only the hyperparameters change, isolating the
  hyperparameter effect) + the orchards expert refit on its natural capped
  set at the winning hyperparameters, unweighted (E5, not E4, tests Stage-3
  weighting).
- Plantation and field experts: frozen from M5 in both arms.

**Carry rule (predeclared verbatim):** treatment tune macro F1 ≥ control +
0.002, AND the alive-crops guard (count of crops with F1 ≥ 0.01 must not
fall). Selection on the calibration half per arm, one tune read per arm — no
re-running after seeing the number.

**Fold 2:** untouched. Every population here derives from
`stage1_train_idx.npy` (fold 0) or `val_cal_idx.npy` (fold 1's calibration
half) only.

**Cost estimate (revised from the plan's "overnight"):** given today's
observed ~45-55 min per single 800k-row Stage-2 fit at n_components=600, the
144 CV-fold fits (72 per component, on smaller ~130-230k-row folds) are
expected to cost roughly 25-45h, not one overnight — this is a multi-day step.
