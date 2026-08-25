# Predeclaration — run `s2_2018_3date_parcel_weighted`

Written 2026-08-26, **before** the run was launched. Copied verbatim from
`docs/PLAN_2026-08-26_WEIGHTED_RUN.md` §3.

> Run `s2_2018_3date_parcel_weighted`, launched 2026-08-26. Identical to M5 except:
> tempered class weights w_i = sqrt(n_max/n_i) from post-cap training counts, applied to
> Stage 2 and all Stage-3 experts. Stage 1, caps, matrix, folds, calibration unchanged.
> Operating point: (α₂, α₃) selected on the calibration half by strict crop_macro_f1_flat
> over the 169-cell grid, scored once on the tune half. Gate: **iff** the tune-half score
> exceeds M5's tune-half score obtained by the same procedure, fold 2 is read once (hard
> routing, strict full-population convention) and becomes the report headline. Otherwise
> fold 2 is not read and the run is reported as a tune-half negative result. No other
> fold-2 access is permitted. The tune half is never used for selection.

## The gate baseline, fixed before this run existed

Produced by `sweep_operating_point.py` on `runs/s2_2018_3date_parcel_m5/`
(2026-08-26 01:5x, artefacts `opsweep.csv`, `opsweep_selection.json`):

| quantity | value |
|---|---|
| grid | 13 × 13, α ∈ {0.0 … 1.2} step 0.1 |
| selected on calibration half | α₂ = 0.3, α₃ = 0.6 (cal macro F1 0.2720) |
| **M5 tune-half macro F1 — the number to beat** | **0.2294** (10 of 13 crops ≥ 0.01) |
| M5 tune half at the fixed cell (0.2, 0.7) | 0.2283 |
| M5 tune half at plain argmax (0.0, 0.0) | 0.2140 (7 alive) |

The weighted run will be scored by the **same script, same grid, same halves, same
strict scorer**, invoked as `RUN_DIR=./runs/s2_2018_3date_parcel_weighted`.

## Launch command

```
SKIP_TEST=1 CLASS_WEIGHT=sqrt CROSSFIT_S1=3 P3_COMPONENTS=1200 \
NPZ_OVERRIDE=./aligned_features/svm_s2_3date_m3_features_labels.npz \
ARM_OUT=./runs/s2_2018_3date_parcel_weighted \
python -u train_parcel_cascade.py
```

Every one of those settings except `CLASS_WEIGHT=sqrt` is M5's. `SKIP_TEST=1` is how
the single-read discipline is enforced mechanically: the run cannot read fold 2 even
if someone wants it to. Reading fold 2 later, if the gate opens, is a separate scored
pass over the saved models.
