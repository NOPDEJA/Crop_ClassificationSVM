# PREDECLARATION — E3 pool-draw sensitivity of the s2mass +0.0081

Written before launch, per docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md E3.

**Script:** `s2mass_pool_sensitivity.py`.

**Design:** M5's rng sequence replayed exactly up to (not including) the
Stage-2 pool draw, verified by the same six checks `s2mass_stage2.py` uses.
For each of three predeclared seeds (1001, 1002, 1003), an INDEPENDENT
generator redraws the 200,000-per-group Stage-2 training pool only. The
calibration rows (`stage2_cal_idx.npy`) are NOT redrawn — reused verbatim from
`runs/s2_2018_3date_parcel_s2mass/`, so only the training pool varies across
draws and against the original s2mass result. Stage 1 and Stage 3 stay frozen
from M5 (hard-linked). Weights: `sqrt(m_max/m_c)` over post-cap within-group
subtype counts, renormalised per group — identical formula to `s2mass_stage2.py`.

**Read:** per-draw paired delta (treatment − control), at (a) each arm's own
cal-selected operating-point cell (`sweep_operating_point.py`'s own
selection), and (b) the fixed cell (0.3, 0.6). Report mean and range across
the three draws.

**Gate G3 (predeclared verbatim):** the subtype weights enter E7 iff
treatment ≥ control in **all three draws** (cal-selected reading) AND the
mean delta ≥ +0.002. Otherwise they stay out and that is reported, not
argued — no re-running after seeing the number.

**Fold 2:** untouched. Every artifact here derives from `splits/split_assign.npy`
folds 0/1 only, same as `s2mass_stage2.py`.

**Cost estimate:** ~2h (3 draws × a control+treatment pair, ~33 min of fitting
plus scoring each, per the original s2mass benchmark).
