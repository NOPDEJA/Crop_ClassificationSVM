# PREDECLARATION — E7 final consolidated cascade, the single fold-2 read

Written before launch, per docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md E7.
This is the ONE read of fold 2 for this entire plan. No other step (E1-E5)
has loaded, scored, or printed anything derived from fold-2 predictions.

**Script:** `e7_final_cascade_fold2_read.py`.

**Exact configuration, and which gate admitted each component:**
| component | source | gate |
|---|---|---|
| Stage 1 | M5's frozen model, unchanged | never touched by any experiment |
| Stage 2 | E4's retuned model (n_components=1200, gamma=0.5x rule, C=30), fit on s2mass's exact pool/calibration rows with subtype-mass weights | **G3 PASS** (weights, mean delta +0.0078 across 3 pool draws) baked into the fit; **G4 PASS** (hyperparameters, delta +0.0041) on top |
| Orchards expert | E4's retuned model, same hyperparameters as Stage 2, UNWEIGHTED | **G4 PASS**; G5's weighted version (runs/s3mass_experts/treatment) is NOT used — **G5 FAILED** (delta -0.0012, alive-crops guard failed) |
| Plantation expert | M5's frozen, original model, unchanged | **G5 FAILED** — no change carried |
| Field expert | M5's frozen, original model, unchanged | never touched by any experiment |

**The read rule (mechanical, no discretion):**
1. Build a virtual fold-1 validation directory from already-fit prob_val
   arrays (E4's Stage-2 + orchards; M5's Stage-1/plantation/field) and run
   `sweep_operating_point.py` on it to select the operating-point cell
   (alpha2, alpha3) that maximises macro F1 on fold 1's CALIBRATION half.
   Fold 2 is not touched by this step.
2. Predict E4's Stage-2 and orchards models on fold 2's candidates (M5's
   frozen `stage2_test_idx.npy` — Stage 1 is untouched, so the candidate set
   is exactly M5's). Reuse M5's frozen plantation/field test predictions
   directly — irreversible from this point.
3. Compose with the SAME hard-rule construction `train_parcel_cascade.py`
   uses (ratio2/ratio3 denominators from the calibration population's
   prior), at the cal-selected (alpha2, alpha3) from step 1.
4. Score fold 2 **once** with the strict convention (full population,
   non-crop truth mapped to 0, all 13 crop labels). **No second read under
   any outcome.**

**The planning scenario being tested** (a scenario, not a forecast — report
whatever lands, including a loss against M5's 0.2344): base **0.24–0.26**,
upside **0.28**. Calibration: the whole prior year of work bought +0.0096;
the best single tune-half intervention (s2mass) bought +0.0081.

**Abort conditions:** a crash before step 2 begins (fold 2 still untouched)
→ fix and relaunch is allowed. A crash mid-step-2/3/4, after fold 2
predictions have been loaded → do not relaunch a variant; report the crash
and the state fold 2 was read in, and treat any completed score from that
run as the one read. Peeking (loading fold-2 arrays outside this script, or
re-running this script with a different config after seeing a number) is
never allowed.

**After the read:** run `/regression-check` against M5's report; produce the
per-crop F1 table baseline-vs-M5-vs-final; update `docs/` (a short dated
report) and the memory files (`ceiling-verdict-2026-08-26`,
`stage2-subtype-mass`) with the outcome; commit.
